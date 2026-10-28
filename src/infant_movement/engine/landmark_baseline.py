"""Bounded ordinary CPU training, validation, checkpoints and private run records."""

import hashlib
import json
import math
import random
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from infant_movement.datasets.synthetic_landmarks import SyntheticLandmarks
from infant_movement.datasets.torch_landmarks import decode_coordinates
from infant_movement.evaluation.landmark_metrics import landmark_errors
from infant_movement.models.landmark_baseline import SmallLandmarkCNN


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 120
    batch_size: int = 8
    learning_rate: float = 0.003
    seed: int = 42
    train_count: int = 32
    validation_count: int = 16
    device: str = "cpu"

    def __post_init__(self):
        for name, low, high in (
            ("epochs", 1, 200),
            ("batch_size", 1, 32),
            ("train_count", 1, 64),
            ("validation_count", 1, 64),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} is outside the bounded synthetic configuration")
        if (
            type(self.seed) is not int
            or not 0 <= self.seed < 2**32
            or type(self.learning_rate) not in (int, float)
            or not math.isfinite(self.learning_rate)
            or not 0 < self.learning_rate <= 0.1
            or self.device != "cpu"
        ):
            raise ValueError("baseline verification requires an explicit valid CPU configuration")


def masked_coordinate_mse(prediction, target, valid):
    """Mean squared error over eligible x/y scalars; empty supervision -> None.

    Index before subtraction so masked NaNs cannot contaminate the loss/gradient.
    Bad predictions or nonfinite eligible targets fail rather than reducing coverage.
    """
    if (
        prediction.ndim != 3
        or prediction.shape[-1] != 2
        or target.shape != prediction.shape
        or valid.shape != prediction.shape[:2]
        or valid.dtype != torch.bool
        or not prediction.is_floating_point()
        or not target.is_floating_point()
    ):
        raise ValueError("loss requires matching floating [B,K,2] and bool [B,K] tensors")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target[valid]).all():
        raise ValueError("predictions and eligible targets must be finite")
    count = int(valid.sum().item())
    return (((prediction[valid] - target[valid]) ** 2).mean() if count else None), count


def run_epoch(model, loader, *, optimizer=None):
    """Reusable loader loop. CPU only; no loader means no implicit real-data access."""
    model.train(optimizer is not None)
    total = point_count = batches = skipped = steps = 0
    predictions, targets, masks = [], [], []
    for batch in loader:
        batches += 1
        if batch["image"].device.type != "cpu":
            raise ValueError("this milestone verifies CPU correctness only")
        with torch.set_grad_enabled(optimizer is not None):
            prediction = model(batch["image"])
            loss, count = masked_coordinate_mse(prediction, batch["target"], batch["valid"])
            if count:
                total += float(loss.detach()) * count
                point_count += count
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    steps += 1
            else:
                skipped += 1
        # Metrics use raw source-coordinate units via each item's recorded inverse.
        raw = decode_coordinates(prediction.detach(), (32, 32), batch["inverse"])
        predictions.extend(raw.tolist())
        targets.extend(batch["source_landmarks"].tolist())
        masks.extend(batch["valid"].tolist())
    if not batches:
        raise ValueError("epoch loader contains no batches")
    metrics = landmark_errors(predictions, targets, masks)
    return {
        "loss": total / point_count if point_count else None,
        "valid_points": point_count,
        "batches": batches,
        "skipped_batches": skipped,
        "optimizer_steps": steps,
        "source_pixel_metrics": metrics,
    }


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_checkpoint(path, model, optimizer, *, epoch, config, loader_generator):
    """Local state dictionaries and primitive metadata, never a pickled module."""
    destination = Path(path)
    payload = {
        "schema_version": 1,
        "architecture": "small-landmark-cnn-v1",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "config": asdict(config),
        "torch_version": str(torch.__version__),
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
        "loader_rng": loader_generator.get_state(),
        "target_contract": "68 normalized pixel-edge pairs; x/W,y/H; no clamping",
    }
    temporary = destination.with_suffix(".pending")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return sha256(destination)


def load_checkpoint(path, *, expected_sha256, restore_rng=False):
    """Load only a locally generated, digest-checked CPU checkpoint, weights_only=True.

    Returns model, restored Adam optimizer and primitive/tensor payload. Restoring
    loader RNG remains explicit to callers; bit-exact training resume is not claimed.
    """
    path = Path(path)
    if (
        not isinstance(expected_sha256, str)
        or not re.fullmatch("[0-9a-f]{64}", expected_sha256)
        or path.stat().st_size > 32 * 1024**2
        or sha256(path) != expected_sha256
    ):
        raise ValueError("checkpoint digest/size check failed before deserialization")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1 or payload.get("architecture") != "small-landmark-cnn-v1":
        raise ValueError("unsupported local checkpoint contract")
    config = TrainConfig(**payload["config"])
    model = SmallLandmarkCNN()
    model.load_state_dict(payload["model"], strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer.load_state_dict(payload["optimizer"])
    if restore_rng:
        torch.set_rng_state(payload["torch_rng"])
        random.setstate(payload["python_rng"])
    model.eval()
    return model, optimizer, payload


def fit(model, train_loader, validation_loader, *, optimizer, config, run_dir, loader_generator):
    """Ordinary bounded loop; run_dir must be a new owned directory created by caller."""
    path = Path(run_dir)
    initial = run_epoch(model, validation_loader)
    if initial["loss"] is None:
        raise ValueError("validation has no eligible targets")
    best_loss, best_epoch, expected_prediction = math.inf, 0, None
    probe = torch.stack(
        [
            validation_loader.dataset[index]["image"]
            for index in range(len(validation_loader.dataset))
        ]
    )
    history = []
    with (path / "epochs.jsonl").open("x") as log:
        for epoch in range(1, config.epochs + 1):
            training = run_epoch(model, train_loader, optimizer=optimizer)
            if training["loss"] is None:
                raise ValueError("training epoch has no eligible targets; no optimizer steps taken")
            validation = run_epoch(model, validation_loader)
            if validation["loss"] is None:
                raise ValueError("validation has no eligible targets")
            row = {"epoch": epoch, "training": training, "validation": validation}
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            history.append(row)
            if validation["loss"] < best_loss:
                best_loss, best_epoch = validation["loss"], epoch
                save_checkpoint(
                    path / "best.pt",
                    model,
                    optimizer,
                    epoch=epoch,
                    config=config,
                    loader_generator=loader_generator,
                )
                with torch.no_grad():
                    expected_prediction = model(probe).clone()
    save_checkpoint(
        path / "last.pt",
        model,
        optimizer,
        epoch=config.epochs,
        config=config,
        loader_generator=loader_generator,
    )
    best_digest = sha256(path / "best.pt")
    restored, _, _ = load_checkpoint(path / "best.pt", expected_sha256=best_digest)
    with torch.no_grad():
        restored_prediction = restored(probe)
    validation = run_epoch(restored, validation_loader)
    return {
        "initial_validation": initial,
        "best_validation": validation,
        "best_epoch": best_epoch,
        "last_epoch": config.epochs,
        "training_steps": sum(row["training"]["optimizer_steps"] for row in history),
        "best_checkpoint_sha256": best_digest,
        "reload_predictions_equal": torch.equal(expected_prediction, restored_prediction),
        "reload_max_absolute_difference": float(
            (expected_prediction - restored_prediction).abs().max()
        ),
        "reload_example_count": len(probe),
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }


def _provenance(root):
    def git(*args):
        result = subprocess.run(["git", *args], cwd=root, capture_output=True, check=False)
        return result.stdout

    files = list((root / "src").rglob("*.py")) + list((root / "tests").glob("*.py"))
    files += list((root / "configs/experiments").glob("*.json"))
    files += [root / "environments/macos-arm64-py313-baseline.lock"]
    return {
        "git_head": git("rev-parse", "HEAD").decode().strip(),
        "git_branch": git("branch", "--show-current").decode().strip(),
        "git_dirty": bool(git("status", "--porcelain")),
        "tracked_diff_sha256": hashlib.sha256(git("diff", "--binary", "HEAD")).hexdigest(),
        "source_config_hashes": {str(p.relative_to(root)): sha256(p) for p in files if p.is_file()},
    }


def run_synthetic_baseline(config: TrainConfig, run_dir, *, project_root):
    """The only runnable training entry point: manufactured data, CPU, no archive argument."""
    import platform
    import sys

    root, path = Path(project_root).resolve(), Path(run_dir).resolve()
    if path == root / "runs" or not path.is_relative_to(root / "runs"):
        raise ValueError("run outputs must be below the project ignored runs directory")
    path.mkdir(parents=True, exist_ok=False)
    old_rng, old_python_rng = torch.get_rng_state(), random.getstate()
    old_threads, old_deterministic = (
        torch.get_num_threads(),
        torch.are_deterministic_algorithms_enabled(),
    )
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        training = SyntheticLandmarks(config.train_count, seed=config.seed)
        validation = SyntheticLandmarks(config.validation_count, seed=config.seed + 1)
        generator = torch.Generator().manual_seed(config.seed + 2)
        train_loader = DataLoader(
            training, batch_size=config.batch_size, shuffle=True, generator=generator, num_workers=0
        )
        validation_loader = DataLoader(validation, batch_size=config.batch_size, num_workers=0)
        model = SmallLandmarkCNN().cpu()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        (path / "config.json").write_text(json.dumps(asdict(config), indent=2))
        provenance = _provenance(root)
        provenance.update(
            {
                "data_kind": "manufactured-stripes-ellipse-v1",
                "real_images_used": 0,
                "train_seed": config.seed,
                "validation_seed": config.seed + 1,
                "train_data_sha256": hashlib.sha256(
                    json.dumps(
                        [
                            {k: v.tolist() for k, v in training[i].items()}
                            for i in range(len(training))
                        ]
                    ).encode()
                ).hexdigest(),
                "validation_data_sha256": hashlib.sha256(
                    json.dumps(
                        [
                            {k: v.tolist() for k, v in validation[i].items()}
                            for i in range(len(validation))
                        ]
                    ).encode()
                ).hexdigest(),
                "python": sys.version,
                "torch": str(torch.__version__),
                "platform": platform.platform(),
                "device": "cpu",
                "threads": 1,
                "deterministic_algorithms": True,
                "pretrained_weights": None,
                "input_target_contract": {
                    "architecture": "small-landmark-cnn-v1",
                    "image_shape_chw": [3, 32, 32],
                    "dtype": "float32",
                    "range": [0, 1],
                    "target": "68 pairs of normalized pixel-edge x/W,y/H",
                    "source_frame": "manufactured base-0 pixel centres; inverse subtracts 0.5",
                    "validity": "explicit bool mask; manufactured examples all valid",
                    "num_workers": 0,
                    "shuffle_training": True,
                    "loader_seed": config.seed + 2,
                    "optimizer": "Adam",
                    "validation_selection": "lowest normalized coordinate MSE",
                    "metrics": "source-pixel image-macro Euclidean error; no anatomical NME",
                },
            }
        )
        (path / "provenance.json").write_text(json.dumps(provenance, indent=2))
        freeze = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "--disable-pip-version-check",
                "--no-cache-dir",
                "freeze",
                "--all",
            ],
            capture_output=True,
            check=True,
        )
        (path / "environment.txt").write_bytes(freeze.stdout)
        summary = fit(
            model,
            train_loader,
            validation_loader,
            optimizer=optimizer,
            config=config,
            run_dir=path,
            loader_generator=generator,
        )
        initial = summary["initial_validation"]["source_pixel_metrics"]["mean_error"]
        final = summary["best_validation"]["source_pixel_metrics"]["mean_error"]
        summary["synthetic_only"] = True
        summary["acceptance"] = {
            "predeclared": "best validation pixel error <1 and <10% of initial; exact reload",
            "passed": final < 1.0 and final < 0.1 * initial and summary["reload_predictions_equal"],
        }
        (path / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
        (path / "status.json").write_text(json.dumps({"status": "complete"}))
        return summary
    except Exception as error:
        (path / "status.json").write_text(
            json.dumps({"status": "failed", "error_type": type(error).__name__})
        )
        raise
    finally:
        torch.set_rng_state(old_rng)
        random.setstate(old_python_rng)
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_deterministic)
