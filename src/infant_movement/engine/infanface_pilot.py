"""One explicitly reviewed, bounded training-only InfAnFace optimization pilot."""

import json
import platform
import random
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from infant_movement.datasets.infanface_images import preprocess_face
from infant_movement.datasets.infanface_pilot import (
    PilotConfig,
    PilotError,
    canonical_hash,
    ground_truth_crop,
    open_training_images,
    require_training,
    select_training_samples,
)
from infant_movement.datasets.torch_landmarks import decode_coordinates, prepared_to_tensors
from infant_movement.engine.landmark_baseline import _provenance, masked_coordinate_mse, sha256
from infant_movement.evaluation.landmark_metrics import landmark_errors
from infant_movement.models.landmark_baseline import SmallLandmarkCNN


def runtime_hashes(root, protocol):
    root, protocol = Path(root), Path(protocol)
    paths = list((root / "src").rglob("*.py")) + list((root / "scripts").glob("*.py"))
    paths += list((root / "tests").glob("*.py"))
    paths += [
        root / "configs/experiments/infanface_train_pilot.json",
        root / "environments/macos-arm64-py313-baseline.lock",
        protocol,
    ]
    return {str(p.resolve()): sha256(p) for p in paths if p.is_file()}


def _new_json(path, content):
    with Path(path).open("x") as stream:
        json.dump(content, stream, indent=2, allow_nan=False)


def _status(path, status, **fields):
    # Only the current run's status is updated; scientific input/evidence stays immutable.
    (Path(path) / "status.json").write_text(
        json.dumps(
            {"status": status, "recorded_at_utc": datetime.now(UTC).isoformat(), **fields}, indent=2
        )
    )


def _run_path(run_dir, root):
    root, path = Path(root).resolve(), Path(run_dir).resolve()
    if path == root / "runs" or not path.is_relative_to(root / "runs"):
        raise PilotError("pilot outputs must be inside the ignored project runs directory")
    return root, path


def _tensor_example(face, box, config):
    prepared = preprocess_face(
        face,
        source_index_base=config.source_index_base,
        annotation_frame=config.annotation_frame,
        orientation_policy=config.orientation_policy,
        crop_box=tuple(box),
        output_size=config.input_size,
    )
    try:
        result = prepared_to_tensors(prepared)
        result["crop_kind"] = "ground-truth-derived-v1"
        result["crop_box"] = torch.tensor(box, dtype=torch.int64)
        return result
    finally:
        prepared.image.close()


def _qc_tile(face, box, ordinal):
    """Annotation-only display of a source crop; never a model input/prediction."""
    from PIL import Image, ImageDraw

    with face.image.crop(tuple(box)) as crop:
        crop.thumbnail((224, 224), Image.Resampling.BICUBIC)
        tile = Image.new("RGB", (256, 272), "white")
        dx, dy = (256 - crop.width) // 2, 30 + (224 - crop.height) // 2
        tile.paste(crop, (dx, dy))
        draw = ImageDraw.Draw(tile)
        draw.text((4, 5), f"training selection {ordinal}; GT crop", fill="black")
        sx, sy = crop.width / (box[2] - box[0]), crop.height / (box[3] - box[1])
        for point in face.sample.landmarks:
            if point is not None:
                x, y = dx + (point[0] + 0.5 - box[0]) * sx, dy + (point[1] + 0.5 - box[1]) * sy
                draw.ellipse((x - 1.4, y - 1.4, x + 1.4, y + 1.4), fill=(255, 20, 20))
        return tile


def prepare_pilot(config, archive_path, expected_sha256, run_dir, *, project_root, protocol_path):
    """No model is instantiated. Writes fixed private selection, coverage and QC panels."""
    from PIL import Image

    root, path = _run_path(run_dir, project_root)
    protocol = Path(protocol_path).resolve()
    if not protocol.is_file():
        raise PilotError("a written pre-execution protocol is required")
    path.mkdir(parents=True, exist_ok=False)
    _status(path, "preparing", execution_started=False)
    try:
        _new_json(path / "config.json", asdict(config))
        records, tiles, orientations, mask_counts = [], [], Counter(), Counter()
        decoded = 0
        with open_training_images(archive_path, expected_sha256=expected_sha256) as images:
            selected = select_training_samples(images.samples, config)
            refs = [
                {
                    "image_set": s.image_set,
                    "filename": s.filename,
                    "source_row": s.source_row,
                    "official_partition": s.official_split,
                }
                for s in selected
            ]
            selection_sha = canonical_hash(refs)
            for ordinal, sample in enumerate(selected, 1):
                record = {
                    **refs[ordinal - 1],
                    "subject_id": None,
                    "source_video_id": None,
                    "source_group_id": None,
                    "project_split": None,
                    "annotation_sha256": canonical_hash(sample.landmarks),
                    "eligible": False,
                }
                face = None
                try:
                    face = images.load(sample)
                    decoded += 1
                    orientations[str(face.orientation)] += 1
                    record["exif_orientation"] = face.orientation
                    record["source_size_wh"] = face.image.size
                    box = ground_truth_crop(
                        sample.landmarks,
                        face.image.size,
                        source_index_base=config.source_index_base,
                        margin=config.crop_margin,
                    )
                    example = _tensor_example(face, box, config)
                    record["crop_box"] = box
                    record["valid_points"] = int(example["valid"].sum())
                    record["mask_false_counts"] = {
                        key: int((~example[key]).sum())
                        for key in (
                            "present",
                            "source_in_bounds",
                            "crop_in_bounds",
                            "output_in_bounds",
                        )
                    }
                    mask_counts.update(record["mask_false_counts"])
                    if not example["valid"].any():
                        raise PilotError("no eligible target slots")
                    record["eligible"] = True
                    tiles.append(_qc_tile(face, box, ordinal))
                except (ValueError, OSError) as error:
                    record["exclusion_code"] = type(error).__name__
                    record["exclusion_reason"] = str(error)
                finally:
                    if face is not None:
                        face.image.close()
                records.append(record)
            pool_count = len(images.samples)
        for page in range((len(tiles) + 15) // 16):
            with Image.new("RGB", (1024, 1088), "white") as sheet:
                for index, tile in enumerate(tiles[page * 16 : page * 16 + 16]):
                    sheet.paste(tile, ((index % 4) * 256, (index // 4) * 272))
                sheet.save(path / f"qc-page-{page + 1:02d}.png")
        for tile in tiles:
            tile.close()
        usable = sum(r["eligible"] for r in records)
        if not usable:
            raise PilotError("no usable selected training example; model execution forbidden")
        manifest = {"selection_sha256": selection_sha, "records": records}
        _new_json(path / "selection.json", manifest)
        coverage = {
            "official_training_pool": pool_count,
            "selected_images": len(records),
            "decoded_images": decoded,
            "usable_images": usable,
            "excluded_images": len(records) - usable,
            "selected_point_slots": len(records) * 68,
            "eligible_point_slots": sum(r.get("valid_points", 0) for r in records if r["eligible"]),
            "mask_false_counts_nonexclusive": dict(mask_counts),
            "exclusion_counts": dict(
                Counter(r["exclusion_code"] for r in records if not r["eligible"])
            ),
            "exif_orientation_counts": dict(orientations),
            "test_images_opened": 0,
            "project_splits_created": False,
        }
        _new_json(path / "coverage.json", coverage)
        metadata = {
            "archive_path": str(Path(archive_path).resolve()),
            "archive_sha256": expected_sha256,
            "selection_sha256": selection_sha,
            "selection_file_sha256": sha256(path / "selection.json"),
            "coverage_sha256": sha256(path / "coverage.json"),
            "config_sha256": sha256(path / "config.json"),
            "protocol_path": str(protocol),
            "protocol_sha256": sha256(protocol),
            "runtime_hashes": runtime_hashes(root, protocol),
            "coordinate_evidence": "UNVERIFIED",
            "experiment_kind": "ground-truth-crop training-only fit",
        }
        _new_json(path / "prepared.json", metadata)
        _new_json(
            path / "alignment-review-template.json",
            {
                "selection_sha256": selection_sha,
                "selection_file_sha256": metadata["selection_file_sha256"],
                "approved": False,
                "gross_alignment_supported": False,
                "reviewed_usable_images": 0,
                "reviewer": "",
                "source_origin": "UNVERIFIED",
                "global_annotation_frame": "UNVERIFIED",
            },
        )
        _status(path, "prepared", execution_started=False)
        return coverage
    except Exception as error:
        _status(
            path, "failed-preparation", execution_started=False, error_type=type(error).__name__
        )
        raise


class PreparedTrainingExamples(Dataset):
    """Only explicit selected training items; no validation aliases or project assignments."""

    def __init__(self, samples, examples):
        require_training(samples)
        if len(samples) != len(examples) or any(
            e["official_partition"] != "infanface-train" for e in examples
        ):
            raise PilotError("non-training or mismatched prepared pilot items")
        self.samples, self.examples = tuple(samples), tuple(examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return {
            key: (value.clone() if isinstance(value, torch.Tensor) else value)
            for key, value in self.examples[index].items()
        }


def pilot_epoch(model, loader, optimizer):
    model.train()
    total = points = steps = skipped = 0
    for batch in loader:
        if batch["image"].device.type != "cpu":
            raise PilotError("pilot is CPU-only")
        prediction = model(batch["image"])
        loss, count = masked_coordinate_mse(prediction, batch["target"], batch["valid"])
        if loss is None:
            skipped += 1
            continue
        if not torch.isfinite(loss):
            raise PilotError("nonfinite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise PilotError("nonfinite training gradient")
        optimizer.step()
        total += float(loss.detach()) * count
        points += count
        steps += 1
    if not points:
        raise PilotError("training epoch has no eligible target slots")
    return {
        "optimization_loss": total / points,
        "eligible_points_seen": points,
        "optimizer_steps": steps,
        "skipped_batches": skipped,
    }


def training_fit(model, loader):
    model.eval()
    input_pred, input_target, source_pred, source_target, masks, normalized = [], [], [], [], [], []
    total = points = 0
    with torch.no_grad():
        for batch in loader:
            prediction = model(batch["image"])
            loss, count = masked_coordinate_mse(prediction, batch["target"], batch["valid"])
            if loss is not None:
                if not torch.isfinite(loss):
                    raise PilotError("nonfinite training-set evaluation loss")
                total += float(loss) * count
                points += count
            input_pred.extend((prediction.double() * 32).tolist())
            input_target.extend((batch["target"].double() * 32).tolist())
            source_pred.extend(decode_coordinates(prediction, (32, 32), batch["inverse"]).tolist())
            source_target.extend(batch["source_landmarks"].tolist())
            masks.extend(batch["valid"].tolist())
            normalized.append(prediction.clone())
    if not points:
        raise PilotError("training-set evaluation has no eligible targets")
    return {
        "normalized_coordinate_mse": total / points,
        "input_pixel_error": landmark_errors(input_pred, input_target, masks),
        "source_pixel_error": landmark_errors(source_pred, source_target, masks),
        "aggregation": "image-macro training-set fit; no held-out evaluation",
    }, torch.cat(normalized)


def _validate_execution(path, root, verification_path):
    status = json.loads((path / "status.json").read_text())
    if status.get("status") != "prepared" or status.get("execution_started") is not False:
        raise PilotError("pilot has already started or is not prepared; rerunning is forbidden")
    metadata = json.loads((path / "prepared.json").read_text())
    for name, key in (
        ("selection.json", "selection_file_sha256"),
        ("coverage.json", "coverage_sha256"),
        ("config.json", "config_sha256"),
    ):
        if sha256(path / name) != metadata[key]:
            raise PilotError("prepared pilot input hash mismatch")
    if (
        sha256(metadata["protocol_path"]) != metadata["protocol_sha256"]
        or runtime_hashes(root, metadata["protocol_path"]) != metadata["runtime_hashes"]
    ):
        raise PilotError("protocol or software changed after preparation")
    verification = json.loads(Path(verification_path).read_text())
    if (
        verification.get("checks_passed") is not True
        or verification.get("runtime_hashes") != metadata["runtime_hashes"]
        or not verification.get("commands")
        or any(c.get("returncode") != 0 for c in verification["commands"])
    ):
        raise PilotError("passing synthetic software checks for this code are required")
    review = json.loads((path / "alignment-review.json").read_text())
    coverage = json.loads((path / "coverage.json").read_text())
    if (
        review.get("approved") is not True
        or review.get("gross_alignment_supported") is not True
        or review.get("selection_sha256") != metadata["selection_sha256"]
        or review.get("selection_file_sha256") != metadata["selection_file_sha256"]
        or type(review.get("reviewed_usable_images")) is not int
        or review["reviewed_usable_images"] != coverage["usable_images"]
        or not review.get("reviewer")
        or review.get("source_origin") != "UNVERIFIED"
        or review.get("global_annotation_frame") != "UNVERIFIED"
    ):
        raise PilotError("all usable training images need a supported, recorded alignment review")
    return PilotConfig(**json.loads((path / "config.json").read_text())), metadata, coverage


def load_final_pilot_checkpoint(path, *, expected_sha256):
    path = Path(path)
    if path.stat().st_size > 32 * 1024**2 or sha256(path) != expected_sha256:
        raise PilotError("local pilot checkpoint digest/size mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("schema_version") != 1
        or payload.get("experiment") != "infanface-train-fit-v1"
        or payload.get("architecture") != "small-landmark-cnn-v1"
        or payload.get("checkpoint_policy") != "fixed-final-epoch"
    ):
        raise PilotError("unsupported pilot checkpoint contract")
    config = PilotConfig(**payload["config"])
    model = SmallLandmarkCNN().float().cpu()
    model.load_state_dict(payload["model"], strict=True)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
    )
    optimizer.load_state_dict(payload["optimizer"])
    model.eval()
    return model, optimizer, payload


def execute_pilot(run_dir, *, project_root, verification_path):
    root, path = _run_path(run_dir, project_root)
    config, metadata, coverage = _validate_execution(path, root, verification_path)
    _status(path, "running", execution_started=True)
    old_rng, old_python = torch.get_rng_state(), random.getstate()
    old_threads, old_deterministic = (
        torch.get_num_threads(),
        torch.are_deterministic_algorithms_enabled(),
    )
    steps = 0
    try:
        manifest = json.loads((path / "selection.json").read_text())
        samples, examples = [], []
        with open_training_images(
            metadata["archive_path"], expected_sha256=metadata["archive_sha256"]
        ) as images:
            selected = select_training_samples(images.samples, config)
            refs = [
                {
                    "image_set": s.image_set,
                    "filename": s.filename,
                    "source_row": s.source_row,
                    "official_partition": s.official_split,
                }
                for s in selected
            ]
            if canonical_hash(refs) != metadata["selection_sha256"]:
                raise PilotError("selected training references changed")
            for sample, record in zip(selected, manifest["records"], strict=True):
                require_training((sample,))
                if (
                    record["official_partition"] != "infanface-train"
                    or canonical_hash(sample.landmarks) != record["annotation_sha256"]
                ):
                    raise PilotError("test membership or changed annotation in pilot record")
                if not record["eligible"]:
                    continue
                face = images.load(sample)
                try:
                    box = ground_truth_crop(sample.landmarks, face.image.size)
                    if tuple(record["crop_box"]) != box:
                        raise PilotError("fixed ground-truth crop changed")
                    example = _tensor_example(face, box, config)
                    if int(example["valid"].sum()) != record["valid_points"]:
                        raise PilotError("prepared target coverage changed")
                    samples.append(sample)
                    examples.append(example)
                finally:
                    face.image.close()
        dataset = PreparedTrainingExamples(samples, examples)
        if len(dataset) != coverage["usable_images"]:
            raise PilotError("usable cohort changed after review")
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        generator = torch.Generator().manual_seed(config.seed + 1)
        training_loader = DataLoader(
            dataset, batch_size=config.batch_size, shuffle=True, generator=generator, num_workers=0
        )
        fit_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
        model = SmallLandmarkCNN().float().cpu()  # Fresh initialization; no checkpoint input.
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
        )
        state_digest = canonical_hash(
            {k: v.detach().tolist() for k, v in model.state_dict().items()}
        )
        provenance = _provenance(root)
        provenance.update(
            {
                **metadata,
                "runtime_hashes": metadata["runtime_hashes"],
                "verification_sha256": sha256(verification_path),
                "alignment_review_sha256": sha256(path / "alignment-review.json"),
                "python": sys.version,
                "torch": str(torch.__version__),
                "platform": platform.platform(),
                "fresh_random_initialization": True,
                "initial_model_state_sha256": state_digest,
                "initialization_seed": config.seed,
                "loader_seed": config.seed + 1,
                "device": "cpu",
                "dtype": "float32",
                "threads": 1,
                "num_workers": 0,
                "deterministic_algorithms": True,
                "pretrained_weights": None,
                "validation": None,
                "checkpoint_policy": "fixed-final-epoch",
                "source_origin": "UNVERIFIED",
                "global_annotation_frame": "UNVERIFIED",
            }
        )
        _new_json(path / "provenance.json", provenance)
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
        initial, _ = training_fit(model, fit_loader)
        _new_json(path / "initial-training-fit.json", initial)
        with (path / "epochs.jsonl").open("x") as log:
            for epoch in range(1, config.epochs + 1):
                row = pilot_epoch(model, training_loader, optimizer)
                steps += row["optimizer_steps"]
                row["epoch"] = epoch
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
        final, predictions = training_fit(model, fit_loader)
        payload = {
            "schema_version": 1,
            "experiment": config.protocol_id,
            "architecture": "small-landmark-cnn-v1",
            "checkpoint_policy": "fixed-final-epoch",
            "epoch": config.epochs,
            "config": asdict(config),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "loader_rng": generator.get_state(),
            "torch_version": str(torch.__version__),
            "selection_sha256": metadata["selection_sha256"],
            "protocol_sha256": metadata["protocol_sha256"],
            "validation": None,
            "source_coordinate_evidence": "UNVERIFIED",
        }
        with (path / "final.pt").open("xb") as stream:
            torch.save(payload, stream)
        checkpoint_sha = sha256(path / "final.pt")
        restored, _, _ = load_final_pilot_checkpoint(
            path / "final.pt", expected_sha256=checkpoint_sha
        )
        reloaded, reloaded_predictions = training_fit(restored, fit_loader)
        if not torch.equal(predictions, reloaded_predictions) or reloaded != final:
            raise PilotError("fixed-final checkpoint reload did not preserve training predictions")
        # Integrity verification reads the container bytes, not test image payloads.
        from infant_movement.datasets.infanface_images import _verify

        _verify(metadata["archive_path"], metadata["archive_sha256"])
        summary = {
            "experiment_kind": "ground-truth-crop training-only optimization smoke test",
            "coverage": coverage,
            "epochs": config.epochs,
            "optimizer_steps": steps,
            "initial_training_fit": initial,
            "fixed_final_training_fit": final,
            "checkpoint_policy": "fixed-final-epoch",
            "final_checkpoint_sha256": checkpoint_sha,
            "reload_predictions_equal": True,
            "reload_max_absolute_difference": 0.0,
            "reload_training_images": len(dataset),
            "validation": None,
            "anatomical_nme": None,
            "held_out_performance_claim": False,
            "patient_video_independence_claim": False,
            "source_coordinate_evidence": "UNVERIFIED",
            "test_images_opened": 0,
        }
        _new_json(path / "summary.json", summary)
        _status(
            path,
            "complete",
            execution_started=True,
            epochs_completed=config.epochs,
            optimizer_steps=steps,
        )
        return summary
    except Exception as error:
        _status(
            path,
            "failed",
            execution_started=True,
            optimizer_steps=steps,
            error_type=type(error).__name__,
        )
        raise
    finally:
        torch.set_rng_state(old_rng)
        random.setstate(old_python)
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_deterministic)
