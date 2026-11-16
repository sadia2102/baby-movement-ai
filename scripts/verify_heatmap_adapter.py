"""Offline manufactured-tensor verification; no dataset/archive/checkpoint arguments."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch

from infant_movement.datasets.coordinate_conventions import source_centres_to_edges
from infant_movement.datasets.landmark_geometry import crop_geometry, letterbox_geometry
from infant_movement.datasets.landmark_heatmaps import (
    HeatmapConfig,
    decode_argmax,
    generate_heatmaps,
    masked_heatmap_mse,
)
from infant_movement.evaluation.provenance_readiness import (
    digest,
    locked_pins,
    new_review_directory,
)

CONFIG_PATH = "configs/experiments/synthetic_heatmap_adapter.json"
PROTOCOL = "reports/heatmap-adapter-protocol-2026-11-16.md"
CONFIG = {
    "protocol_id": "synthetic-heatmap-adapter-v1",
    "input_size": [32, 32],
    "heatmap_size": [8, 8],
    "sigma": 1.0,
    "landmark_slots": 68,
    "device": "cpu",
    "target_dtype": "float32",
    "arithmetic_reference_dtype": "float64",
    "source_index_base": 0,
    "annotation_frame": "stored",
    "float64_atol": 1e-12,
    "geometry_atol": 1e-10,
    "float32_atol": 1e-7,
    "float32_rtol": 1e-6,
}


def synthetic_checks():
    config = HeatmapConfig()
    points = torch.tensor([[[10.0, 14.0]] * 68])
    mask = torch.ones((1, 68), dtype=torch.bool)
    points[0, 66] = float("nan")
    points[0, 67] = torch.tensor([32.0, 4.0])
    mask[0, 66] = False
    targets = generate_heatmaps(points, mask, config=config)
    affine = torch.tensor(
        [[[2.0, 0.0, 10.0], [0.0, 2.0, -5.0], [0.0, 0.0, 1.0]]], dtype=torch.float64
    )
    decoded = decode_argmax(
        targets.heatmaps,
        targets.eligible,
        inverse=affine,
        source_index_base=0,
        annotation_frame="stored",
        config=config,
    )
    fractional = generate_heatmaps(
        torch.tensor([[[11.2, 14.8]]], dtype=torch.float64), torch.tensor([[True]]), config=config
    )
    quantized = decode_argmax(
        fractional.heatmaps,
        fractional.eligible,
        inverse=torch.eye(3).double().unsqueeze(0),
        source_index_base=0,
        annotation_frame="stored",
        config=config,
    )
    _, crop, size = crop_geometry((4, 2, 16, 8), (20, 10))
    fit, _, _ = letterbox_geometry(size, (30, 18))
    forward = source_centres_to_edges(source_index_base=0).then(crop).then(fit)
    non_square = HeatmapConfig(input_size=(30, 18), heatmap_size=(7, 5))
    non_square_target = generate_heatmaps(
        torch.tensor([[[15.0, 9.0]]], dtype=torch.float64),
        torch.tensor([[True]]),
        config=non_square,
    )
    recovered = decode_argmax(
        non_square_target.heatmaps,
        non_square_target.eligible,
        inverse=torch.tensor([forward.inverse().matrix], dtype=torch.float64),
        source_index_base=0,
        annotation_frame="stored",
        config=non_square,
    )
    prediction = torch.tensor(
        [[[[1.0, 3.0]], [[float("nan"), float("nan")]]], [[[2.0, 4.0]], [[0.0, 2.0]]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    truth = torch.zeros_like(prediction)
    truth[0, 1] = float("nan")
    loss_mask = torch.tensor([[True, False], [True, True]])
    loss = masked_heatmap_mse(prediction, truth, loss_mask)
    loss.loss.backward()  # Leaf-tensor arithmetic only; no model, optimizer or training loop.
    expected_gradient = torch.tensor(
        [[[[1 / 3, 1.0]], [[0.0, 0.0]]], [[[2 / 3, 4 / 3]], [[0.0, 2 / 3]]]], dtype=torch.float64
    )
    gradient_error = float((prediction.grad - expected_gradient).abs().max())
    no_loss = masked_heatmap_mse(prediction, truth, torch.zeros_like(loss_mask))
    flat = decode_argmax(
        torch.zeros((1, 1, 8, 8)),
        torch.tensor([[True]]),
        inverse=torch.eye(3).double().unsqueeze(0),
        source_index_base=0,
        annotation_frame="stored",
    )
    quantization_error = float(
        torch.linalg.vector_norm(quantized.input_edges[0, 0] - fractional.input_edges[0, 0])
    )
    source_error = max(
        abs(a - b)
        for a, b in zip(recovered.source_coordinates[0, 0].tolist(), [9.5, 4.5], strict=True)
    )
    checks = {
        "initial_68_slot_layout": tuple(targets.heatmaps.shape) == (1, 68, 8, 8),
        "known_unit_peak": float(targets.heatmaps[0, 0, 3, 2]) == 1,
        "gaussian_neighbor": math.isclose(
            float(targets.heatmaps[0, 0, 3, 3]), math.exp(-0.5), abs_tol=1e-7, rel_tol=1e-6
        ),
        "missing_oob_mask_counts": int(targets.eligible.sum()) == 66
        and targets.coverage["requested_eligible_slots"] == 67,
        "source_no_extra_half_pixel": decoded.source_coordinates[0, 0].tolist() == [30.0, 23.0],
        "fractional_unnormalized_peak": math.isclose(
            float(quantized.peak_scores[0, 0]), math.exp(-0.065), abs_tol=1e-12, rel_tol=0
        ),
        "argmax_quantization_expected": math.isclose(
            quantization_error, math.sqrt(2.08), abs_tol=1e-12, rel_tol=0
        ),
        "nonsquare_crop_padding_inverse": source_error <= 1e-10,
        "masked_loss_34_over_6": math.isclose(
            float(loss.loss.detach()), 34 / 6, abs_tol=1e-12, rel_tol=0
        )
        and loss.denominator == 6,
        "masked_gradients": gradient_error <= 1e-12,
        "all_missing_no_loss": no_loss.loss is None and no_loss.denominator == 0,
        "flat_is_undefined": not flat.supported.any().item()
        and torch.isnan(flat.source_coordinates).all().item(),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "initial_target_coverage": targets.coverage,
        "gaussian_neighbor_value": float(targets.heatmaps[0, 0, 3, 3]),
        "fractional_peak_score": float(quantized.peak_scores[0, 0]),
        "fractional_decoded_input_edges": quantized.input_edges[0, 0].tolist(),
        "fractional_input_quantization_error": quantization_error,
        "nonsquare_case_config": asdict(non_square),
        "nonsquare_source_coordinates": recovered.source_coordinates[0, 0].tolist(),
        "nonsquare_source_max_absolute_error": source_error,
        "masked_loss": float(loss.loss.detach()),
        "loss_cell_denominator": loss.denominator,
        "maximum_gradient_error": gradient_error,
        "dataset_samples_used": 0,
        "images_opened": 0,
        "model_operations": 0,
        "optimizer_steps": 0,
        "checkpoint_operations": 0,
        "network_operations": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", help="optional new ignored run directory; omitted prints checks"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = None
    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()

    def status(value, **fields):
        if path is not None:
            (path / "status.json").write_text(
                json.dumps(
                    {"status": value, "recorded_at_utc": datetime.now(UTC).isoformat(), **fields},
                    indent=2,
                )
            )

    def save(name, value):
        with (path / name).open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)

    try:
        if Path(sys.prefix).resolve() != (root / ".venv-baseline").resolve():
            raise ValueError("use the unchanged project .venv-baseline")
        if args.run_dir:
            path = new_review_directory(args.run_dir, root)
            status("running", synthetic_only=True)
        config_path = root / CONFIG_PATH
        if json.loads(config_path.read_text()) != CONFIG:
            raise ValueError("fixed synthetic heatmap configuration changed")
        lock = root / "environments/macos-arm64-py313-baseline.lock"
        pins = locked_pins(lock, root / "environments")
        installed = {
            re.sub("[-_.]+", "-", d.metadata["Name"]).lower(): d.version
            for d in importlib.metadata.distributions()
        }
        expected = {re.sub("[-_.]+", "-", k).lower(): v for k, v in pins.items()}
        expected["infant-movement"] = "0.1.0"
        if len(pins) != 19 or installed != expected:
            raise ValueError("installed inventory differs from retained exact lock plus project")
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        result = synthetic_checks()
        if path is not None:
            save("config.json", CONFIG)
            save("arithmetic.json", result)
        if not result["passed"]:
            raise ValueError("manufactured heatmap arithmetic failed")
        if path is not None:
            paths = list((root / "src").rglob("*.py")) + list((root / "tests").glob("*.py"))
            paths += list((root / "scripts").glob("*.py")) + [
                config_path,
                lock,
                root / PROTOCOL,
                root / "pyproject.toml",
                root / "configs/datasets/infanface_conventions.json",
            ]
            freeze = subprocess.check_output(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "--disable-pip-version-check",
                    "--no-cache-dir",
                    "freeze",
                    "--all",
                ]
            )
            (path / "environment.txt").write_bytes(freeze)
            save(
                "provenance.json",
                {
                    "python": sys.version,
                    "torch": str(torch.__version__),
                    "platform": platform.platform(),
                    "installed_versions": installed,
                    "locked_distributions": len(pins),
                    "device": "cpu",
                    "threads": 1,
                    "deterministic_algorithms": True,
                    "code_config_protocol_sha256": {
                        str(p.relative_to(root)): digest(p) for p in paths
                    },
                    "environment_sha256": digest(path / "environment.txt"),
                    "git_head": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=root, text=True
                    ).strip(),
                    "git_branch": subprocess.check_output(
                        ["git", "branch", "--show-current"], cwd=root, text=True
                    ).strip(),
                    "git_status": subprocess.check_output(
                        ["git", "status", "--short"], cwd=root, text=True
                    ),
                    "tracked_diff_sha256": hashlib.sha256(
                        subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=root)
                    ).hexdigest(),
                    "real_data_or_historical_results_accessed": False,
                },
            )
            status(
                "complete",
                synthetic_only=True,
                checks_passed=len(result["checks"]),
                model_operations=0,
                optimizer_steps=0,
                checkpoint_operations=0,
            )
            save(
                "output-sha256.json",
                {"algorithm": "sha256", "files": {p.name: digest(p) for p in path.iterdir()}},
            )
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        status("failed", error_type=type(error).__name__)
        print(json.dumps({"passed": False, "error_type": type(error).__name__}))
        return 2
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_deterministic)


if __name__ == "__main__":
    raise SystemExit(main())
