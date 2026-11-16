"""Single frozen-checkpoint image-correspondence control; evaluation only."""

import platform
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import torch
from torch.utils.data import DataLoader

from infant_movement.datasets.infanface_pilot import PilotConfig, PilotError, canonical_hash
from infant_movement.engine.infanface_pilot import (
    _new_json,
    _run_path,
    _status,
    runtime_hashes,
    training_fit,
)
from infant_movement.engine.landmark_baseline import _provenance, sha256
from infant_movement.engine.mean_shape_control import (
    matching_environment_inventory,
    read_json,
    reconstruct_reference,
    snapshot_reference,
    verify_hashes,
)
from infant_movement.evaluation.image_correspondence import (
    FIELDS,
    ReassignedImages,
    cyclic_mapping,
    paired_comparison,
    tensor_digest,
)
from infant_movement.evaluation.mean_shape import (
    check_reproduction,
    evaluate_predictions,
    relative_reduction,
)
from infant_movement.models.landmark_baseline import SmallLandmarkCNN

PROTOCOL = "reports/infanface-correspondence-protocol-2026-11-16.md"
CONFIG_PATH = "configs/experiments/infanface_correspondence.json"
PILOT_PROTOCOL = "reports/infanface-training-pilot-protocol-2026-10-28.md"
PRESERVATION = "reports/private/infanface-correspondence-preservation-2026-11-16.json"
CONFIG = {
    "protocol_id": "infanface-image-correspondence-v1",
    "reference_run": "runs/infanface-train-pilot-2026-10-28",
    "mean_reference_run": "runs/infanface-mean-control-2026-10-28-verified",
    "sample_count": 32,
    "input_size": [32, 32],
    "device": "cpu",
    "batch_size": 8,
    "permutation": "destination_i_gets_prepared_image_at_(i+1)%32",
    "mse_atol": 1e-7,
    "pixel_atol": 1e-6,
    "rtol": 1e-6,
    "prediction_atol": 0,
    "prediction_rtol": 0,
    "paired_tie_atol": 1e-12,
}


def comparison_hashes(root):
    root = Path(root)
    return {
        **runtime_hashes(root, root / PILOT_PROTOCOL),
        **{str(root / p): sha256(root / p) for p in (PROTOCOL, CONFIG_PATH, PRESERVATION)},
    }


def snapshot_mean_reference(mean, pilot, frozen):
    """Validate saved mean evidence; do not fit or evaluate the constant again."""
    manifest = read_json(mean / "output-sha256.json")
    names = {p.name for p in mean.iterdir() if p.is_file()} - {"output-sha256.json"}
    if (
        manifest.get("algorithm") != "sha256"
        or set(manifest["files"]) != names
        or any(Path(p).name != p for p in manifest["files"])
    ):
        raise PilotError("mean-control output manifest membership mismatch")
    verify_hashes({str(mean / p): h for p, h in manifest["files"].items()})
    prior = read_json(mean / "frozen-reference.json")
    verify_hashes(prior["original_files"])
    verify_hashes(prior["historically_recorded_hashes"])
    provenance = read_json(mean / "provenance.json")
    verify_hashes(provenance["runtime_hashes"])
    verify_hashes(
        {str(pilot.parent.parent / p): h for p, h in provenance["source_config_hashes"].items()}
    )
    verify_hashes({str(mean / "frozen-reference.json"): provenance["frozen_reference_sha256"]})
    summary = read_json(mean / "summary.json")
    if (
        read_json(mean / "status.json").get("status") != "complete"
        or prior["original_files"] != frozen["original_files"]
        or prior["checkpoint_sha256"] != frozen["checkpoint_sha256"]
        or prior["selection_sha256"] != frozen["selection_sha256"]
        or read_json(mean / "paired-images.json")["ordered_references"]
        != read_json(pilot / "selection.json")["records"]
        or summary.get("training_updates") != 0
        or summary.get("training_images") != 32
        or summary.get("matched_eligible_points") != 2176
        or summary.get("mse_coordinate_scalars") != 4352
        or summary.get("cnn_reproduction_passed") is not True
        or any(read_json(mean / "cnn-reproduction.json")["discrepancies"].values())
    ):
        raise PilotError("completed mean-control reference disagrees with pilot")
    return {str(p): sha256(p) for p in mean.rglob("*") if p.is_file()}


def verify_annotation_container(root, metadata):
    """Read only the shared CSV bytes for hashing; no test label interpretation."""
    inventory_path = root / "reports/private/infanface-archive-inventory-2026-10-27.json"
    inventory = read_json(inventory_path)
    receipt = read_json(root / "data/raw/infanface/acquisition-receipt-2026-10-27.json")
    if (
        inventory["archive_sha256"] != metadata["archive_sha256"]
        or receipt["sha256"] != metadata["archive_sha256"]
    ):
        raise PilotError("archive receipt/inventory reference mismatch")
    entries = [r for r in inventory["files"] if r["archive_path"] == "infanface-public/labels.csv"]
    if len(entries) != 1:
        raise PilotError("original annotation inventory must have exactly one shared index")
    import hashlib

    with ZipFile(metadata["archive_path"]) as archive:
        digest = hashlib.sha256(archive.read(entries[0]["archive_path"])).hexdigest()
    if digest != entries[0]["sha256"]:
        raise PilotError("original annotation container hash mismatch")
    return digest


def load_frozen_model(path, frozen, metadata):
    """No optimizer is constructed or restored; all parameters are frozen."""
    if path.stat().st_size > 32 * 1024**2 or sha256(path) != frozen["checkpoint_sha256"]:
        raise PilotError("fixed-final checkpoint digest/size mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("schema_version") != 1
        or payload.get("experiment") != "infanface-train-fit-v1"
        or payload.get("architecture") != "small-landmark-cnn-v1"
        or payload.get("checkpoint_policy") != "fixed-final-epoch"
        or payload.get("epoch") != 20
        or PilotConfig(**payload["config"]) != PilotConfig()
        or payload.get("selection_sha256") != frozen["selection_sha256"]
        or payload.get("protocol_sha256") != metadata["protocol_sha256"]
    ):
        raise PilotError("fixed-final checkpoint contract mismatch")
    model = SmallLandmarkCNN().float().cpu()
    model.load_state_dict(payload["model"], strict=True)
    model.requires_grad_(False)
    model.eval()
    return model


def stacked_targets(dataset):
    return tuple(
        torch.stack([e[k] for e in dataset.examples])
        for k in ("target", "valid", "inverse", "source_landmarks")
    )


def prepared_digests(dataset):
    return [
        {k: tensor_digest(v) for k, v in e.items() if isinstance(v, torch.Tensor)}
        for e in dataset.examples
    ]


def verify_target_evidence(dataset, saved):
    values = stacked_targets(dataset)
    target, valid, inverse, source = values
    if (
        len(dataset) != 32
        or tuple(target.shape) != (32, 68, 2)
        or not valid.all()
        or canonical_hash([target.tolist(), valid.tolist(), inverse.tolist()])
        != saved["target_mask_transform_sha256"]
        or any(
            value.tolist() != saved[key]
            for value, key in zip(
                values, ("targets", "valid", "inverse", "source_targets"), strict=True
            )
        )
    ):
        raise PilotError("reconstructed destination targets/masks/transforms changed")
    return target, valid, inverse, source


def run_comparison(config_path, run_dir, *, project_root, verification_path):
    root, path = _run_path(run_dir, project_root)
    pilot, mean = (root / CONFIG[k] for k in ("reference_run", "mean_reference_run"))
    for historical in (root / "runs").iterdir():
        if path.is_relative_to(historical) or historical.is_relative_to(path):
            raise PilotError("refuse existing or historical run destinations and ancestors")
        prior_config = historical / "config.json"
        if (
            prior_config.is_file()
            and read_json(prior_config).get("protocol_id") == CONFIG["protocol_id"]
        ):
            raise PilotError("this single comparison phase has already attempted a run")
    if read_json(config_path) != CONFIG or Path(config_path).resolve() != root / CONFIG_PATH:
        raise PilotError("requires the exact predeclared correspondence config")
    if Path(sys.prefix).resolve() != (root / ".venv-baseline").resolve():
        raise PilotError("use the unchanged project .venv-baseline")
    path.mkdir(parents=True, exist_ok=False)
    old_rng, old_threads = torch.get_rng_state(), torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    stage = "preflight"
    _status(path, "running", stage=stage, training_updates=0)
    try:
        _new_json(path / "config.json", CONFIG)
        current = comparison_hashes(root)
        checks = read_json(verification_path)
        if (
            checks.get("checks_passed") is not True
            or checks.get("comparison_hashes") != current
            or not checks.get("commands")
            or any(c.get("returncode") != 0 for c in checks["commands"])
        ):
            raise PilotError("passing current-source software/dependency checks required")
        preservation = read_json(root / PRESERVATION)
        preserved = {
            p: h
            for p, h in preservation["file_hashes"].items()
            if Path(p) not in (root / "README.md", root / "AGENTS.md")
        }
        verify_hashes(preserved)
        frozen = snapshot_reference(pilot)
        frozen["mean_files"] = snapshot_mean_reference(mean, pilot, frozen)
        _new_json(path / "frozen-reference.json", frozen)
        metadata = read_json(pilot / "prepared.json")
        annotation_digest = verify_annotation_container(root, metadata)
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
        old = (pilot / "environment.txt").read_bytes()
        environment_matches = matching_environment_inventory(old, freeze)
        (path / "environment.txt").write_bytes(freeze)
        _new_json(
            path / "environment-comparison.json",
            {
                "inventory_matches": environment_matches,
                "raw_text_equal": old == freeze,
                "allowed_difference": "same editable infant_movement repository Git revision only",
                "original_environment_sha256": sha256(pilot / "environment.txt"),
                "current_environment_sha256": sha256(path / "environment.txt"),
            },
        )
        if not environment_matches:
            raise PilotError("installed inventory differs from original pilot")
        _new_json(
            path / "provenance.json",
            {
                **_provenance(root),
                "comparison_hashes": current,
                "verification_path": str(Path(verification_path).resolve()),
                "verification_sha256": sha256(verification_path),
                "preservation_sha256": sha256(root / PRESERVATION),
                "frozen_reference_sha256": sha256(path / "frozen-reference.json"),
                "archive_sha256": metadata["archive_sha256"],
                "annotation_container_sha256": annotation_digest,
                "checkpoint_sha256": frozen["checkpoint_sha256"],
                "python": sys.version,
                "platform": platform.platform(),
                "torch": str(torch.__version__),
                "device": "cpu",
                "dtype": "float32",
                "threads": 1,
                "deterministic_algorithms": True,
                "num_workers": 0,
                "batch_size": 8,
                "training_updates": 0,
                "source_origin": "UNVERIFIED",
                "annotation_frame": "UNVERIFIED",
            },
        )
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        stage = "reconstruction"
        dataset = reconstruct_reference(pilot)
        saved = read_json(mean / "private-predictions.json")
        target, valid, inverse, source = verify_target_evidence(dataset, saved)
        before = prepared_digests(dataset)
        mapping = cyclic_mapping(len(dataset))
        _new_json(
            path / "private-cohort-permutation.json",
            {
                "ordered_references": read_json(pilot / "selection.json")["records"],
                "source_position_for_destination": mapping,
                "bijection": sorted(mapping) == list(range(32)),
                "fixed_positions": sum(i == j for i, j in enumerate(mapping)),
                "prepared_tensor_digests": before,
            },
        )
        model = load_frozen_model(pilot / "final.pt", frozen, metadata)
        state_before = {k: tensor_digest(v) for k, v in model.state_dict().items()}
        loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
        stage = "paired_reproduction"
        reproduced, paired_prediction = training_fit(model, loader)
        paired, paired_rows = evaluate_predictions(
            paired_prediction, target, valid, inverse, source
        )
        original = read_json(pilot / "summary.json")["fixed_final_training_fit"]
        discrepancy = {
            "original_evaluator": check_reproduction(reproduced, original),
            "independent_against_pilot": check_reproduction(paired, original),
            "independent_against_mean_record": check_reproduction(
                paired, read_json(mean / "summary.json")["cnn"]
            ),
            "predictions": [] if paired_prediction.tolist() == saved["cnn"] else ["not_exact"],
        }
        _new_json(
            path / "cnn-reproduction.json",
            {
                "tolerances": {
                    k: CONFIG[k]
                    for k in (
                        "mse_atol",
                        "pixel_atol",
                        "rtol",
                        "prediction_atol",
                        "prediction_rtol",
                    )
                },
                "original": original,
                "reproduced": reproduced,
                "independent": paired,
                "discrepancies": discrepancy,
            },
        )
        if any(discrepancy.values()):
            raise PilotError("paired reproduction mismatch; retain evidence and stop")
        stage = "reassigned_evaluation"
        reassigned_loader = DataLoader(
            ReassignedImages(dataset, mapping), batch_size=8, shuffle=False, num_workers=0
        )
        _, reassigned_prediction = training_fit(model, reassigned_loader)
        reassigned, reassigned_rows = evaluate_predictions(
            reassigned_prediction, target, valid, inverse, source
        )
        descriptive, differences = paired_comparison(paired_rows, reassigned_rows)
        comparisons = {}
        for key in FIELDS:
            a, b = paired[key], reassigned[key]
            if isinstance(a, dict):
                a, b = a["mean_error"], b["mean_error"]
            relative = relative_reduction(a, b)
            comparisons[key] = {
                "reassigned_minus_paired": b - a if a is not None and b is not None else None,
                "relative_increase": -relative if relative is not None else None,
                "undefined_relative": relative is None,
            }
        _new_json(
            path / "private-predictions.json",
            {
                "paired": paired_prediction.tolist(),
                "reassigned": reassigned_prediction.tolist(),
                "targets": target.tolist(),
                "valid": valid.tolist(),
                "forward": [e["forward"].tolist() for e in dataset.examples],
                "inverse": inverse.tolist(),
                "source_targets": source.tolist(),
            },
        )
        _new_json(
            path / "paired-images.json",
            {
                "paired": paired_rows,
                "reassigned": reassigned_rows,
                "differences": differences,
            },
        )
        if prepared_digests(dataset) != before or state_before != {
            k: tensor_digest(v) for k, v in model.state_dict().items()
        }:
            raise PilotError("prepared input or frozen model state changed during evaluation")
        verify_hashes(preserved)
        verify_hashes(frozen["original_files"])
        verify_hashes(frozen["historically_recorded_hashes"])
        verify_hashes(frozen["mean_files"])
        verify_hashes(current)
        summary = {
            "experiment_kind": "one frozen-checkpoint training-cohort image-correspondence control",
            "training_images": 32,
            "matched_eligible_points": int(valid.sum()),
            "selected_point_slots": int(valid.numel()),
            "coordinate_scalars": 2 * int(valid.sum()),
            "new_exclusions": 0,
            "paired": paired,
            "reassigned": reassigned,
            "comparisons": comparisons,
            "descriptive_paired": descriptive,
            "historical_mean_context": read_json(mean / "summary.json")["control"],
            "cnn_reproduction_passed": True,
            "training_updates": 0,
            "test_images_opened": 0,
            "reassigned_conditions_evaluated": 1,
            "held_out_evaluation": False,
            "source_coordinate_evidence": "UNVERIFIED",
            "anatomical_nme": None,
        }
        _new_json(path / "summary.json", summary)
        _status(
            path,
            "complete",
            training_updates=0,
            reassigned_conditions_evaluated=1,
            historical_files_unchanged=True,
            prepared_tensors_unchanged=True,
            frozen_model_unchanged=True,
        )
        _new_json(
            path / "output-sha256.json",
            {
                "algorithm": "sha256",
                "scope": "all outputs except this manifest",
                "files": {p.name: sha256(p) for p in path.iterdir() if p.is_file()},
            },
        )
        return summary
    except Exception as error:
        _status(
            path,
            "failed",
            stage=stage,
            training_updates=0,
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
    finally:
        torch.set_rng_state(old_rng)
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_deterministic)
