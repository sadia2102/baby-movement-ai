"""Frozen-reference, evaluation-only constant-shape control; no training API."""

import json
import re
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from infant_movement.datasets.infanface_pilot import (
    PilotConfig,
    PilotError,
    canonical_hash,
    ground_truth_crop,
    open_training_images,
    require_training,
)
from infant_movement.engine.infanface_pilot import (
    PreparedTrainingExamples,
    _new_json,
    _run_path,
    _status,
    _tensor_example,
    load_final_pilot_checkpoint,
    runtime_hashes,
    training_fit,
)
from infant_movement.engine.landmark_baseline import _provenance, sha256
from infant_movement.evaluation.mean_shape import (
    check_reproduction,
    evaluate_predictions,
    paired_differences,
    relative_reduction,
)
from infant_movement.models.mean_shape import fit_mean_shape

CONTROL_CONFIG = {
    "protocol_id": "infanface-mean-control-v1",
    "reference_run": "runs/infanface-train-pilot-2026-10-28",
    "sample_count": 32,
    "input_size": [32, 32],
    "device": "cpu",
    "mean_dtype": "float64",
    "mse_atol": 1e-7,
    "pixel_atol": 1e-6,
    "rtol": 1e-6,
    "paired_tie_atol": 1e-12,
}
PROTOCOL = "reports/infanface-mean-control-protocol-2026-10-28.md"
PILOT_PROTOCOL = "reports/infanface-training-pilot-protocol-2026-10-28.md"


def read_json(path):
    return json.loads(Path(path).read_text())


def matching_environment_inventory(original, current):
    """Ignore only editable project's Git commit field; verify code separately.

    pip freeze changes that field after a commit. All dependency pins, editable
    repository/egg and other text must still match; old source hashes are checked.
    """
    pattern = re.compile(r"^(-e git\+[^\s]+)@[0-9a-f]{40}(#egg=infant_movement)$", re.MULTILINE)

    def normalize(value):
        return pattern.sub(r"\1@RECORDED-GIT-REVISION\2", value.decode())

    return normalize(original) == normalize(current)


def verify_hashes(hashes):
    if not hashes or any(not Path(p).is_file() or sha256(p) != h for p, h in hashes.items()):
        raise PilotError("frozen reference/software digest mismatch")


def snapshot_reference(reference):
    """Check historical cross-record digests, then freeze every original run file.

    Summary/status/etc without a previously recorded digest are only frozen now;
    their consistency is checked, but no historical authenticity is invented.
    """
    reference = Path(reference).resolve()
    status = read_json(reference / "status.json")
    config = PilotConfig(**read_json(reference / "config.json"))
    metadata = read_json(reference / "prepared.json")
    manifest = read_json(reference / "selection.json")
    summary = read_json(reference / "summary.json")
    coverage = read_json(reference / "coverage.json")
    provenance = read_json(reference / "provenance.json")
    if (
        status.get("status") != "complete"
        or status.get("epochs_completed") != config.epochs
        or summary.get("epochs") != config.epochs
        or summary.get("checkpoint_policy") != "fixed-final-epoch"
        or summary.get("coverage") != coverage
        or summary.get("validation") is not None
        or not summary.get("reload_predictions_equal")
        or config != PilotConfig()
    ):
        raise PilotError("requires the exact completed original 32-image/20-epoch pilot")
    recorded = {
        str(reference / name): metadata[key]
        for name, key in (
            ("config.json", "config_sha256"),
            ("selection.json", "selection_file_sha256"),
            ("coverage.json", "coverage_sha256"),
        )
    }
    recorded.update(metadata["runtime_hashes"])
    root = reference.parent.parent
    recorded.update({str(root / p): h for p, h in provenance["source_config_hashes"].items()})
    recorded[
        str(root / "reports/private/infanface-pilot-software-verification-2026-10-28.json")
    ] = provenance["verification_sha256"]
    recorded[metadata["protocol_path"]] = metadata["protocol_sha256"]
    recorded[str(reference / "alignment-review.json")] = provenance["alignment_review_sha256"]
    recorded[str(reference / "final.pt")] = summary["final_checkpoint_sha256"]
    for name in ("selection_sha256", "archive_sha256", "config_sha256", "protocol_sha256"):
        if provenance[name] != metadata[name]:
            raise PilotError("original provenance cross-reference mismatch")
    review = read_json(reference / "alignment-review.json")
    if (
        not review.get("approved")
        or not review.get("gross_alignment_supported")
        or review["selection_sha256"] != metadata["selection_sha256"]
        or review["selection_file_sha256"] != metadata["selection_file_sha256"]
        or review["reviewed_usable_images"] != coverage["usable_images"]
    ):
        raise PilotError("original alignment review mismatch")
    verify_hashes(recorded)
    records = manifest["records"]
    refs = [
        {k: r[k] for k in ("image_set", "filename", "source_row", "official_partition")}
        for r in records
    ]
    if (
        manifest["selection_sha256"] != metadata["selection_sha256"]
        or canonical_hash(refs) != metadata["selection_sha256"]
        or len(records) != 32
        or len({(r["image_set"], r["filename"]) for r in records}) != 32
        or any(
            r["official_partition"] != "infanface-train" or r["eligible"] is not True
            for r in records
        )
        or coverage["usable_images"] != 32
        or coverage["eligible_point_slots"] != 2176
        or any(r["valid_points"] != 68 for r in records)
    ):
        raise PilotError("original usable cohort/membership/coverage mismatch")
    files = {str(p): sha256(p) for p in reference.rglob("*") if p.is_file()}
    return {
        "original_files": files,
        "historically_recorded_hashes": recorded,
        "selection_sha256": metadata["selection_sha256"],
        "checkpoint_sha256": summary["final_checkpoint_sha256"],
        "records_without_prior_digest": "original_files entries not historically_recorded_hashes",
    }


def reconstruct_reference(reference):
    """Join only frozen rows; no ranking, extra image access, replacements or new crop."""
    reference = Path(reference)
    metadata = read_json(reference / "prepared.json")
    config = PilotConfig(**read_json(reference / "config.json"))
    records = read_json(reference / "selection.json")["records"]
    samples, examples = [], []
    with open_training_images(
        metadata["archive_path"], expected_sha256=metadata["archive_sha256"]
    ) as images:
        lookup = {(s.image_set, s.filename): s for s in images.samples}
        for record in records:
            if record["official_partition"] != "infanface-train" or not record["eligible"]:
                raise PilotError("unauthorized or changed usable reference")
            sample = lookup.get((record["image_set"], record["filename"]))
            if sample is None:
                raise PilotError("frozen reference is absent from official training pool")
            require_training((sample,))
            if (
                sample.source_row != record["source_row"]
                or canonical_hash(sample.landmarks) != record["annotation_sha256"]
            ):
                raise PilotError("frozen source row/annotation changed")
            face = images.load(sample)
            try:
                box = tuple(record["crop_box"])
                if (
                    face.image.size != tuple(record["source_size_wh"])
                    or face.orientation != record["exif_orientation"]
                    or ground_truth_crop(sample.landmarks, face.image.size) != box
                ):
                    raise PilotError("source raster/orientation/crop changed")
                example = _tensor_example(
                    face, box, config
                )  # Apply the recorded box, never replace it.
                counts = {
                    key: int((~example[key]).sum())
                    for key in ("present", "source_in_bounds", "crop_in_bounds", "output_in_bounds")
                }
                if (
                    counts != record["mask_false_counts"]
                    or int(example["valid"].sum()) != record["valid_points"]
                ):
                    raise PilotError("original geometric masks/coverage changed")
                # The original run stored counts, not tensors. Its all-valid 68/68
                # contract uniquely fixes every mask bit for this exact cohort.
                if not example["valid"].all():
                    raise PilotError("exact original all-valid mask was not reproduced")
                samples.append(sample)
                examples.append(example)
            finally:
                face.image.close()
    return PreparedTrainingExamples(samples, examples)


def nullable_coordinates(tensor):
    """Keep unsupported coordinate pairs undefined in strict JSON."""
    return [row.tolist() if torch.isfinite(row).all() else None for row in tensor]


def run_control(config_path, run_dir, *, project_root, verification_path):
    root, path = _run_path(run_dir, project_root)
    config_path = Path(config_path).resolve()
    config = read_json(config_path)
    if config != CONTROL_CONFIG:
        raise PilotError("comparison config must match the predeclared fixed control")
    reference = (root / config["reference_run"]).resolve()
    if path.is_relative_to(reference) or reference.is_relative_to(path):
        raise PilotError("comparison must not write into or above the original run")
    protocol = root / PROTOCOL
    if not protocol.is_file():
        raise PilotError("write the comparison protocol before calculation")
    current = runtime_hashes(root, root / PILOT_PROTOCOL)
    checks = read_json(verification_path)
    if (
        checks.get("checks_passed") is not True
        or checks.get("runtime_hashes") != current
        or not checks.get("commands")
        or any(c.get("returncode") != 0 for c in checks["commands"])
    ):
        raise PilotError("passing software checks for current code are required")
    frozen = snapshot_reference(reference)
    # An unused destination is required even if the previous comparison failed.
    path.mkdir(parents=True, exist_ok=False)
    _status(path, "running", training_updates=0)
    old_rng, old_threads = torch.get_rng_state(), torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        _new_json(path / "config.json", config)
        _new_json(path / "frozen-reference.json", frozen)
        runtime = {
            **current,
            str(protocol): sha256(protocol),
            str(config_path): sha256(config_path),
        }
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
        old_environment = (reference / "environment.txt").read_bytes()
        inventory_matches = matching_environment_inventory(old_environment, freeze)
        _new_json(
            path / "environment-comparison.json",
            {
                "raw_text_equal": freeze == old_environment,
                "inventory_matches": inventory_matches,
                "allowed_difference": "editable infant_movement Git revision field only",
                "original_environment_sha256": sha256(reference / "environment.txt"),
            },
        )
        if not inventory_matches:
            raise PilotError("installed inventory differs from the original pilot")
        (path / "environment.txt").write_bytes(freeze)
        provenance = {
            **_provenance(root),
            "runtime_hashes": runtime,
            "verification_sha256": sha256(verification_path),
            "reference_run": str(reference),
            "frozen_reference_sha256": sha256(path / "frozen-reference.json"),
            "python": sys.version,
            "torch": str(torch.__version__),
            "device": "cpu",
            "threads": 1,
            "cnn_dtype": "float32",
            "mean_dtype": "float64",
            "training_updates": 0,
            "source_origin": "UNVERIFIED",
            "annotation_frame": "UNVERIFIED",
            "experiment_kind": "same-cohort ground-truth-crop training-only control",
        }
        _new_json(path / "provenance.json", provenance)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        dataset = reconstruct_reference(reference)
        loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
        model, _, payload = load_final_pilot_checkpoint(
            reference / "final.pt", expected_sha256=frozen["checkpoint_sha256"]
        )
        original_metadata = read_json(reference / "prepared.json")
        if (
            payload["epoch"] != 20
            or PilotConfig(**payload["config"])
            != PilotConfig(**read_json(reference / "config.json"))
            or payload["selection_sha256"] != frozen["selection_sha256"]
            or payload["protocol_sha256"] != original_metadata["protocol_sha256"]
        ):
            raise PilotError("fixed-final checkpoint reference mismatch")
        original = read_json(reference / "summary.json")["fixed_final_training_fit"]
        reproduced, prediction = training_fit(model, loader)  # eval/no_grad; never training.
        target = torch.stack([e["target"] for e in dataset.examples])
        valid = torch.stack([e["valid"] for e in dataset.examples])
        inverse = torch.stack([e["inverse"] for e in dataset.examples])
        source = torch.stack([e["source_landmarks"] for e in dataset.examples])
        cnn, cnn_rows = evaluate_predictions(prediction, target, valid, inverse, source)
        discrepancies = {
            name: check_reproduction(value, original)
            for name, value in (("original_evaluator", reproduced), ("independent_float64", cnn))
        }
        _new_json(
            path / "cnn-reproduction.json",
            {
                "tolerances": {
                    "mse_atol": config["mse_atol"],
                    "pixel_atol": config["pixel_atol"],
                    "rtol": config["rtol"],
                },
                "original": original,
                "reproduced": reproduced,
                "independent": cnn,
                "discrepancies": discrepancies,
            },
        )
        if any(discrepancies.values()):
            raise PilotError(
                "CNN reproduction mismatch; stop and investigate private discrepancy record"
            )
        control = fit_mean_shape(target, valid)  # Only targets/masks enter the constant estimator.
        matched = valid & (control.support > 0).unsqueeze(0)
        if not torch.equal(matched, valid):
            raise PilotError("matched observations must not shrink the original cohort coverage")
        constant = control.predict(len(dataset))
        mean, mean_rows = evaluate_predictions(constant, target, matched, inverse, source)
        paired, private = paired_differences(
            cnn_rows, mean_rows, tie_atol=config["paired_tie_atol"]
        )
        _new_json(
            path / "mean-shape.json",
            {
                "coordinates": nullable_coordinates(control.coordinates),
                "support_per_landmark": control.support.tolist(),
                "units": "normalized input pixel-edge pairs",
            },
        )
        _new_json(
            path / "private-predictions.json",
            {
                "cnn": prediction.tolist(),
                "control": constant.tolist(),
                "targets": target.tolist(),
                "valid": valid.tolist(),
                "inverse": inverse.tolist(),
                "source_targets": source.tolist(),
                "target_mask_transform_sha256": canonical_hash(
                    [target.tolist(), valid.tolist(), inverse.tolist()]
                ),
            },
        )
        _new_json(
            path / "paired-images.json",
            {
                "cnn": cnn_rows,
                "control": mean_rows,
                "paired": private,
                "ordered_references": read_json(reference / "selection.json")["records"],
            },
        )
        comparisons = {}
        for key in ("normalized_coordinate_mse", "input_pixel_error", "source_pixel_error"):
            c, m = cnn[key], mean[key]
            if isinstance(c, dict):
                c, m = c["mean_error"], m["mean_error"]
            comparisons[key] = {
                "cnn_minus_control": c - m,
                "relative_error_reduction": relative_reduction(m, c),
                "cnn_lower": c < m,
                "control_denominator_zero": m == 0,
            }
        summary = {
            "experiment_kind": "ground-truth-crop same-training-cohort mean-shape control",
            "training_images": len(dataset),
            "selected_point_slots": int(valid.numel()),
            "original_eligible_points": int(valid.sum()),
            "matched_eligible_points": int(matched.sum()),
            "mse_coordinate_scalars": 2 * int(matched.sum()),
            "landmark_slots": len(control.support),
            "zero_support_landmarks": int((control.support == 0).sum()),
            "minimum_landmark_support": int(control.support.min()),
            "maximum_landmark_support": int(control.support.max()),
            "cnn": cnn,
            "control": mean,
            "comparisons": comparisons,
            "paired": paired,
            "cnn_reproduction_passed": True,
            "training_updates": 0,
            "test_images_opened": 0,
            "held_out_evaluation": False,
            "source_coordinate_evidence": "UNVERIFIED",
            "anatomical_nme": None,
        }
        verify_hashes(frozen["original_files"])
        verify_hashes(frozen["historically_recorded_hashes"])
        verify_hashes(runtime)
        _new_json(path / "summary.json", summary)
        _status(path, "complete", training_updates=0, original_files_unchanged=True)
        return summary
    except Exception as error:
        _status(path, "failed", training_updates=0, error_type=type(error).__name__)
        raise
    finally:
        torch.set_rng_state(old_rng)
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_deterministic)
