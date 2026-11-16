"""Global reassignment and complete offline control with manufactured images only."""

# ruff: noqa: E402
import hashlib
import json
from zipfile import ZipFile

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL.Image")

from torch.utils.data import DataLoader, Dataset

from infant_movement.engine import image_correspondence as control
from infant_movement.engine import mean_shape_control as mean
from infant_movement.engine.landmark_baseline import sha256
from infant_movement.evaluation.image_correspondence import (
    ReassignedImages,
    cyclic_mapping,
    paired_comparison,
    tensor_digest,
    validate_mapping,
)
from infant_movement.evaluation.mean_shape import evaluate_predictions
from tests.test_mean_shape_control import frozen_pilot  # noqa: F401


class ManufacturedPositions(Dataset):
    def __len__(self):
        return 32

    def __getitem__(self, i):
        return {
            "image": torch.full((3, 32, 32), float(i)),
            "target": torch.tensor([[i / 32, 0.5], [float("nan"), float("nan")]]),
            "valid": torch.tensor([True, False]),
            "forward": torch.eye(3, dtype=torch.float64) * (i + 1),
            "inverse": torch.eye(3, dtype=torch.float64) / (i + 1),
            "crop_box": torch.tensor([i, i, i + 4, i + 4]),
            "official_partition": "infanface-train",
            "source_row": i,
        }


@pytest.mark.parametrize("batch_size", [1, 7, 8, 32])
def test_global_mapping_crosses_batch_boundaries_and_only_changes_images(batch_size):
    original = ManufacturedPositions()
    mapping = cyclic_mapping(32)
    assert sorted(mapping) == list(range(32))
    assert all(i != j for i, j in enumerate(mapping))
    reassigned = ReassignedImages(original, mapping)
    seen = []
    for batch in DataLoader(reassigned, batch_size=batch_size, shuffle=False):
        seen.extend(batch["image"][:, 0, 0, 0].tolist())
    assert seen == list(range(1, 32)) + [0]
    for i in range(32):
        a, b = original[i], reassigned[i]
        for key in a:
            if key == "image":
                assert torch.equal(b[key], original[(i + 1) % 32][key])
            elif isinstance(a[key], torch.Tensor):
                assert tensor_digest(a[key]) == tensor_digest(b[key])
            else:
                assert a[key] == b[key]


@pytest.mark.parametrize(
    "mapping",
    [
        tuple(range(32)),
        (1,) * 32,
        tuple((i + 2) % 32 for i in range(32)),
        tuple(range(1, 32)),
        (True,) * 32,
    ],
)
def test_changed_nonbijective_or_unauthorized_mapping_rejected(mapping):
    with pytest.raises(ValueError):
        validate_mapping(mapping, 32)


@pytest.mark.parametrize("count", [0, 1, 31, 33, True])
def test_changed_cohort_size_rejected(count):
    with pytest.raises(ValueError):
        cyclic_mapping(count)


def test_destination_inverse_and_masked_arithmetic_with_unequal_coverage():
    target = torch.tensor([[[0.0, 0.0], [float("nan"), float("nan")]], [[0.0, 0.0], [0.0, 0.0]]])
    valid = torch.tensor([[True, False], [True, True]])
    inverse = torch.tensor(
        [
            [[2.0, 0.0, 10.0], [0.0, 2.0, -5.0], [0.0, 0.0, 1.0]],
            [[3.0, 0.0, -7.0], [0.0, 3.0, 4.0], [0.0, 0.0, 1.0]],
        ],
        dtype=torch.float64,
    )
    source = torch.tensor(
        [[[10.0, -5.0], [float("nan"), float("nan")]], [[-7.0, 4.0], [-7.0, 4.0]]],
        dtype=torch.float64,
    )
    paired = target.clone()
    changed = torch.tensor(
        [[[3 / 32, 4 / 32], [float("nan"), float("nan")]], [[0.0, 0.0], [0.0, 10 / 32]]]
    )
    a, ar = evaluate_predictions(paired, target, valid, inverse, source)
    b, br = evaluate_predictions(changed, target, valid, inverse, source)
    assert a["normalized_coordinate_mse"] == 0
    assert b["normalized_coordinate_mse"] == pytest.approx(125 / (1024 * 6))
    assert b["input_pixel_error"]["mean_error"] == 5
    assert b["source_pixel_error"]["mean_error"] == 12.5
    summary, rows = paired_comparison(ar, br)
    assert rows[0]["source_pixel_error"]["reassigned_minus_paired"] == 10
    assert rows[1]["source_pixel_error"]["reassigned_minus_paired"] == 15
    assert summary["input_pixel_error"]["paired_lower_images"] == 2
    assert summary["input_pixel_error"]["undefined_relative_images"] == 2
    assert all(r["input_pixel_error"]["relative_increase"] is None for r in rows)


def test_all_missing_metrics_and_zero_denominator_are_undefined():
    target = torch.full((1, 2, 2), float("nan"))
    _, rows = evaluate_predictions(
        target,
        target,
        torch.zeros((1, 2), dtype=torch.bool),
        torch.eye(3).double().unsqueeze(0),
        target,
    )
    summary, private = paired_comparison(rows, rows)
    assert summary["input_pixel_error"]["paired_images"] == 0
    assert summary["input_pixel_error"]["undefined_images"] == 1
    assert private[0]["input_pixel_error"]["reassigned_minus_paired"] is None
    assert private[0]["input_pixel_error"]["relative_increase"] is None


def test_paired_sign_relative_increase_ties_and_mismatched_denominators():
    def row(value, scalars=2):
        return {
            "eligible_points": 1,
            "coordinate_scalars": scalars,
            **{k: value for k in control.FIELDS},
        }

    summary, private = paired_comparison([row(2), row(4), row(0)], [row(3), row(2), row(0)])
    assert private[0]["input_pixel_error"]["relative_increase"] == 0.5
    assert private[1]["input_pixel_error"]["relative_increase"] == -0.5
    assert summary["input_pixel_error"]["paired_lower_images"] == 1
    assert summary["input_pixel_error"]["reassigned_lower_images"] == 1
    assert summary["input_pixel_error"]["tied_images"] == 1
    with pytest.raises(ValueError, match="scalar"):
        paired_comparison([row(1)], [row(2, 4)])
    with pytest.raises(ValueError):
        paired_comparison([row(None)], [row(2)])


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


@pytest.fixture
def correspondence_fixture(frozen_pilot, monkeypatch):  # noqa: F811
    root, pilot, mean_config, checks = frozen_pilot
    mean_run = root / control.CONFIG["mean_reference_run"]
    mean.run_control(mean_config, mean_run, project_root=root, verification_path=checks)
    write_json(
        mean_run / "output-sha256.json",
        {
            "algorithm": "sha256",
            "files": {p.name: sha256(p) for p in mean_run.iterdir()},
        },
    )
    metadata = mean.read_json(pilot / "prepared.json")
    with ZipFile(metadata["archive_path"]) as archive:
        annotation_hash = hashlib.sha256(archive.read("infanface-public/labels.csv")).hexdigest()
    write_json(
        root / "reports/private/infanface-archive-inventory-2026-10-27.json",
        {
            "archive_sha256": metadata["archive_sha256"],
            "files": [{"archive_path": "infanface-public/labels.csv", "sha256": annotation_hash}],
        },
    )
    write_json(
        root / "data/raw/infanface/acquisition-receipt-2026-10-27.json",
        {"sha256": metadata["archive_sha256"]},
    )
    config = root / control.CONFIG_PATH
    write_json(config, control.CONFIG)
    (root / control.PROTOCOL).write_text("Manufactured correspondence protocol")
    preserved = {str(p): sha256(p) for p in root.rglob("*") if p.is_file() and p != checks}
    write_json(root / control.PRESERVATION, {"file_hashes": preserved})
    write_json(
        checks,
        {
            "checks_passed": True,
            "commands": [{"returncode": 0}],
            "comparison_hashes": control.comparison_hashes(root),
        },
    )
    # Fixture gate is synthetic; the real entry point independently requires the
    # existing baseline prefix and current-source full verification record.
    monkeypatch.setattr(control.sys, "prefix", str(root / ".venv-baseline"))
    yield root, pilot, mean_run, config, checks


def invoke(fixture, name="correspondence"):
    root, _, _, config, checks = fixture
    return control.run_comparison(
        config, root / "runs" / name, project_root=root, verification_path=checks
    )


def test_complete_control_forbids_training_save_and_additional_conditions(
    correspondence_fixture, monkeypatch
):
    root, pilot, mean_run, _, _ = correspondence_fixture
    original = {str(p): sha256(p) for d in (pilot, mean_run) for p in d.rglob("*") if p.is_file()}
    calls = []
    original_fit = control.training_fit

    def fit(model, loader):
        assert not model.training
        assert all(not p.requires_grad for p in model.parameters())

        def guard(module, args):
            assert not torch.is_grad_enabled()

        handle = model.register_forward_pre_hook(guard)
        calls.append(type(loader.dataset).__name__)
        try:
            return original_fit(model, loader)
        finally:
            handle.remove()

    def forbidden(*args, **kwargs):
        pytest.fail("training, saving or refitting a mean is forbidden")

    monkeypatch.setattr(control, "training_fit", fit)
    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.autograd, "backward", forbidden)
    monkeypatch.setattr(torch, "save", forbidden)
    monkeypatch.setattr(mean, "fit_mean_shape", forbidden)
    summary = invoke(correspondence_fixture)
    assert calls == ["PreparedTrainingExamples", "ReassignedImages"]
    assert summary["training_updates"] == 0
    assert summary["reassigned_conditions_evaluated"] == 1
    assert summary["matched_eligible_points"] == 2176
    run = root / "runs/correspondence"
    assert mean.read_json(run / "status.json")["status"] == "complete"
    integrity = mean.read_json(run / "output-sha256.json")
    mean.verify_hashes({str(run / k): v for k, v in integrity["files"].items()})
    mean.verify_hashes(original)
    assert not list(run.glob("*.pt"))
    with pytest.raises(ValueError, match="historical"):
        invoke(correspondence_fixture)
    with pytest.raises(ValueError, match="already attempted"):
        invoke(correspondence_fixture, "second")


@pytest.mark.parametrize(
    "violation",
    [
        "crop",
        "annotation",
        "membership",
        "order",
        "checkpoint",
        "config",
        "mean_output",
        "code_gate",
    ],
)
def test_changed_or_unauthorized_inputs_fail_before_pixels(
    correspondence_fixture, monkeypatch, violation
):
    root, pilot, mean_run, config, checks = correspondence_fixture
    if violation in ("crop", "annotation", "membership", "order"):
        selection = mean.read_json(pilot / "selection.json")
        if violation == "order":
            selection["records"].reverse()
        elif violation == "crop":
            selection["records"][0]["crop_box"][0] += 1
        elif violation == "annotation":
            selection["records"][0]["annotation_sha256"] = "0" * 64
        else:
            selection["records"][0]["official_partition"] = "infanface-test"
        write_json(pilot / "selection.json", selection)
    elif violation == "checkpoint":
        with (pilot / "final.pt").open("ab") as stream:
            stream.write(b"changed")
    elif violation == "config":
        write_json(config, {**control.CONFIG, "batch_size": 4})
    elif violation == "mean_output":
        write_json(mean_run / "summary.json", {})
    else:
        write_json(checks, {"checks_passed": False})

    def forbidden(*args):
        pytest.fail("changed inputs must fail before real reconstruction/inference")

    monkeypatch.setattr(control, "reconstruct_reference", forbidden)
    with pytest.raises(ValueError):
        invoke(correspondence_fixture, "rejected")
    status = root / "runs/rejected/status.json"
    if status.exists():
        assert mean.read_json(status)["status"] == "failed"


def test_discrepancy_stops_before_reassignment_and_preserves_failure(
    correspondence_fixture, monkeypatch
):
    root, _, _, _, _ = correspondence_fixture
    original_fit = control.training_fit
    calls = []

    def discrepancy(model, loader):
        calls.append(type(loader.dataset).__name__)
        metrics, predictions = original_fit(model, loader)
        metrics["normalized_coordinate_mse"] += 1
        return metrics, predictions

    monkeypatch.setattr(control, "training_fit", discrepancy)
    with pytest.raises(ValueError, match="reproduction mismatch"):
        invoke(correspondence_fixture, "failed")
    assert calls == ["PreparedTrainingExamples"]
    run = root / "runs/failed"
    assert mean.read_json(run / "cnn-reproduction.json")["discrepancies"]["original_evaluator"]
    assert mean.read_json(run / "status.json")["stage"] == "paired_reproduction"
    assert not (run / "paired-images.json").exists()


def test_destination_paths_and_target_evidence_rejected(correspondence_fixture):
    root, pilot, mean_run, config, checks = correspondence_fixture
    for path in (root / "public", root / "runs", pilot, pilot / "nested", mean_run / "nested"):
        with pytest.raises(ValueError):
            control.run_comparison(config, path, project_root=root, verification_path=checks)
    escape = root / "runs/escape"
    escape.symlink_to(root / "reports", target_is_directory=True)
    with pytest.raises(ValueError):
        control.run_comparison(config, escape / "run", project_root=root, verification_path=checks)
    dataset = control.reconstruct_reference(pilot)
    saved = mean.read_json(mean_run / "private-predictions.json")
    dataset.examples[0]["inverse"][0, 2] += 1
    with pytest.raises(ValueError, match="destination"):
        control.verify_target_evidence(dataset, saved)
