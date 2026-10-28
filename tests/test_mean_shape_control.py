"""Frozen-reference guards and complete comparison using manufactured images."""

# ruff: noqa: E402
import csv
import io
import json
import subprocess
import sys
from dataclasses import asdict
from zipfile import ZipFile

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL.Image")

from torch.utils.data import DataLoader

from infant_movement.datasets.infanface import COLUMNS
from infant_movement.datasets.infanface_pilot import PilotConfig, PilotError
from infant_movement.engine import mean_shape_control as control
from infant_movement.engine.infanface_pilot import prepare_pilot, runtime_hashes, training_fit
from infant_movement.engine.landmark_baseline import _provenance, sha256
from infant_movement.models.landmark_baseline import SmallLandmarkCNN


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def frozen_pilot(tmp_path, pilot_archive):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    base, _ = pilot_archive(tmp_path)
    with ZipFile(base) as archive:
        pixel = archive.read("infanface-public/images/ads/manufactured-a.png")
        rows = list(
            csv.DictReader(io.StringIO(archive.read("infanface-public/labels.csv").decode()))
        )
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerow(rows[0])  # Invalid test numeric payload must remain uninterpreted.
    for index in range(32):
        row = {**rows[1], "filename": f"manufactured-{index}.png", "gt-x67": "4", "gt-y67": "3"}
        writer.writerow(row)
    archive = tmp_path / "synthetic32.zip"
    with ZipFile(archive, "w") as out:
        out.writestr("infanface-public/labels.csv", text.getvalue())
        for index in range(32):
            out.writestr(f"infanface-public/images/ads/manufactured-{index}.png", pixel)
    protocol = tmp_path / control.PILOT_PROTOCOL
    protocol.parent.mkdir(parents=True)
    protocol.write_text("Manufactured original protocol")
    config_path = tmp_path / "configs/experiments/infanface_mean_control.json"
    write_json(config_path, control.CONTROL_CONFIG)
    (tmp_path / control.PROTOCOL).write_text("Manufactured comparison protocol; no infant data")
    reference = tmp_path / control.CONTROL_CONFIG["reference_run"]
    coverage = prepare_pilot(
        PilotConfig(),
        archive,
        sha256(archive),
        reference,
        project_root=tmp_path,
        protocol_path=protocol,
    )
    review = json.loads((reference / "alignment-review-template.json").read_text())
    review.update(
        approved=True,
        gross_alignment_supported=True,
        reviewed_usable_images=32,
        reviewer="synthetic fixture",
    )
    write_json(reference / "alignment-review.json", review)
    original_checks = (
        tmp_path / "reports/private/infanface-pilot-software-verification-2026-10-28.json"
    )
    write_json(original_checks, {"synthetic_fixture": True})
    prepared = json.loads((reference / "prepared.json").read_text())
    dataset = control.reconstruct_reference(reference)
    model = SmallLandmarkCNN().float().cpu()
    metrics, _ = training_fit(model, DataLoader(dataset, batch_size=8, shuffle=False))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    torch.save(
        {
            "schema_version": 1,
            "experiment": "infanface-train-fit-v1",
            "architecture": "small-landmark-cnn-v1",
            "checkpoint_policy": "fixed-final-epoch",
            "config": asdict(PilotConfig()),
            "epoch": 20,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "selection_sha256": prepared["selection_sha256"],
            "protocol_sha256": prepared["protocol_sha256"],
        },
        reference / "final.pt",
    )
    write_json(
        reference / "summary.json",
        {
            "epochs": 20,
            "coverage": coverage,
            "validation": None,
            "checkpoint_policy": "fixed-final-epoch",
            "reload_predictions_equal": True,
            "final_checkpoint_sha256": sha256(reference / "final.pt"),
            "fixed_final_training_fit": metrics,
        },
    )
    write_json(reference / "status.json", {"status": "complete", "epochs_completed": 20})
    provenance = {
        **prepared,
        **_provenance(tmp_path),
        "alignment_review_sha256": sha256(reference / "alignment-review.json"),
        "verification_sha256": sha256(original_checks),
    }
    write_json(reference / "provenance.json", provenance)
    (reference / "environment.txt").write_bytes(
        subprocess.check_output(
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
    )
    checks = tmp_path / "comparison-checks.json"
    write_json(
        checks,
        {
            "checks_passed": True,
            "runtime_hashes": runtime_hashes(tmp_path, protocol),
            "commands": [{"returncode": 0, "command": "manufactured test gate"}],
        },
    )
    yield tmp_path, reference, config_path, checks
    torch.set_num_threads(old_threads)


def test_complete_comparison_no_training_or_reference_changes(frozen_pilot, monkeypatch):
    root, reference, config, checks = frozen_pilot
    before = control.snapshot_reference(reference)

    def forbidden(*args, **kwargs):
        pytest.fail("comparison must never update optimizer/model")

    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    run = root / "runs/control"
    summary = control.run_control(config, run, project_root=root, verification_path=checks)
    assert summary["training_updates"] == 0 and summary["test_images_opened"] == 0
    assert summary["training_images"] == 32 and summary["matched_eligible_points"] == 2176
    assert summary["mse_coordinate_scalars"] == 4352 and summary["zero_support_landmarks"] == 0
    assert summary["control"]["normalized_coordinate_mse"] == pytest.approx(0, abs=1e-14)
    assert summary["cnn_reproduction_passed"]
    assert control.read_json(run / "status.json")["status"] == "complete"
    control.verify_hashes(before["original_files"])
    assert not (run / "final.pt").exists()
    with pytest.raises(FileExistsError):
        control.run_control(config, run, project_root=root, verification_path=checks)


@pytest.mark.parametrize(
    "violation", ["crop", "annotation", "membership", "checkpoint", "config", "incomplete"]
)
def test_changed_reference_is_rejected_before_comparison(frozen_pilot, violation):
    root, reference, config, checks = frozen_pilot
    if violation in ("crop", "annotation", "membership"):
        selection = control.read_json(reference / "selection.json")
        row = selection["records"][0]
        if violation == "crop":
            row["crop_box"][0] += 1
        elif violation == "annotation":
            row["annotation_sha256"] = "0" * 64
        else:
            row["official_partition"] = "infanface-test"
        write_json(reference / "selection.json", selection)
    elif violation == "checkpoint":
        with (reference / "final.pt").open("ab") as stream:
            stream.write(b"changed")
    elif violation == "config":
        content = control.read_json(reference / "config.json")
        content["source_index_base"] = 1
        write_json(reference / "config.json", content)
    else:
        write_json(reference / "status.json", {"status": "failed", "epochs_completed": 20})
    with pytest.raises(ValueError):
        control.run_control(
            config, root / "runs/rejected", project_root=root, verification_path=checks
        )
    assert not (root / "runs/rejected").exists()


def test_reconstruct_denies_test_or_unknown_reference_before_load(frozen_pilot, monkeypatch):
    _, reference, _, _ = frozen_pilot
    from infant_movement.datasets.infanface_pilot import TrainingOnlyImages

    def forbidden(*args):
        pytest.fail("must reject the row before opening pixels")

    monkeypatch.setattr(TrainingOnlyImages, "load", forbidden)
    selection = control.read_json(reference / "selection.json")
    selection["records"][0]["official_partition"] = "infanface-test"
    write_json(reference / "selection.json", selection)
    with pytest.raises(PilotError, match="unauthorized"):
        control.reconstruct_reference(reference)
    selection["records"][0]["official_partition"] = "infanface-train"
    selection["records"][0]["filename"] = "unauthorized.png"
    write_json(reference / "selection.json", selection)
    with pytest.raises(PilotError, match="absent"):
        control.reconstruct_reference(reference)


def test_failed_reproduction_preserves_original_and_records_failure(frozen_pilot):
    root, reference, config, checks = frozen_pilot
    # Summary has no older recorded digest: freeze it now but require reproduction.
    summary = control.read_json(reference / "summary.json")
    summary["fixed_final_training_fit"]["normalized_coordinate_mse"] = 999
    write_json(reference / "summary.json", summary)
    before = control.snapshot_reference(reference)
    run = root / "runs/discrepancy"
    with pytest.raises(PilotError, match="reproduction mismatch"):
        control.run_control(config, run, project_root=root, verification_path=checks)
    assert control.read_json(run / "status.json")["status"] == "failed"
    assert control.read_json(run / "cnn-reproduction.json")["discrepancies"]["original_evaluator"]
    assert not (run / "mean-shape.json").exists()
    control.verify_hashes(before["original_files"])


def test_private_destination_and_fixed_config_guards(frozen_pilot):
    root, reference, config, checks = frozen_pilot
    for destination in (reference, reference / "nested", root / "public"):
        with pytest.raises(PilotError):
            control.run_control(config, destination, project_root=root, verification_path=checks)
    write_json(config, {**control.CONTROL_CONFIG, "sample_count": 31})
    with pytest.raises(PilotError, match="fixed control"):
        control.run_control(
            config, root / "runs/rejected", project_root=root, verification_path=checks
        )


def test_undefined_json_coordinates_are_null():
    assert control.nullable_coordinates(
        torch.tensor([[float("nan"), float("nan")], [0.5, 1.0]])
    ) == [None, [0.5, 1.0]]


def test_inventory_allows_only_editable_project_revision_metadata():
    old = (
        "torch==2.14.1\npillow==13.0.0\n-e git+https://example.org/repo.git@"
        + "a" * 40
        + "#egg=infant_movement\n"
    ).encode()
    committed = old.replace(b"a" * 40, b"b" * 40)
    assert control.matching_environment_inventory(old, committed)
    assert not control.matching_environment_inventory(old, committed.replace(b"13.0.0", b"14.0.0"))
    assert not control.matching_environment_inventory(
        old, committed.replace(b"repo.git", b"other.git")
    )
    assert not control.matching_environment_inventory(old, committed + b"numpy==2.3.0\n")
    other = old.replace(b"#egg=infant_movement", b"#egg=other_package")
    assert not control.matching_environment_inventory(other, other.replace(b"a" * 40, b"b" * 40))
