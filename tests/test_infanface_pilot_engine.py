"""Optional Torch safeguards on manufactured ZIP images, never infant data."""

# ruff: noqa: E402
import json
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL.Image")

from torch.utils.data import DataLoader

from infant_movement.datasets.infanface_pilot import PilotConfig, PilotError, open_training_images
from infant_movement.datasets.synthetic_landmarks import SyntheticLandmarks
from infant_movement.engine import infanface_pilot as engine
from infant_movement.models.landmark_baseline import SmallLandmarkCNN


@pytest.fixture
def prepared(tmp_path, pilot_archive):
    archive, digest = pilot_archive(tmp_path)
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Manufactured test protocol only.")
    run = tmp_path / "runs/manufactured"
    coverage = engine.prepare_pilot(
        PilotConfig(sample_count=2, epochs=1, batch_size=2),
        archive,
        digest,
        run,
        project_root=tmp_path,
        protocol_path=protocol,
    )
    review = json.loads((run / "alignment-review-template.json").read_text())
    review.update(
        approved=True,
        gross_alignment_supported=True,
        reviewed_usable_images=coverage["usable_images"],
        reviewer="synthetic test",
    )
    (run / "alignment-review.json").write_text(json.dumps(review))
    checks = tmp_path / "manufactured-checks.json"
    checks.write_text(
        json.dumps(
            {
                "checks_passed": True,
                "runtime_hashes": engine.runtime_hashes(tmp_path, protocol),
                "commands": [{"command": "manufactured test fixture", "returncode": 0}],
            }
        )
    )
    return tmp_path, run, checks, coverage


def test_preparation_no_model_or_replacement_and_invalid_exif_retained(
    tmp_path, monkeypatch, pilot_archive
):
    def forbidden():
        pytest.fail("model must not be created during preparation")

    monkeypatch.setattr(engine, "SmallLandmarkCNN", forbidden)
    archive, digest = pilot_archive(tmp_path, degenerate=True)
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Synthetic only")
    run = tmp_path / "runs/prepared"
    coverage = engine.prepare_pilot(
        PilotConfig(sample_count=2),
        archive,
        digest,
        run,
        project_root=tmp_path,
        protocol_path=protocol,
    )
    assert coverage["selected_images"] == 2 and coverage["usable_images"] == 1
    assert coverage["excluded_images"] == 1 and coverage["eligible_point_slots"] == 68
    assert coverage["exif_orientation_counts"] == {"0": 2}
    assert coverage["test_images_opened"] == 0 and not coverage["project_splits_created"]
    records = json.loads((run / "selection.json").read_text())["records"]
    assert all(
        r["subject_id"] is r["source_video_id"] is r["project_split"] is None for r in records
    )
    assert (run / "qc-page-01.png").is_file() and not (run / "final.pt").exists()
    assert json.loads((run / "status.json").read_text())["execution_started"] is False
    with pytest.raises(FileExistsError):
        engine.prepare_pilot(
            PilotConfig(sample_count=2),
            archive,
            digest,
            run,
            project_root=tmp_path,
            protocol_path=protocol,
        )


def test_end_to_end_fixed_final_reload_and_second_run_rejected(prepared):
    root, run, checks, coverage = prepared
    assert coverage["eligible_point_slots"] == 134
    assert coverage["mask_false_counts_nonexclusive"]["source_in_bounds"] == 2
    summary = engine.execute_pilot(run, project_root=root, verification_path=checks)
    assert summary["epochs"] == summary["optimizer_steps"] == 1
    assert summary["reload_predictions_equal"] and summary["reload_max_absolute_difference"] == 0
    assert summary["validation"] is None and summary["anatomical_nme"] is None
    assert summary["checkpoint_policy"] == "fixed-final-epoch"
    assert (
        not summary["held_out_performance_claim"]
        and not summary["patient_video_independence_claim"]
    )
    assert not (run / "best.pt").exists()
    model, optimizer, payload = engine.load_final_pilot_checkpoint(
        run / "final.pt", expected_sha256=summary["final_checkpoint_sha256"]
    )
    assert isinstance(model, SmallLandmarkCNN) and optimizer.state and payload["epoch"] == 1
    assert json.loads((run / "provenance.json").read_text())["fresh_random_initialization"] is True
    assert json.loads((run / "status.json").read_text())["status"] == "complete"
    with pytest.raises(PilotError, match="rerunning"):
        engine.execute_pilot(run, project_root=root, verification_path=checks)
    with pytest.raises(PilotError, match="digest"):
        engine.load_final_pilot_checkpoint(run / "final.pt", expected_sha256="0" * 64)


@pytest.mark.parametrize(
    "violation",
    [
        "review",
        "coverage_count",
        "review_hash",
        "checks",
        "failed_command",
        "protocol_changed",
        "selection_changed",
        "already_started",
    ],
)
def test_execution_gate_precedes_model_and_training(prepared, monkeypatch, violation):
    root, run, checks, _ = prepared

    def forbidden():
        pytest.fail("gate must fail before model initialization")

    monkeypatch.setattr(engine, "SmallLandmarkCNN", forbidden)
    if violation in ("review", "coverage_count", "review_hash"):
        review = json.loads((run / "alignment-review.json").read_text())
        if violation == "review":
            review["gross_alignment_supported"] = False
        if violation == "coverage_count":
            review["reviewed_usable_images"] = 1
        if violation == "review_hash":
            review["selection_sha256"] = "0" * 64
        (run / "alignment-review.json").write_text(json.dumps(review))
    elif violation in ("checks", "failed_command"):
        record = json.loads(checks.read_text())
        if violation == "checks":
            record["checks_passed"] = False
        else:
            record["commands"][0]["returncode"] = 1
        checks.write_text(json.dumps(record))
    elif violation == "protocol_changed":
        (root / "protocol.md").write_text("changed")
    elif violation == "selection_changed":
        with (run / "selection.json").open("a") as stream:
            stream.write(" ")
    else:
        (run / "status.json").write_text(
            json.dumps({"status": "prepared", "execution_started": True})
        )
    with pytest.raises(PilotError):
        engine.execute_pilot(run, project_root=root, verification_path=checks)
    assert not (run / "final.pt").exists()


def test_all_missing_epoch_does_not_step_or_claim_zero_loss():
    dataset = SyntheticLandmarks(2, seed=1)
    for example in dataset.examples:
        example["valid"].fill_(False)
        example["target"].fill_(float("nan"))
    model = SmallLandmarkCNN()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters())
    with pytest.raises(PilotError, match="no eligible"):
        engine.pilot_epoch(model, DataLoader(dataset, batch_size=2), optimizer)
    assert not optimizer.state
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_partial_missing_targets_and_metric_arithmetic():
    dataset = SyntheticLandmarks(1, seed=2)
    example = dataset.examples[0]
    example["target"][0] = float("nan")
    example["source_landmarks"][0] = float("nan")
    example["valid"][0] = False

    class Fixed(torch.nn.Module):
        def forward(self, image):
            value = example["target"].clone()
            value[0] = 0
            value[1:, 0] += 3 / 32
            value[1:, 1] += 4 / 32
            return value.unsqueeze(0)

    summary, _ = engine.training_fit(Fixed(), DataLoader(dataset, batch_size=1))
    assert summary["input_pixel_error"]["mean_error"] == pytest.approx(5, abs=1e-5)
    assert summary["source_pixel_error"]["mean_error"] == pytest.approx(5, abs=1e-5)
    assert summary["normalized_coordinate_mse"] == pytest.approx(12.5 / 1024, abs=1e-7)
    assert summary["input_pixel_error"]["eligible_points"] == 67


def test_prepared_dataset_rejects_test_and_clones_tensors(tmp_path, pilot_archive):
    archive, digest = pilot_archive(tmp_path)
    with open_training_images(archive, expected_sha256=digest) as images:
        sample = images.samples[0]
    example = {"official_partition": "infanface-train", "image": torch.zeros(3, 32, 32)}
    dataset = engine.PreparedTrainingExamples((sample,), (example,))
    dataset[0]["image"].fill_(1)
    assert not example["image"].any()
    with pytest.raises(PilotError):
        engine.PreparedTrainingExamples(
            (replace(sample, official_split="infanface-test"),), (example,)
        )
    with pytest.raises(PilotError):
        engine.PreparedTrainingExamples(
            (sample,), ({**example, "official_partition": "infanface-test"},)
        )


def test_public_or_symlink_escaped_output_rejected(tmp_path):
    with pytest.raises(PilotError, match="runs"):
        engine._run_path(tmp_path / "public", tmp_path)
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs/escape").symlink_to(tmp_path)
    with pytest.raises(PilotError, match="runs"):
        engine._run_path(tmp_path / "runs/escape/private", tmp_path)
