"""Synthetic-only optional Torch tests, including actual ZIP/preprocessing integration."""

# Optional dependency guards must run before importing the Torch-backed modules.
# ruff: noqa: E402

import csv
import hashlib
import io
import json
from dataclasses import replace
from zipfile import ZipFile

import pytest

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")

from torch.utils.data import DataLoader

from infant_movement.datasets.infanface import COLUMNS
from infant_movement.datasets.infanface_images import open_infanface_images, preprocess_face
from infant_movement.datasets.synthetic_landmarks import SyntheticLandmarks
from infant_movement.datasets.torch_landmarks import (
    InfAnFaceTensorDataset,
    LandmarkInputConfig,
    decode_coordinates,
    prepared_to_tensors,
)
from infant_movement.engine.landmark_baseline import (
    TrainConfig,
    fit,
    load_checkpoint,
    masked_coordinate_mse,
    run_epoch,
    run_synthetic_baseline,
    save_checkpoint,
)
from infant_movement.models.landmark_baseline import SmallLandmarkCNN


@pytest.fixture(autouse=True)
def cpu_correctness_settings():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old_threads)


def synthetic_archive(tmp_path):
    image = Image.new("RGB", (8, 4), (255, 128, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image.close()
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=COLUMNS)
    writer.writeheader()
    for batch, partition, subset in [
        ("ads", "infanface-train", "infanface-train"),
        ("google", "infanface-test", "infanface-common"),
    ]:
        row = dict.fromkeys(COLUMNS, "0.0")
        row.update(
            {
                "image-set": batch,
                "filename": "synthetic.png",
                "partition": partition,
                "subpartition": subset,
            }
        )
        for i in range(68):
            row[f"gt-x{i}"], row[f"gt-y{i}"] = "1.75", "0.25"
        row["gt-x67"] = "-1.0"
        writer.writerow(row)
    path = tmp_path / "synthetic.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("infanface-public/labels.csv", text.getvalue())
        archive.writestr("infanface-public/images/ads/synthetic.png", buffer.getvalue())
        # Deliberately no test pixels: train-only iteration must not touch test images.
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_adapter_preserves_annotations_masks_partition_and_inverse(tmp_path):
    path, digest = synthetic_archive(tmp_path)
    config = LandmarkInputConfig(0, "stored", "stored", (32, 32), None)
    with open_infanface_images(path, expected_sha256=digest) as images:
        dataset = InfAnFaceTensorDataset(
            images, official_partition="infanface-train", config=config
        )
        assert len(dataset) == 1 and dataset.samples[0].project_split is None
        example = dataset[0]
        assert example["image"].shape == (3, 32, 32) and example["image"].dtype == torch.float32
        assert example["image"].is_contiguous()
        assert example["image"][:, 16, 16].tolist() == pytest.approx([1, 128 / 255, 0])
        assert example["target"][0].tolist() == pytest.approx([9 / 32, 11 / 32])
        assert example["valid"][:67].all() and not example["valid"][67]
        assert example["target"][67, 0] < 0  # No clamping.
        recovered = decode_coordinates(example["target"], (32, 32), example["inverse"])
        torch.testing.assert_close(recovered, example["source_landmarks"])
        assert example["official_partition"] == "infanface-train"
        assert example["exif_orientation"] == "absent"
        assert example["source_coordinate_evidence"] == "UNVERIFIED"
        assert dataset.samples[0].subject_id is dataset.samples[0].source_group_id is None
        # Feed the real adapter around manufactured ZIP images through the ordinary loop.
        model = SmallLandmarkCNN()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        stats = run_epoch(
            model, DataLoader(dataset, batch_size=1, num_workers=0), optimizer=optimizer
        )
        assert stats["optimizer_steps"] == 1 and stats["valid_points"] == 67
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_missing_tensor_target_kept_nan_and_worker_guard(tmp_path, monkeypatch):
    path, digest = synthetic_archive(tmp_path)
    with open_infanface_images(path, expected_sha256=digest) as images:
        face = images.load(images.samples[0])
        try:
            face = replace(
                face, sample=replace(face.sample, landmarks=(None,) + face.sample.landmarks[1:])
            )
            prepared = preprocess_face(
                face,
                source_index_base=0,
                annotation_frame="stored",
                orientation_policy="stored",
                crop_box=None,
                output_size=(32, 32),
            )
            try:
                example = prepared_to_tensors(prepared)
                assert torch.isnan(example["target"][0]).all() and not example["valid"][0]
            finally:
                prepared.image.close()
        finally:
            face.image.close()
        dataset = InfAnFaceTensorDataset(
            images,
            official_partition="infanface-train",
            config=LandmarkInputConfig(0, "stored", "stored", (32, 32), None),
        )
        monkeypatch.setattr(
            "infant_movement.datasets.torch_landmarks.get_worker_info", lambda: object()
        )
        with pytest.raises(ValueError, match="num_workers"):
            dataset[0]


def test_masked_loss_arithmetic_gradient_and_all_missing():
    prediction = torch.tensor([[[1.0, 2.0], [100.0, 100.0]]], requires_grad=True)
    target = torch.tensor([[[4.0, 6.0], [float("nan"), float("nan")]]])
    mask = torch.tensor([[True, False]])
    loss, count = masked_coordinate_mse(prediction, target, mask)
    assert count == 1 and loss.item() == 12.5
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([[[-3.0, -4.0], [0.0, 0.0]]]))
    assert masked_coordinate_mse(prediction, target, torch.zeros_like(mask)) == (None, 0)
    with pytest.raises(ValueError, match="finite"):
        masked_coordinate_mse(prediction, target, torch.ones_like(mask))


def test_all_missing_epoch_skips_optimizer_without_fake_zero_loss():
    dataset = SyntheticLandmarks(2, seed=5)
    for example in dataset.examples:
        example["valid"].fill_(False)
        example["target"].fill_(float("nan"))
        example["source_landmarks"].fill_(float("nan"))
    model = SmallLandmarkCNN()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters())
    stats = run_epoch(model, DataLoader(dataset, batch_size=2), optimizer=optimizer)
    assert stats["loss"] is None and stats["optimizer_steps"] == 0 and stats["skipped_batches"] == 1
    assert stats["source_pixel_metrics"]["mean_error"] is None
    assert not optimizer.state
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())


def test_checkpoint_load_restores_prediction_optimizer_and_rng(tmp_path):
    torch.manual_seed(3)
    model = SmallLandmarkCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loader = DataLoader(SyntheticLandmarks(2, seed=4), batch_size=2)
    run_epoch(model, loader, optimizer=optimizer)
    probe = loader.dataset[0]["image"].unsqueeze(0)
    with torch.no_grad():
        expected = model(probe)
    path = tmp_path / "owned.pt"
    digest = save_checkpoint(
        path,
        model,
        optimizer,
        epoch=1,
        config=TrainConfig(),
        loader_generator=torch.Generator().manual_seed(5),
    )
    expected_random = torch.rand(3)
    restored, restored_optimizer, payload = load_checkpoint(
        path, expected_sha256=digest, restore_rng=True
    )
    assert torch.equal(expected_random, torch.rand(3))
    with torch.no_grad():
        assert torch.equal(expected, restored(probe))
    assert restored_optimizer.state and payload["epoch"] == 1
    with pytest.raises(ValueError, match="digest"):
        load_checkpoint(path, expected_sha256="0" * 64)


@pytest.mark.parametrize("shape", [(3, 32, 32), (1, 3, 16, 16), (1, 1, 32, 32)])
def test_model_rejects_incompatible_shapes(shape):
    with pytest.raises(ValueError):
        SmallLandmarkCNN()(torch.zeros(shape))


@pytest.mark.parametrize(
    "changes",
    [
        {"device": "cuda"},
        {"epochs": 0},
        {"epochs": 201},
        {"epochs": True},
        {"learning_rate": float("nan")},
        {"train_count": 65},
        {"seed": -1},
    ],
)
def test_invalid_configuration_rejected(changes):
    with pytest.raises(ValueError):
        TrainConfig(**changes)


def test_synthetic_targets_generation_decoding_and_repeatability():
    a, b = SyntheticLandmarks(2, seed=1), SyntheticLandmarks(2, seed=1)
    assert torch.equal(a[0]["image"], b[0]["image"])
    assert not torch.equal(a[0]["image"], a[1]["image"])
    point = a[0]["target"]
    # Ellipse offsets have zero mean: target centroid is the stripe intersection.
    centre = point.mean(dim=0)
    assert torch.all((centre > 0.3) & (centre < 0.7))
    torch.testing.assert_close(point[0] - centre, torch.tensor([0.12, 0.0]), atol=1e-7, rtol=0)
    decoded = decode_coordinates(point, (32, 32), a[0]["inverse"])
    torch.testing.assert_close(decoded, a[0]["source_landmarks"])
    a.examples[0]["target"][0, 0] = float("nan")
    assert torch.isfinite(b[0]["target"]).all()


def test_batched_decode_handles_non_square_dimensions_and_unclamped_predictions():
    points = torch.tensor([[[1.25, -0.25]], [[0.5, 0.75]]])
    inverse = torch.tensor(
        [[[2, 0, -3], [0, 2, 7], [0, 0, 1]], [[1, 0, 1], [0, 1, -2], [0, 0, 1]]],
        dtype=torch.float64,
    )
    decoded = decode_coordinates(points, (8, 4), inverse)
    torch.testing.assert_close(
        decoded, torch.tensor([[[17.0, 5.0]], [[5.0, 1.0]]], dtype=torch.float64)
    )
    with pytest.raises(ValueError):
        decode_coordinates(points, (8, 4), inverse[0])


def test_bounded_run_records_no_overwrite_and_no_real_data_argument(tmp_path):
    run = tmp_path / "runs/synthetic-test"
    result = run_synthetic_baseline(
        TrainConfig(epochs=2, train_count=4, validation_count=3), run, project_root=tmp_path
    )
    assert result["training_steps"] == 2 and result["reload_predictions_equal"]
    assert result["reload_example_count"] == 3
    assert json.loads((run / "status.json").read_text())["status"] == "complete"
    assert json.loads((run / "provenance.json").read_text())["real_images_used"] == 0
    assert len((run / "epochs.jsonl").read_text().splitlines()) == 2
    with pytest.raises(FileExistsError):
        run_synthetic_baseline(TrainConfig(epochs=1), run, project_root=tmp_path)
    with pytest.raises(ValueError, match="runs"):
        run_synthetic_baseline(TrainConfig(epochs=1), tmp_path / "public", project_root=tmp_path)


@pytest.mark.parametrize("missing_partition", ["training", "validation"])
def test_fit_refuses_entirely_missing_supervision(tmp_path, missing_partition):
    training, validation = SyntheticLandmarks(2, seed=2), SyntheticLandmarks(2, seed=3)
    missing = training if missing_partition == "training" else validation
    for example in missing.examples:
        example["valid"].fill_(False)
        example["target"].fill_(float("nan"))
    model = SmallLandmarkCNN()
    optimizer = torch.optim.Adam(model.parameters())
    with pytest.raises(ValueError, match="no eligible targets"):
        fit(
            model,
            DataLoader(training, batch_size=2),
            DataLoader(validation, batch_size=2),
            optimizer=optimizer,
            config=TrainConfig(epochs=1),
            run_dir=tmp_path,
            loader_generator=torch.Generator().manual_seed(4),
        )
    assert not (tmp_path / "best.pt").exists()


@pytest.mark.parametrize(
    "invalid_kind", ["image_nan", "image_range", "loss_mask", "loss_prediction"]
)
def test_invalid_data_is_rejected_without_lowering_coverage(invalid_kind):
    if invalid_kind.startswith("image"):
        image = torch.zeros(1, 3, 32, 32)
        image[0, 0, 0, 0] = float("nan") if invalid_kind == "image_nan" else 2.0
        with pytest.raises(ValueError):
            SmallLandmarkCNN()(image)
    else:
        prediction, target = torch.zeros(1, 68, 2), torch.zeros(1, 68, 2)
        valid = torch.ones(1, 68, dtype=torch.bool)
        if invalid_kind == "loss_mask":
            valid = valid.float()
        else:
            prediction[0, 0, 0] = float("nan")
        with pytest.raises(ValueError):
            masked_coordinate_mse(prediction, target, valid)
