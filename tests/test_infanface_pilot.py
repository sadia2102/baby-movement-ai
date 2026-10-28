"""Manufactured metadata/pixels only: training-only access and oracle geometry."""

import hashlib
from dataclasses import replace
from zipfile import ZipFile

import pytest

from infant_movement.datasets.infanface_pilot import (
    PilotConfig,
    PilotError,
    ground_truth_crop,
    open_training_images,
    require_training,
    select_training_samples,
)

Image = pytest.importorskip("PIL.Image")


def test_train_filter_before_numeric_parse_and_deny_test_before_pixels(
    tmp_path, monkeypatch, pilot_archive
):
    path, digest = pilot_archive(tmp_path)
    original_open = ZipFile.open
    opened = []

    def guard(archive, name, *args, **kwargs):
        member = name.filename if hasattr(name, "filename") else name
        assert "/google/" not in member
        opened.append(member)
        return original_open(archive, name, *args, **kwargs)

    monkeypatch.setattr(ZipFile, "open", guard)
    with open_training_images(path, expected_sha256=digest) as images:
        assert len(images.samples) == 2
        assert [s.source_row for s in images.samples] == [2, 3]
        face = images.load(images.samples[0])
        assert face.orientation == 0 and face.image.size == (20, 12)
        face.image.close()
        forbidden = replace(images.samples[0], image_set="google", official_split="infanface-test")
        with pytest.raises(PilotError, match="forbidden"):
            images.load(forbidden)
        forged = replace(forbidden, official_split="infanface-train")
        with pytest.raises(PilotError, match="forbidden"):
            images.load(forged)
        assert len(images.opened_training_references) == 1
    assert len(opened) == 2  # Shared CSV plus exactly one training image.
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert [p.name for p in tmp_path.iterdir()] == ["manufactured.zip"]


def test_selection_matches_independent_sha_rule_and_rejects_mixed_pool(tmp_path, pilot_archive):
    path, digest = pilot_archive(tmp_path)
    with open_training_images(path, expected_sha256=digest) as images:
        pool = images.samples
    config = PilotConfig(sample_count=1)
    expected = min(
        pool,
        key=lambda s: hashlib.sha256(
            f"infanface-train-fit-v1|42|ads/{s.filename}".encode()
        ).hexdigest(),
    )
    assert select_training_samples(pool, config) == (expected,)
    assert select_training_samples(tuple(reversed(pool)), config) == (expected,)
    with pytest.raises(PilotError, match="duplicate"):
        select_training_samples((pool[0], pool[0]), config)
    with pytest.raises(PilotError, match="insufficient"):
        select_training_samples(pool, PilotConfig(sample_count=3))
    with pytest.raises(PilotError, match="forbidden"):
        select_training_samples((*pool, replace(pool[0], official_split="infanface-test")), config)
    with pytest.raises(PilotError):
        require_training(())


@pytest.mark.parametrize(
    "changes",
    [
        {"sample_count": 65},
        {"epochs": 21},
        {"sample_count": True},
        {"epochs": 0},
        {"batch_size": 9},
        {"seed": -1},
        {"seed": True},
        {"learning_rate": float("nan")},
        {"source_index_base": 1},
        {"source_index_base": False},
        {"orientation_policy": "normalize"},
        {"annotation_frame": "display"},
        {"crop_margin": 0.3},
        {"input_size": [64, 64]},
        {"device": "mps"},
        {"protocol_id": "other"},
    ],
)
def test_pilot_bounds_and_fixed_contract(changes):
    with pytest.raises(ValueError):
        PilotConfig(**changes)


def test_crop_offsets_rounding_excludes_oob_without_changing_annotation():
    points = ((4, 3), (14, 3), (14, 8), (4, 8), (-10, 4), None)
    assert ground_truth_crop(points, (20, 12)) == (2, 2, 17, 10)
    assert points[-2:] == ((-10, 4), None)
    assert ground_truth_crop(((0, 0), (19, 0), (19, 11)), (20, 12)) == (0, 0, 20, 12)


@pytest.mark.parametrize(
    "points",
    [
        (),
        (None,) * 4,
        ((2, 2),) * 3,
        ((1, 1), (2, 2)),
        ((0, 0), (2, 0), (4, 0)),
        ((float("nan"), 2),),
    ],
)
def test_crop_failure_is_explicit(points):
    with pytest.raises(ValueError):
        ground_truth_crop(points, (20, 12))


def test_checksum_failure_precedes_index_and_image_access(tmp_path, pilot_archive):
    path, _ = pilot_archive(tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        with open_training_images(path, expected_sha256="0" * 64):
            pytest.fail("checksum must gate access")
