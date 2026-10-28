"""Synthetic pixels only: reversible geometry, frame declarations and ZIP limits."""

import csv
import hashlib
import io
import warnings
from dataclasses import replace
from zipfile import ZipFile, ZipInfo

import pytest

from infant_movement.datasets.infanface import COLUMNS
from infant_movement.datasets.infanface_images import (
    open_infanface_images,
    preprocess_face,
)
from infant_movement.datasets.landmark_geometry import (
    Affine2D,
    PreprocessingError,
    exif_geometry,
    transform_landmarks,
)

Image = pytest.importorskip("PIL.Image")


def make_archive(tmp_path, *, orientation=None, mode="RGB", variant=None, size=(6, 4)):
    path = tmp_path / "synthetic.zip"
    image = Image.new(mode, size)
    if mode == "RGB":
        for y in range(size[1]):
            for x in range(size[0]):
                image.putpixel((x, y), (x * 25, y * 25, 50))
    else:
        image.paste(0, (0, 0, *size))
    buffer = io.BytesIO()
    exif = Image.Exif()
    if orientation is not None:
        exif[274] = orientation
    exif[315] = "synthetic-private-tag"
    image.save(buffer, format="PNG", exif=exif)
    image.close()
    row = dict.fromkeys(COLUMNS, "0.0")
    row.update(
        {
            "image-set": "ads",
            "filename": "synthetic.png",
            "partition": "infanface-train",
            "subpartition": "infanface-train",
        }
    )
    for i in range(68):
        row[f"gt-x{i}"], row[f"gt-y{i}"] = "1.0", "0.0"
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerow(row)
    with ZipFile(path, "w") as archive:
        archive.writestr("infanface-public/labels.csv", text.getvalue())
        member = "infanface-public/images/ads/synthetic.png"
        content = b"invalid pixels" if variant == "corrupt" else buffer.getvalue()
        if variant == "symlink":
            info = ZipInfo(member)
            info.create_system = 3
            info.external_attr = 0o120777 << 16
            archive.writestr(info, content)
        elif variant != "missing":
            archive.writestr(member, content)
        if variant == "duplicate":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                archive.writestr(member, content)
        archive.writestr("../../not-extracted.txt", b"must never be extracted")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(face, **changes):
    kwargs = dict(
        source_index_base=0,
        annotation_frame="stored",
        orientation_policy="stored",
        crop_box=None,
        output_size=face.image.size,
    )
    kwargs.update(changes)
    return preprocess_face(face, **kwargs)


def assert_roundtrip(result):
    back = transform_landmarks(result.landmarks, result.inverse)
    for original, recovered in zip(result.source.landmarks, back, strict=True):
        if original is None:
            assert recovered is None
        else:
            assert recovered == pytest.approx(original, abs=1e-10)


def test_archive_load_is_local_detached_and_preserves_raw(tmp_path):
    path, digest = make_archive(tmp_path)
    before = {p.name for p in tmp_path.iterdir()}
    with open_infanface_images(path, expected_sha256=digest) as reader:
        sample = reader.samples[0]
        face = reader.load(sample)
        assert face.image.mode == "RGB" and face.image.size == (6, 4)
        assert face.image.info == {}
        assert sample.subject_id is sample.source_video_id is sample.project_split is None
        with pytest.raises(PreprocessingError, match="belong"):
            reader.load(replace(sample, filename="../private-source.png"))
    assert face.image.getpixel((1, 0)) == (25, 0, 50)
    with pytest.raises(PreprocessingError, match="closed"):
        reader.load(sample)
    face.image.close()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert {p.name for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize(
    "orientation,expected",
    [
        (1, (1, 0)),
        (2, (4, 0)),
        (3, (4, 3)),
        (4, (1, 3)),
        (5, (0, 1)),
        (6, (3, 1)),
        (7, (3, 4)),
        (8, (0, 4)),
    ],
)
def test_all_exif_operations_move_marker_and_same_landmark_index(tmp_path, orientation, expected):
    path, digest = make_archive(tmp_path, orientation=orientation)
    with open_infanface_images(path, expected_sha256=digest) as reader:
        face = reader.load(reader.samples[0])
        original_pixels = face.image.tobytes()
        _, size = exif_geometry(face.image.size, orientation)
        result = prepare(face, orientation_policy="normalize", output_size=size)
        assert result.image.size == size
        assert result.landmarks[0] == (expected[0] + 0.5, expected[1] + 0.5)
        assert result.image.getpixel(expected) == face.image.getpixel((1, 0))
        assert result.image.info == {}  # Never reapply orientation or leak EXIF tags.
        assert result.source is face.sample and result.source.landmarks == ((1.0, 0.0),) * 68
        assert face.image.tobytes() == original_pixels
        assert_roundtrip(result)
        result.image.close()
        face.image.close()


@pytest.mark.parametrize("policy", ["stored", "normalize"])
def test_display_annotations_converted_consistently(tmp_path, policy):
    path, digest = make_archive(tmp_path, orientation=6)
    with open_infanface_images(path, expected_sha256=digest) as reader:
        face = reader.load(reader.samples[0])
        # Display coordinates of stored marker (1,0) under orientation 6.
        face = replace(face, sample=replace(face.sample, landmarks=((3.0, 1.0),) * 68))
        size = (4, 6) if policy == "normalize" else (6, 4)
        result = prepare(
            face, annotation_frame="display", orientation_policy=policy, output_size=size
        )
        expected = (3, 1) if policy == "normalize" else (1, 0)
        assert result.landmarks[0] == (expected[0] + 0.5, expected[1] + 0.5)
        assert result.image.getpixel(expected) == (25, 0, 50)
        assert_roundtrip(result)
        result.image.close()
        face.image.close()


@pytest.mark.parametrize("orientation", [None, 0, 1, 2, 6, 8])
def test_stored_policy_never_silently_rotates(tmp_path, orientation):
    path, digest = make_archive(tmp_path, orientation=orientation)
    with open_infanface_images(path, expected_sha256=digest) as reader:
        face = reader.load(reader.samples[0])
        result = prepare(face)
        assert result.image.tobytes() == face.image.tobytes()
        assert result.landmarks[0] == (1.5, 0.5)
        assert result.orientation == orientation
        assert result.source_coordinate_evidence == "UNVERIFIED"
        result.image.close()
        face.image.close()


@pytest.mark.parametrize("frame,policy", [("stored", "normalize"), ("display", "stored")])
def test_invalid_orientation_refuses_frame_conversion(tmp_path, frame, policy):
    path, digest = make_archive(tmp_path, orientation=0)
    with open_infanface_images(path, expected_sha256=digest) as reader:
        face = reader.load(reader.samples[0])
        with pytest.raises(PreprocessingError, match="EXIF"):
            prepare(face, annotation_frame=frame, orientation_policy=policy)
        face.image.close()


def test_crop_offset_uniform_scale_padding_and_missing_flags(tmp_path):
    path, digest = make_archive(tmp_path, size=(10, 6))
    with open_infanface_images(path, expected_sha256=digest) as reader:
        face = reader.load(reader.samples[0])
        points = ((3.0, 2.0), (-1.0, 0.0), (0.0, 0.0), None) + ((3.0, 2.0),) * 64
        face = replace(face, sample=replace(face.sample, landmarks=points))
        result = prepare(face, crop_box=(2, 1, 8, 5), output_size=(12, 12))
        assert result.scale == 2 and result.padding_xy == (0, 2)
        assert result.landmarks[0] == (3, 5)
        assert result.landmarks[1] == (-5, 1)  # Retained, never clamped.
        assert result.landmarks[3] is None and not result.present[3]
        assert result.source_in_bounds[:4] == (True, False, True, False)
        assert result.crop_in_bounds[:4] == (True, False, False, False)
        assert result.output_in_bounds[:4] == (True, False, False, False)
        assert result.image.getpixel((5, 0)) == (0, 0, 0)
        assert result.image.getpixel((5, 11)) == (0, 0, 0)
        assert result.crop_kind == "caller-supplied"
        assert result.source.landmarks == points
        assert_roundtrip(result)
        result.image.close()
        face.image.close()


@pytest.mark.parametrize("size,output", [((7, 3), (13, 13)), ((3, 7), (12, 9)), ((1, 1), (1, 5))])
def test_non_square_fractional_padding_has_one_exact_scale(tmp_path, size, output):
    path, digest = make_archive(tmp_path, size=size)
    with open_infanface_images(path, expected_sha256=digest) as reader:
        face = reader.load(reader.samples[0])
        result = prepare(face, output_size=output)
        assert result.forward.a == result.forward.e
        assert result.scale == min(output[0] / size[0], output[1] / size[1])
        assert result.padding_xy[0] * 2 + size[0] * result.scale == pytest.approx(output[0])
        assert result.padding_xy[1] * 2 + size[1] * result.scale == pytest.approx(output[1])
        assert result.image.size == output
        assert_roundtrip(result)
        result.image.close()
        face.image.close()


def test_zero_and_one_index_hypotheses_require_explicit_choice(tmp_path):
    path, digest = make_archive(tmp_path)
    with open_infanface_images(path, expected_sha256=digest) as reader:
        face = reader.load(reader.samples[0])
        base0 = prepare(face)
        shifted = replace(face, sample=replace(face.sample, landmarks=((2.0, 1.0),) * 68))
        base1 = prepare(shifted, source_index_base=1)
        assert base0.landmarks == base1.landmarks
        assert base1.inverse.c == 0.5
        with pytest.raises(TypeError):
            preprocess_face(face)  # No hidden geometry assumptions.
        base0.image.close()
        base1.image.close()
        face.image.close()


def test_crop_after_orientation_and_resize_aligns_pixel_centres(tmp_path):
    path, digest = make_archive(tmp_path, orientation=6, size=(10, 6))
    with open_infanface_images(path, expected_sha256=digest) as reader:
        face = reader.load(reader.samples[0])
        face = replace(face, sample=replace(face.sample, landmarks=((3.0, 2.0),) * 68))
        result = prepare(
            face, orientation_policy="normalize", crop_box=(1, 2, 5, 8), output_size=(12, 18)
        )
        assert result.landmarks[0] == (7.5, 4.5)
        assert result.image.getpixel((7, 4)) == face.image.getpixel((3, 2))
        assert_roundtrip(result)
        result.image.close()
        face.image.close()


@pytest.mark.parametrize(
    "changes",
    [
        {"source_index_base": True},
        {"source_index_base": 2},
        {"annotation_frame": "auto"},
        {"orientation_policy": "auto"},
        {"output_size": (0, 4)},
        {"output_size": (True, 4)},
        {"output_size": (3.5, 4)},
        {"output_size": (10000, 10000)},
        {"fill_rgb": (0, -1, 0)},
        {"crop_box": (-1, 0, 4, 3)},
        {"crop_box": (1, 1, 1, 3)},
        {"crop_box": (0, 0, 7, 4)},
        {"crop_box": (0.5, 0, 4, 3)},
    ],
)
def test_invalid_preprocessing_inputs_fail(tmp_path, changes):
    path, digest = make_archive(tmp_path)
    with open_infanface_images(path, expected_sha256=digest) as reader:
        face = reader.load(reader.samples[0])
        with pytest.raises(PreprocessingError):
            prepare(face, **changes)
        face.image.close()


@pytest.mark.parametrize("variant", ["missing", "duplicate", "corrupt", "symlink"])
def test_invalid_archive_images_fail_without_extracting_or_echoing_names(tmp_path, variant):
    path, digest = make_archive(tmp_path, variant=variant)
    with open_infanface_images(path, expected_sha256=digest) as reader:
        with pytest.raises(PreprocessingError) as error:
            reader.load(reader.samples[0])
        assert "synthetic.png" not in str(error.value)
    assert {p.name for p in tmp_path.iterdir()} == {"synthetic.zip"}


def test_checksum_checked_before_parsing_and_after_context(tmp_path):
    path, digest = make_archive(tmp_path)
    with pytest.raises(PreprocessingError, match="checksum"):
        with open_infanface_images(path, expected_sha256="0" * 64):
            pytest.fail("must fail before yielding")
    with pytest.raises(PreprocessingError, match="checksum"):
        with open_infanface_images(path, expected_sha256=digest):
            with path.open("ab") as stream:
                stream.write(b"synthetic mutation")


@pytest.mark.parametrize("limit", ["MAX_IMAGE_BYTES", "MAX_PIXELS"])
def test_byte_and_pixel_limits(tmp_path, monkeypatch, limit):
    import infant_movement.datasets.infanface_images as module

    path, digest = make_archive(tmp_path)
    monkeypatch.setattr(module, limit, 1)
    with open_infanface_images(path, expected_sha256=digest) as reader:
        with pytest.raises(PreprocessingError):
            reader.load(reader.samples[0])


@pytest.mark.parametrize("mode", ["L", "RGBA", "P", "I;16"])
def test_colour_modes_and_high_depth_rejection(tmp_path, mode):
    path, digest = make_archive(tmp_path, mode=mode)
    with open_infanface_images(path, expected_sha256=digest) as reader:
        if mode == "I;16":
            with pytest.raises(PreprocessingError, match="depth"):
                reader.load(reader.samples[0])
        else:
            face = reader.load(reader.samples[0])
            assert face.source_mode == mode and face.image.mode == "RGB"
            if mode == "RGBA":
                assert face.image.getpixel((0, 0)) == (255, 255, 255)
            face.image.close()


def test_affine_composition_and_fractional_roundtrip():
    transform = Affine2D(c=-8, f=3).then(Affine2D(0.25, 0, 6.75, 0, 0.25, 7.125))
    points = ((-2.25, 150.125), (80.5, -30.75), None)
    recovered = transform_landmarks(transform_landmarks(points, transform), transform.inverse())
    assert recovered[:2] == points[:2] and recovered[2] is None
    assert transform.matrix[2] == (0, 0, 1)
    with pytest.raises(PreprocessingError, match="singular"):
        Affine2D(0, 0, 0, 0, 0, 0).inverse()


def test_exif_mirror_preserves_semantic_indices_and_crop_mask_is_not_output_mask(tmp_path):
    path, digest = make_archive(tmp_path, orientation=2)
    with open_infanface_images(path, expected_sha256=digest) as reader:
        face = reader.load(reader.samples[0])
        points = ((0.0, 1.0), (5.0, 1.0)) + face.sample.landmarks[2:]
        face = replace(face, sample=replace(face.sample, landmarks=points))
        mirrored = prepare(face, orientation_policy="normalize")
        assert mirrored.landmarks[:2] == ((5.5, 1.5), (0.5, 1.5))
        mirrored.image.close()
        # A real image point outside a short crop can land in output padding.
        padded = prepare(face, crop_box=(0, 1, 6, 3), output_size=(6, 6))
        assert padded.crop_in_bounds[2] is False
        assert padded.output_in_bounds[2] is True
        assert padded.source_in_bounds[2] is True
        padded.image.close()
        face.image.close()


@pytest.mark.parametrize("orientation,expected_code", [(None, 0), (0, 1)])
def test_smoke_script_aggregate_success_and_orientation_rejection(
    tmp_path, orientation, expected_code
):
    import json
    import subprocess
    import sys
    from pathlib import Path

    path, digest = make_archive(tmp_path, orientation=orientation)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"sha256": digest}))
    script = Path(__file__).resolve().parents[1] / "scripts/smoke_infanface_preprocessing.py"
    output = tmp_path / "reports/private/smoke.json"
    command = [
        sys.executable,
        str(script),
        "--archive",
        str(path),
        "--receipt",
        str(receipt),
        "--source-index-base",
        "0",
        "--annotation-frame",
        "stored",
        "--orientation-policy",
        "normalize",
        "--output",
        str(output),
    ]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert result.returncode == expected_code, result.stderr
    report = json.loads(result.stdout)
    assert report["sample_count"] == report["decoded_count"] == 1
    assert report["preprocessing_rejected_count"] == expected_code
    assert report["processed_count"] == 1 - expected_code
    assert report["original_annotation_changes"] == 0
    assert report["max_roundtrip_error_source_pixels"] < 1e-10
    assert report["project_splits_created"] is False
    assert "synthetic.png" not in result.stdout
    assert "synthetic-private-tag" not in result.stdout
    assert json.loads(output.read_text()) == report
    second = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert second.returncode == 2 and json.loads(output.read_text()) == report
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_smoke_script_rejects_public_output_before_reading_input(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts/smoke_infanface_preprocessing.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--archive",
            "not-opened",
            "--receipt",
            "not-opened",
            "--source-index-base",
            "0",
            "--annotation-frame",
            "stored",
            "--orientation-policy",
            "stored",
            "--output",
            "public.json",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2 and not (tmp_path / "public.json").exists()


@pytest.mark.parametrize("points", [[], [(float("nan"), 0)], [(True, 2)], [(0,)], [0]])
def test_invalid_landmarks_are_not_replaced_with_zeros(points):
    with pytest.raises(PreprocessingError):
        transform_landmarks(points, Affine2D())
