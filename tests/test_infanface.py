"""Synthetic-only checks for the independent InfAnFace annotation reader."""

import csv
import io
import warnings
from dataclasses import FrozenInstanceError
from zipfile import ZipFile

import pytest

from infant_movement.datasets.infanface import (
    InfAnFaceError,
    landmark_bounds_mask,
    load_infanface_archive,
    load_infanface_csv,
)


def synthetic_row(batch="ads", name="synthetic.jpg", attribute="0.0"):
    training = batch in ("ads", "google2", "youtube2")
    row = {
        "image-set": batch,
        "filename": name,
        "partition": "infanface-train" if training else "infanface-test",
        "subpartition": (
            "infanface-train"
            if training
            else "infanface-challenging"
            if float(attribute)
            else "infanface-common"
        ),
        "turned": attribute,
        "occluded": "0.0",
        "expressive": "0.0",
        "tilted": "0.0",
    }
    for index in range(68):
        row[f"gt-x{index}"] = "10.25"
        row[f"gt-y{index}"] = "20.75"
    return row


def table(rows, columns=None):
    stream = io.StringIO()
    columns = columns or list(rows[0])
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def read(tmp_path, rows):
    path = tmp_path / "synthetic.csv"
    path.write_bytes(table(rows))
    before = path.read_bytes()
    result = load_infanface_csv(path)
    assert path.read_bytes() == before
    return result


def test_preserves_fractional_coordinates_attributes_membership_and_unknowns(tmp_path):
    samples = read(tmp_path, [synthetic_row(), synthetic_row("youtube", attribute="1.0")])
    first, second = samples
    assert len(first.landmarks) == 68
    assert first.landmarks[0] == (10.25, 20.75)
    assert first.image_relative_path.as_posix() == "images/ads/synthetic.jpg"
    assert first.official_split == "infanface-train"
    assert second.official_subpartition == "infanface-challenging"
    assert second.attributes.turned is True
    assert second.attributes.occluded is False
    assert first.subject_id is first.source_video_id is first.source_group_id is None
    assert first.project_split is None
    assert (first.source_row, second.source_row) == (1, 2)
    with pytest.raises(FrozenInstanceError):
        first.official_split = "test"


def test_reads_bom_and_reordered_columns(tmp_path):
    path = tmp_path / "synthetic.csv"
    row = synthetic_row()
    path.write_bytes(b"\xef\xbb\xbf" + table([row], list(reversed(row))))
    assert load_infanface_csv(path)[0].landmarks[0] == (10.25, 20.75)


@pytest.mark.parametrize(
    "column,value",
    [
        ("gt-x0", "nan"),
        ("gt-y67", "inf"),
        ("gt-x1", ""),
        ("gt-y2", "text"),
        ("turned", "2.0"),
        ("occluded", "false"),
        ("partition", "test"),
        ("subpartition", "infanface-common"),
        ("image-set", "other"),
    ],
)
def test_invalid_values_fail_without_echoing_private_filename(tmp_path, column, value):
    row = synthetic_row(name="private-source-name.jpg")
    row[column] = value
    path = tmp_path / "synthetic.csv"
    path.write_bytes(table([row]))
    with pytest.raises(InfAnFaceError) as result:
        load_infanface_csv(path)
    assert "private-source-name" not in str(result.value)


@pytest.mark.parametrize(
    "name",
    [
        "../escape.jpg",
        "/absolute.jpg",
        "dir/file.jpg",
        "dir\\file.jpg",
        "C:escape.jpg",
        "\nimage.jpg",
        "https://example.org/image.jpg",
        "file.py",
    ],
)
def test_unsafe_or_nonimage_references_rejected(tmp_path, name):
    with pytest.raises(InfAnFaceError, match="basename"):
        read(tmp_path, [synthetic_row(name=name)])


def test_duplicate_references_rejected_but_same_basename_in_different_batch_allowed(tmp_path):
    with pytest.raises(InfAnFaceError, match="duplicate image reference"):
        read(tmp_path, [synthetic_row(), synthetic_row()])
    assert len(read(tmp_path, [synthetic_row(), synthetic_row("google2")])) == 2


@pytest.mark.parametrize(
    "variant",
    ["missing", "extra", "duplicate", "empty", "invalid_encoding", "extra_value", "missing_value"],
)
def test_schema_and_csv_failures(tmp_path, variant):
    row = synthetic_row()
    content = table([row])
    if variant == "missing":
        del row["gt-y67"]
        content = table([row])
    elif variant == "extra":
        row["subject_id"] = "fabricated"
        content = table([row])
    elif variant == "duplicate":
        content = content.replace(b"gt-y67", b"gt-x67", 1)
    elif variant == "empty":
        content = content.splitlines(keepends=True)[0]
    elif variant == "invalid_encoding":
        content = b"\xff" + content
    elif variant == "extra_value":
        content = content.rstrip() + b",unexpected\n"
    elif variant == "missing_value":
        content = content.rsplit(b",", 1)[0] + b"\n"
    path = tmp_path / "synthetic.csv"
    path.write_bytes(content)
    with pytest.raises(InfAnFaceError):
        load_infanface_csv(path)


def test_geometric_qc_preserves_out_of_bounds_annotation(tmp_path):
    row = synthetic_row()
    row["gt-x0"] = "-0.5"
    row["gt-y67"] = "100.0"
    sample = read(tmp_path, [row])[0]
    mask = landmark_bounds_mask(sample, 100, 100)
    assert mask[0] is False and mask[67] is False
    assert sum(mask) == 66
    assert sample.landmarks[0] == (-0.5, 20.75)
    assert sample.landmarks[67] == (10.25, 100.0)


@pytest.mark.parametrize(
    "width,height", [(True, 100), (100, False), (0, 100), (-1, 100), (100, 10.5)]
)
def test_qc_rejects_invalid_dimensions(tmp_path, width, height):
    sample = read(tmp_path, [synthetic_row()])[0]
    with pytest.raises(InfAnFaceError):
        landmark_bounds_mask(sample, width, height)


def test_local_archive_reads_only_real_table_and_extracts_nothing(tmp_path):
    path = tmp_path / "synthetic.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("infanface-public/labels.csv", table([synthetic_row()]))
        archive.writestr("__MACOSX/infanface-public/._labels.csv", b"\xffbinary sidecar")
        archive.writestr("../../not-extracted.py", b"raise RuntimeError('must not execute')")
    before = set(tmp_path.iterdir())
    assert len(load_infanface_archive(path)) == 1
    assert set(tmp_path.iterdir()) == before


@pytest.mark.parametrize("variant", ["missing", "duplicate", "oversized", "not_zip"])
def test_archive_member_limits(tmp_path, variant):
    path = tmp_path / "synthetic.zip"
    if variant == "not_zip":
        path.write_bytes(b"not a zip file")
    else:
        with ZipFile(path, "w") as archive:
            if variant not in ("missing", "oversized"):
                archive.writestr("infanface-public/labels.csv", table([synthetic_row()]))
            if variant == "duplicate":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    archive.writestr("infanface-public/labels.csv", table([synthetic_row()]))
            if variant == "oversized":
                archive.writestr("infanface-public/labels.csv", b"x" * (5 * 1024 * 1024 + 1))
    with pytest.raises(InfAnFaceError):
        load_infanface_archive(path)


def test_oversized_csv_rejected(tmp_path):
    path = tmp_path / "synthetic.csv"
    path.write_bytes(b"x" * (5 * 1024 * 1024 + 1))
    with pytest.raises(InfAnFaceError, match="size limit"):
        load_infanface_csv(path)
