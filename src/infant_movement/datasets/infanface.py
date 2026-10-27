"""Read local InfAnFace landmark annotations without loading images or making splits.

This adapter targets the 144-column table observed in the project's 2026-10-27
receipt. Coordinates stay in source-image pixels; their indexing origin and
per-landmark visibility remain unspecified. No downloads or extraction occur.
"""

import csv
import io
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

ATTRIBUTES = ("turned", "occluded", "expressive", "tilted")
BATCH_PARTITIONS = {
    "ads": "infanface-train",
    "google2": "infanface-train",
    "youtube2": "infanface-train",
    "google": "infanface-test",
    "youtube": "infanface-test",
}
COORDINATE_COLUMNS = tuple(f"gt-{axis}{index}" for index in range(68) for axis in ("x", "y"))
COLUMNS = (
    "image-set",
    "filename",
    "partition",
    "subpartition",
    *ATTRIBUTES,
    *COORDINATE_COLUMNS,
)
MAX_TABLE_BYTES = 5 * 1024 * 1024


class InfAnFaceError(ValueError):
    """Annotations cannot be interpreted under the observed InfAnFace schema."""


@dataclass(frozen=True)
class FaceAttributes:
    turned: bool
    occluded: bool
    expressive: bool
    tilted: bool


@dataclass(frozen=True)
class InfAnFaceSample:
    """Dataset-native supervision, deliberately separate from an M1 split record.

    Construct samples through the validated reader. Official membership is
    retained verbatim; source groups and project assignments are not invented.
    The image-level occlusion attribute is not a landmark visibility mask.
    """

    source_row: int
    image_set: str
    filename: str
    official_split: str
    official_subpartition: str
    attributes: FaceAttributes
    landmarks: tuple[tuple[float, float], ...]

    @property
    def image_relative_path(self) -> PurePosixPath:
        return PurePosixPath("images", self.image_set, self.filename)

    @property
    def subject_id(self) -> None:
        return None

    @property
    def source_video_id(self) -> None:
        return None

    @property
    def source_group_id(self) -> None:
        return None

    @property
    def project_split(self) -> None:
        return None


def _number(value: str, row: int, column: str) -> float:
    try:
        result = float(value)
    except (ValueError, TypeError) as error:
        raise InfAnFaceError(f"row {row}: invalid numeric value in {column}") from error
    if not math.isfinite(result):
        raise InfAnFaceError(f"row {row}: nonfinite numeric value in {column}")
    return result


def _parse(text: str) -> tuple[InfAnFaceSample, ...]:
    reader = csv.DictReader(io.StringIO(text), strict=True)
    columns = reader.fieldnames
    if columns is None or len(columns) != len(COLUMNS) or set(columns) != set(COLUMNS):
        raise InfAnFaceError("expected exactly the documented 144 InfAnFace columns")
    records = []
    seen = set()
    for row_number, row in enumerate(reader, 1):
        if None in row or any(value is None for value in row.values()):
            raise InfAnFaceError(f"row {row_number}: incorrect field count")
        batch, name = row["image-set"], row["filename"]
        if batch not in BATCH_PARTITIONS:
            raise InfAnFaceError(f"row {row_number}: unsupported source batch")
        path = PurePosixPath(name)
        if (
            not name
            or name in (".", "..")
            or path.name != name
            or "\\" in name
            or ":" in name
            or any(ord(character) < 32 for character in name)
            or path.suffix.lower() not in (".jpg", ".jpeg", ".png")
        ):
            raise InfAnFaceError(f"row {row_number}: expected a safe image basename")
        key = (batch, name)
        if key in seen:
            raise InfAnFaceError(f"row {row_number}: duplicate image reference")
        seen.add(key)
        values = {
            attribute: _number(row[attribute], row_number, attribute) for attribute in ATTRIBUTES
        }
        if any(value not in (0.0, 1.0) for value in values.values()):
            raise InfAnFaceError(f"row {row_number}: attributes must be binary")
        if row["partition"] != BATCH_PARTITIONS[batch]:
            raise InfAnFaceError(
                f"row {row_number}: source batch disagrees with official partition"
            )
        expected_subpartition = (
            "infanface-train"
            if row["partition"] == "infanface-train"
            else "infanface-challenging"
            if any(values.values())
            else "infanface-common"
        )
        if row["subpartition"] != expected_subpartition:
            raise InfAnFaceError(
                f"row {row_number}: attributes disagree with official subpartition"
            )
        landmarks = tuple(
            (
                _number(row[f"gt-x{index}"], row_number, f"gt-x{index}"),
                _number(row[f"gt-y{index}"], row_number, f"gt-y{index}"),
            )
            for index in range(68)
        )
        records.append(
            InfAnFaceSample(
                source_row=row_number,
                image_set=batch,
                filename=name,
                official_split=row["partition"],
                official_subpartition=row["subpartition"],
                attributes=FaceAttributes(**{key: bool(value) for key, value in values.items()}),
                landmarks=landmarks,
            )
        )
    if not records:
        raise InfAnFaceError("annotation table must contain at least one sample")
    return tuple(records)


def _decode_and_parse(content: bytes) -> tuple[InfAnFaceSample, ...]:
    if len(content) > MAX_TABLE_BYTES:
        raise InfAnFaceError("annotation table exceeds the local reader size limit")
    try:
        return _parse(content.decode("utf-8-sig"))
    except (UnicodeDecodeError, csv.Error) as error:
        raise InfAnFaceError("annotation table must be valid UTF-8 CSV") from error


def load_infanface_csv(path: str | Path) -> tuple[InfAnFaceSample, ...]:
    """Read an explicit local table; never fetch a URL or read image pixels."""
    with Path(path).open("rb") as stream:
        return _decode_and_parse(stream.read(MAX_TABLE_BYTES + 1))


def load_infanface_archive(path: str | Path) -> tuple[InfAnFaceSample, ...]:
    """Read only the exact labels member of the observed local ZIP layout.

    No extraction occurs. Duplicate label members, encrypted labels and oversized
    tables fail closed. macOS resource forks are not annotation tables.
    """
    try:
        with ZipFile(path) as archive:
            matches = [
                item
                for item in archive.infolist()
                if item.filename == "infanface-public/labels.csv"
            ]
            if len(matches) != 1:
                raise InfAnFaceError("expected one labels table in the observed archive layout")
            member = matches[0]
            if member.flag_bits & 1 or member.file_size > MAX_TABLE_BYTES:
                raise InfAnFaceError("encrypted or oversized labels table is unsupported")
            with archive.open(member) as stream:
                return _decode_and_parse(stream.read(MAX_TABLE_BYTES + 1))
    except (BadZipFile, RuntimeError) as error:
        raise InfAnFaceError("cannot read a valid unencrypted InfAnFace archive") from error


def landmark_bounds_mask(sample: InfAnFaceSample, width: int, height: int) -> tuple[bool, ...]:
    """Flag coordinates in [0,width) × [0,height), preserving every source value.

    This is an explicit geometric QC convention, not a visibility annotation or
    a claim about the source's indexing origin. Nothing is clipped or dropped.
    """
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise InfAnFaceError("image dimensions must be positive integers")
    return tuple(0 <= x < width and 0 <= y < height for x, y in sample.landmarks)
