"""Training-only access, deterministic pilot selection and explicit oracle crops."""

import csv
import hashlib
import io
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass, replace
from zipfile import BadZipFile, ZipFile

from infant_movement.datasets.infanface import (
    BATCH_PARTITIONS,
    COLUMNS,
    MAX_TABLE_BYTES,
    InfAnFaceSample,
    _decode_and_parse,
)
from infant_movement.datasets.infanface_images import InfAnFaceImages, _verify
from infant_movement.datasets.landmark_geometry import finite_number, image_size


class PilotError(ValueError):
    """A training-only pilot safeguard or declared data condition failed."""


@dataclass(frozen=True)
class PilotConfig:
    protocol_id: str = "infanface-train-fit-v1"
    sample_count: int = 32
    seed: int = 42
    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 0.003
    crop_margin: float = 0.2
    source_index_base: int = 0
    annotation_frame: str = "stored"
    orientation_policy: str = "stored"
    input_size: tuple[int, int] = (32, 32)
    device: str = "cpu"

    def __post_init__(self):
        for name, maximum in (("sample_count", 64), ("epochs", 20), ("batch_size", 8)):
            if type(getattr(self, name)) is not int or not 1 <= getattr(self, name) <= maximum:
                raise PilotError("pilot sample/epoch/batch bound violated")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise PilotError("pilot seed must be a nonnegative 32-bit integer")
        if (
            type(self.learning_rate) not in (int, float)
            or not math.isfinite(self.learning_rate)
            or not 0 < self.learning_rate <= 0.1
        ):
            raise PilotError("invalid pilot learning rate")
        if (
            self.protocol_id != "infanface-train-fit-v1"
            or self.crop_margin != 0.2
            or type(self.source_index_base) is not int
            or self.source_index_base != 0
            or self.annotation_frame != "stored"
            or self.orientation_policy != "stored"
            or image_size(self.input_size) != (32, 32)
            or self.device != "cpu"
        ):
            raise PilotError("pilot-v1 requires its explicit fixed coordinate/crop/CPU contract")
        object.__setattr__(self, "input_size", tuple(self.input_size))


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def require_training(samples):
    if not samples or len(samples) > 64:
        raise PilotError("pilot requires 1..64 explicitly selected training samples")
    seen = set()
    for sample in samples:
        if (
            not isinstance(sample, InfAnFaceSample)
            or sample.official_split != "infanface-train"
            or BATCH_PARTITIONS.get(sample.image_set) != "infanface-train"
        ):
            raise PilotError("official test/non-training samples are forbidden")
        key = (sample.image_set, sample.filename)
        if key in seen:
            raise PilotError("duplicate pilot sample reference")
        seen.add(key)


def select_training_samples(pool, config: PilotConfig):
    # Validate the entire metadata pool before ranking; require_training's cap
    # applies to selected examples, not the metadata-only official training pool.
    for sample in pool:
        require_training((sample,))
    keys = [(s.image_set, s.filename) for s in pool]
    if len(set(keys)) != len(keys) or len(pool) < config.sample_count:
        raise PilotError("duplicate references or insufficient official training pool")

    def rank(sample):
        text = f"{config.protocol_id}|{config.seed}|{sample.image_set}/{sample.filename}"
        return hashlib.sha256(text.encode()).hexdigest(), sample.image_set, sample.filename

    result = tuple(sorted(pool, key=rank)[: config.sample_count])
    require_training(result)
    return result


def _read_training_table(content):
    if len(content) > MAX_TABLE_BYTES:
        raise PilotError("shared index exceeds the established table limit")
    try:
        reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")), strict=True)
        columns = reader.fieldnames
        if columns is None or len(columns) != len(COLUMNS) or set(columns) != set(COLUMNS):
            raise PilotError("unsupported shared annotation index schema")
        training, source_rows = [], []
        for source_row, row in enumerate(reader, 1):
            # Do not numerically parse test landmarks/attributes or use them in
            # any eligibility/selection decision. Only official membership gates access.
            partition = row.get("partition")
            if partition == "infanface-test":
                continue
            if partition != "infanface-train":
                raise PilotError("unknown official membership in shared index")
            if None in row or any(value is None for value in row.values()):
                raise PilotError("malformed training index row")
            training.append(row)
            source_rows.append(source_row)
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(training)
        samples = _decode_and_parse(stream.getvalue().encode())
        return tuple(
            replace(s, source_row=row) for s, row in zip(samples, source_rows, strict=True)
        )
    except (UnicodeDecodeError, csv.Error) as error:
        raise PilotError("invalid UTF-8 shared CSV index") from error


class TrainingOnlyImages(InfAnFaceImages):
    """Reuses the existing bounded decoder, with a deny-first partition guard."""

    def __init__(self, archive, samples):
        super().__init__(archive, samples)
        self.opened_training_references = []

    def load(self, sample):
        require_training((sample,))
        face = super().load(sample)
        self.opened_training_references.append((sample.image_set, sample.filename))
        return face


@contextmanager
def open_training_images(path, *, expected_sha256):
    """Verify the container, read its shared index, open no test image payloads.

    The shared CSV is filtered on official membership before landmark parsing.
    Existing generic loader/annotation contracts and the raw archive are unchanged.
    """
    _verify(path, expected_sha256)
    try:
        with ZipFile(path) as archive:
            members = [i for i in archive.infolist() if i.filename == "infanface-public/labels.csv"]
            if (
                len(members) != 1
                or members[0].flag_bits & 1
                or members[0].file_size > MAX_TABLE_BYTES
            ):
                raise PilotError("missing/duplicate/encrypted/oversized shared index")
            with archive.open(members[0]) as stream:
                samples = _read_training_table(stream.read(MAX_TABLE_BYTES + 1))
            yield TrainingOnlyImages(archive, samples)
    except (BadZipFile, RuntimeError, OSError) as error:
        raise PilotError("cannot use the local training-only image archive") from error
    finally:
        _verify(path, expected_sha256)


def ground_truth_crop(landmarks, size, *, source_index_base=0, margin=0.2):
    """Oracle box; retain annotations unchanged and clip only crop box edges."""
    width, height = image_size(size)
    if type(source_index_base) is not int or source_index_base != 0 or margin != 0.2:
        raise PilotError("crop requires the fixed pilot-v1 base-0/margin hypotheses")
    points = []
    for point in landmarks:
        if point is None:
            continue
        if len(point) != 2 or not all(finite_number(v) for v in point):
            raise PilotError("crop landmarks must be finite pairs or explicit None")
        x, y = point[0] + 0.5, point[1] + 0.5
        if 0 <= x < width and 0 <= y < height:
            points.append((x, y))
    if len(points) < 3:
        raise PilotError("ground-truth crop has fewer than three eligible points")
    left, right = min(p[0] for p in points), max(p[0] for p in points)
    top, bottom = min(p[1] for p in points), max(p[1] for p in points)
    if left == right or top == bottom:
        raise PilotError("ground-truth crop has a degenerate landmark span")
    dx, dy = (right - left) * margin, (bottom - top) * margin
    box = (
        max(0, math.floor(left - dx)),
        max(0, math.floor(top - dy)),
        min(width, math.ceil(right + dx)),
        min(height, math.ceil(bottom + dy)),
    )
    if not (box[0] < box[2] and box[1] < box[3]):
        raise PilotError("ground-truth crop has no raster area")
    return box
