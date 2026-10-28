"""Explicit local ZIP image access and reversible InfAnFace preprocessing.

Uses the existing optional pinned Pillow decoder; no extraction, networking,
model transforms, caches, split generation or annotation mutation.
"""

import hashlib
import io
import re
import stat
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

from infant_movement.datasets.infanface import InfAnFaceSample, load_infanface_archive
from infant_movement.datasets.landmark_geometry import (
    Affine2D,
    Landmarks,
    PreprocessingError,
    bounds_mask,
    crop_geometry,
    exif_geometry,
    image_size,
    letterbox_geometry,
    transform_landmarks,
)

METHOD_VERSION = "infanface-preprocessing-v1"
MAX_IMAGE_BYTES = 32 * 1024**2
MAX_PIXELS = 25_000_000


def _pillow():
    try:
        from PIL import Image
    except ImportError as error:
        raise PreprocessingError("use the existing hash-locked image-audit environment") from error
    return Image


def _verify(path, expected):
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise PreprocessingError("expected_sha256 must be a lowercase SHA-256 digest")
    with Path(path).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected:
        raise PreprocessingError("archive checksum mismatch")


@dataclass(frozen=True)
class LoadedFace:
    sample: InfAnFaceSample
    image: Any  # Owned Pillow RGB raster, in stored orientation; caller closes it.
    orientation: int | None
    source_mode: str


@dataclass(frozen=True)
class PreparedFace:
    source: InfAnFaceSample
    image: Any  # New metadata-free RGB raster; caller closes it.
    landmarks: Landmarks  # Output pixel-edge coordinates; same index order.
    forward: Affine2D  # Raw source annotation numbers -> output edge coordinates.
    inverse: Affine2D  # Output edge coordinates -> raw source annotation numbers.
    present: tuple[bool, ...]
    source_in_bounds: tuple[bool, ...]
    crop_in_bounds: tuple[bool, ...]
    output_in_bounds: tuple[bool, ...]
    crop_box: tuple[int, int, int, int]
    crop_kind: str
    source_index_base: int
    annotation_frame: str
    orientation_policy: str
    orientation: int | None
    scale: float
    padding_xy: tuple[float, float]
    source_coordinate_evidence: str = "UNVERIFIED"


class InfAnFaceImages:
    """Use only within open_infanface_images; samples come from the validated table."""

    def __init__(self, archive, samples):
        self._archive = archive
        self.samples = samples
        self._members = {}
        for info in archive.infolist():
            self._members.setdefault(info.filename, []).append(info)

    def load(self, sample: InfAnFaceSample) -> LoadedFace:
        if self._archive.fp is None:
            raise PreprocessingError("image archive context is closed")
        if not isinstance(sample, InfAnFaceSample) or sample not in self.samples:
            raise PreprocessingError("sample must belong to this archive's validated annotations")
        name = "infanface-public/" + sample.image_relative_path.as_posix()
        members = self._members.get(name, [])
        if len(members) != 1:
            raise PreprocessingError("image reference must resolve to exactly one ZIP member")
        member = members[0]
        if (
            member.flag_bits & 1
            or member.file_size > MAX_IMAGE_BYTES
            or member.is_dir()
            or stat.S_ISLNK(member.external_attr >> 16)
        ):
            raise PreprocessingError("encrypted, oversized or non-regular image member")
        try:
            with self._archive.open(member) as stream:
                content = stream.read(MAX_IMAGE_BYTES + 1)
            if len(content) > MAX_IMAGE_BYTES:
                raise PreprocessingError("image member exceeds the byte limit")
            return _decode(content, sample)
        except (OSError, BadZipFile, RuntimeError) as error:
            raise PreprocessingError("cannot read the local image member") from error


@contextmanager
def open_infanface_images(path: str | Path, *, expected_sha256: str):
    """Verify local archive before/after use; read exact members, never extract.

    The digest must come from the private acquisition receipt. It establishes
    local integrity, not independently authenticated publisher provenance.
    """
    try:
        _verify(path, expected_sha256)
        samples = load_infanface_archive(path)
        try:
            with ZipFile(path) as archive:
                yield InfAnFaceImages(archive, samples)
        finally:
            _verify(path, expected_sha256)
    except (OSError, BadZipFile, RuntimeError) as error:
        raise PreprocessingError("cannot use the verified local image archive") from error


def _decode(content, sample):
    Image = _pillow()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as image:
                if image.format not in ("JPEG", "PNG") or getattr(image, "n_frames", 1) != 1:
                    raise PreprocessingError("only single-frame JPEG/PNG images are supported")
                if image.width * image.height > MAX_PIXELS or min(image.size) < 1:
                    raise PreprocessingError("image exceeds the pixel limit")
                if image.mode not in ("1", "L", "LA", "P", "RGB", "RGBA", "CMYK"):
                    raise PreprocessingError("unsupported high-depth/floating image mode")
                image.load()
                orientation = image.getexif().get(274)
                rgba = image.convert("RGBA")
                try:
                    # Fixed white alpha background; never copy EXIF/XMP to outputs.
                    with Image.new("RGBA", image.size, (255, 255, 255, 255)) as background:
                        with Image.alpha_composite(background, rgba) as composed:
                            rgb = composed.convert("RGB")
                finally:
                    rgba.close()
                rgb.info.clear()
                return LoadedFace(sample, rgb, orientation, image.mode)
    except (
        OSError,
        ValueError,
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
    ) as error:
        if isinstance(error, PreprocessingError):
            raise
        raise PreprocessingError("cannot decode a supported local image") from error


def preprocess_face(
    face: LoadedFace,
    *,
    source_index_base: int,
    annotation_frame: str,
    orientation_policy: str,
    crop_box: tuple[int, int, int, int] | None,
    output_size: tuple[int, int],
    fill_rgb: tuple[int, int, int] = (0, 0, 0),
) -> PreparedFace:
    """Explicit crop, uniform resize and centred padding with reversible geometry.

    Source base 0 means raw (0,0) names the first pixel centre; base 1 means
    (1,1) does. Neither hypothesis is verified. annotation_frame is 'stored'
    or 'display'; orientation_policy is 'stored' or 'normalize'. Crops are
    integer edges in the policy's image frame, never inferred from landmarks.
    No horizontal-flip augmentation or anatomical index reordering is exposed.
    """
    Image = _pillow()
    if type(source_index_base) is not int or source_index_base not in (0, 1):
        raise PreprocessingError("explicit source_index_base must be 0 or 1")
    if annotation_frame not in ("stored", "display") or orientation_policy not in (
        "stored",
        "normalize",
    ):
        raise PreprocessingError("explicit annotation frame/orientation policy is unsupported")
    output_size = image_size(output_size)
    if output_size[0] * output_size[1] > MAX_PIXELS:
        raise PreprocessingError("output exceeds the pixel limit")
    if (
        not isinstance(fill_rgb, tuple)
        or len(fill_rgb) != 3
        or any(type(v) is not int or not 0 <= v <= 255 for v in fill_rgb)
    ):
        raise PreprocessingError("fill_rgb must contain three integer bytes")
    if not isinstance(face, LoadedFace) or face.image.mode != "RGB":
        raise PreprocessingError("expected an owned RGB LoadedFace")
    if len(face.sample.landmarks) != 68:
        raise PreprocessingError("InfAnFace requires all 68 landmark slots")
    size = image_size(face.image.size)
    # Conversion is required only when the caller explicitly requests that frame.
    orientation, display_size = (
        exif_geometry(size, face.orientation)
        if annotation_frame == "display" or orientation_policy == "normalize"
        else (Affine2D(), size)
    )
    source_edges = Affine2D(c=0.5 - source_index_base, f=0.5 - source_index_base)
    annotation_size = display_size if annotation_frame == "display" else size
    native_points = transform_landmarks(face.sample.landmarks, source_edges)
    source_mask = bounds_mask(native_points, annotation_size)
    to_stored = orientation.inverse() if annotation_frame == "display" else Affine2D()
    to_processing = orientation if orientation_policy == "normalize" else Affine2D()
    forward = source_edges.then(to_stored).then(to_processing)
    processing_size = display_size if orientation_policy == "normalize" else size
    box, crop_transform, crop_size = crop_geometry(crop_box, processing_size)
    forward = forward.then(crop_transform)
    cropped_points = transform_landmarks(face.sample.landmarks, forward)
    resize, scale, padding = letterbox_geometry(crop_size, output_size)
    forward = forward.then(resize)
    points = transform_landmarks(face.sample.landmarks, forward)
    operations = {
        2: Image.Transpose.FLIP_LEFT_RIGHT,
        3: Image.Transpose.ROTATE_180,
        4: Image.Transpose.FLIP_TOP_BOTTOM,
        5: Image.Transpose.TRANSPOSE,
        6: Image.Transpose.ROTATE_270,
        7: Image.Transpose.TRANSVERSE,
        8: Image.Transpose.ROTATE_90,
    }
    operation = operations.get(face.orientation) if orientation_policy == "normalize" else None
    raster = face.image.copy() if operation is None else face.image.transpose(operation)
    try:
        with raster.crop(box) as cropped:
            inverse_resize = resize.inverse()
            data = tuple(v for row in inverse_resize.matrix[:2] for v in row)
            output = cropped.transform(
                output_size,
                Image.Transform.AFFINE,
                data,
                resample=Image.Resampling.BICUBIC,
                fillcolor=fill_rgb,
            )
    finally:
        raster.close()
    output.info.clear()
    return PreparedFace(
        face.sample,
        output,
        points,
        forward,
        forward.inverse(),
        tuple(p is not None for p in points),
        source_mask,
        bounds_mask(cropped_points, crop_size),
        bounds_mask(points, output_size),
        box,
        "full-raster" if crop_box is None else "caller-supplied",
        source_index_base,
        annotation_frame,
        orientation_policy,
        face.orientation,
        scale,
        padding,
    )
