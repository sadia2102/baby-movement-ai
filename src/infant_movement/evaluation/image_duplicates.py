"""Whole-image pixel equality and perceptual candidates; never infant identity.

Pillow is an optional image-audit extra, imported only when pixels are decoded.
Hashes are private derived data and must not be published per sample.
"""

import hashlib
import io
import struct
import warnings
from dataclasses import dataclass
from itertools import combinations

METHOD_VERSION = "whole-image-v1"
DHASH_MAX = 6
AHASH_MAX = 8
MAX_PIXELS = 25_000_000


class ImageAuditError(ValueError):
    """An image cannot be safely interpreted by the declared audit method."""


@dataclass(frozen=True)
class ImageView:
    name: str
    dhash: int
    ahash: int
    rgb_thumbnail: bytes


@dataclass(frozen=True)
class ImageFingerprint:
    stored_sha256: str
    display_sha256: str
    stored_size: tuple[int, int]
    display_size: tuple[int, int]
    mode: str
    orientation: int | None
    source_metadata: tuple[tuple[str, str], ...]
    views: tuple[ImageView, ...]

    def private_dict(self) -> dict:
        return {
            "stored_sha256": self.stored_sha256,
            "display_sha256": self.display_sha256,
            "stored_size": self.stored_size,
            "display_size": self.display_size,
            "mode": self.mode,
            "orientation": self.orientation,
            "source_metadata": dict(self.source_metadata),
            "views": [
                {
                    "name": view.name,
                    "dhash": f"{view.dhash:016x}",
                    "ahash": f"{view.ahash:016x}",
                    "rgb_thumbnail": view.rgb_thumbnail.hex(),
                }
                for view in self.views
            ],
        }


def _pillow():
    try:
        from PIL import Image, ImageOps
    except ImportError as error:
        raise ImageAuditError("install the hash-locked image-audit extra first") from error
    return Image, ImageOps


def _pixel_hash(image) -> str:
    # Dimensions and mode are part of equality; equal byte streams with different
    # raster shapes are not equal images. No ICC colour-management is applied.
    digest = hashlib.sha256(b"RGBA8-v1\0" + struct.pack(">II", *image.size))
    digest.update(image.tobytes())
    return digest.hexdigest()


def _view(image, name: str) -> ImageView:
    Image, _ = _pillow()
    gray = image.convert("L")
    difference = list(gray.resize((9, 8), Image.Resampling.LANCZOS).get_flattened_data())
    dhash = 0
    for y in range(8):
        for x in range(8):
            dhash = (dhash << 1) | (difference[y * 9 + x] > difference[y * 9 + x + 1])
    average = list(gray.resize((8, 8), Image.Resampling.LANCZOS).get_flattened_data())
    total = sum(average)
    ahash = 0
    for value in average:
        ahash = (ahash << 1) | (value * 64 > total)
    return ImageView(
        name,
        dhash,
        ahash,
        image.resize((16, 16), Image.Resampling.LANCZOS).tobytes(),
    )


def fingerprint_image(content: bytes) -> ImageFingerprint:
    """Decode one JPEG/PNG and return private whole-image signatures.

    Exact equality uses dimensions plus RGBA8 pixels, both as stored and after
    EXIF orientation. Near-duplicate views are white-composited RGB full-frame
    and central 80% crops, each also mirrored. No face crops or embeddings occur.
    """
    Image, ImageOps = _pillow()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as source:
                if source.format not in ("JPEG", "PNG"):
                    raise ImageAuditError("only JPEG and PNG are supported")
                if source.width * source.height > MAX_PIXELS or min(source.size) < 2:
                    raise ImageAuditError("image dimensions exceed audit limits")
                if getattr(source, "n_frames", 1) != 1:
                    raise ImageAuditError("multi-frame images require a separate audit protocol")
                if source.mode not in ("1", "L", "LA", "P", "RGB", "RGBA", "CMYK"):
                    raise ImageAuditError("unsupported high-depth or floating image mode")
                source.load()
                exif = source.getexif()
                orientation = exif.get(274)
                metadata = []
                for tag, label in (
                    (270, "description"),
                    (305, "software"),
                    (306, "datetime"),
                    (315, "artist"),
                    (33432, "copyright"),
                    (36867, "datetime_original"),
                ):
                    value = exif.get(tag)
                    if isinstance(value, str):
                        metadata.append((label, value[:2048]))
                stored = source.convert("RGBA")
                display = ImageOps.exif_transpose(source).convert("RGBA")
                stored_hash = _pixel_hash(stored)
                display_hash = _pixel_hash(display)
                rgb = Image.new("RGBA", display.size, "white")
                rgb.alpha_composite(display)
                rgb = rgb.convert("RGB")
                w, h = rgb.size
                crop = rgb.crop((int(w * 0.1), int(h * 0.1), w - int(w * 0.1), h - int(h * 0.1)))
                views = tuple(
                    _view(view, name)
                    for name, view in (
                        ("full", rgb),
                        ("full-mirror", ImageOps.mirror(rgb)),
                        ("center80", crop),
                        ("center80-mirror", ImageOps.mirror(crop)),
                    )
                )
                return ImageFingerprint(
                    stored_hash,
                    display_hash,
                    stored.size,
                    display.size,
                    source.mode,
                    orientation if isinstance(orientation, int) else None,
                    tuple(metadata),
                    views,
                )
    except ImageAuditError:
        raise
    except (
        OSError,
        ValueError,
        SyntaxError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as error:
        raise ImageAuditError("image decode failed under the declared audit limits") from error


def exact_groups(
    fingerprints: dict[str, ImageFingerprint], *, oriented: bool = True
) -> list[list[str]]:
    buckets = {}
    for key in sorted(fingerprints):
        image = fingerprints[key]
        digest = image.display_sha256 if oriented else image.stored_sha256
        buckets.setdefault(digest, []).append(key)
    return sorted(group for group in buckets.values() if len(group) > 1)


def candidate_pairs(fingerprints: dict[str, ImageFingerprint]) -> list[dict]:
    """Fixed-threshold exhaustive screening; no automatic confirmation/group IDs.

    Thresholds are inclusive, on the same view pair: dHash <=6 AND aHash <=8.
    All 4x4 view comparisons are tested; RGB thumbnail MAE ranks passing views.
    Uniform images intentionally illustrate hash collisions, not proof of origin.
    """
    result = []
    for left, right in combinations(sorted(fingerprints), 2):
        a, b = fingerprints[left], fingerprints[right]
        passing = []
        for av in a.views:
            for bv in b.views:
                dh = (av.dhash ^ bv.dhash).bit_count()
                ah = (av.ahash ^ bv.ahash).bit_count()
                if dh <= DHASH_MAX and ah <= AHASH_MAX:
                    mae = sum(
                        abs(x - y) for x, y in zip(av.rgb_thumbnail, bv.rgb_thumbnail, strict=True)
                    ) / (768 * 255)
                    passing.append((dh + ah, mae, dh, ah, av.name, bv.name))
        exact = a.display_sha256 == b.display_sha256
        if passing or exact:
            score = min(passing) if passing else (0, 0.0, 0, 0, "full", "full")
            result.append(
                {
                    "left": left,
                    "right": right,
                    "exact_display_pixels": exact,
                    "dhash_distance": score[2],
                    "ahash_distance": score[3],
                    "rgb_thumbnail_mae": score[1],
                    "left_view": score[4],
                    "right_view": score[5],
                    "review_status": "unreviewed",
                    "identity_inferred": False,
                }
            )
    return result
