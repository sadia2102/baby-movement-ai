"""Reversible 2D geometry in pixel-edge coordinates; no decoder dependency.

Working raster edges are [0,width] x [0,height], with pixel centres at
(column+0.5,row+0.5). Bounds flags use half-open raster footprints. Raw source
indexing is NOT established by choosing this working convention.
"""

import math
from dataclasses import dataclass

Point = tuple[float, float]
Landmarks = tuple[Point | None, ...]


class PreprocessingError(ValueError):
    """Explicit geometry or local image inputs cannot be interpreted safely."""


def finite_number(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def image_size(size) -> tuple[int, int]:
    if (
        not isinstance(size, (tuple, list))
        or len(size) != 2
        or any(type(v) is not int or v <= 0 for v in size)
    ):
        raise PreprocessingError("size must be two positive integers in width,height order")
    return tuple(size)


@dataclass(frozen=True)
class Affine2D:
    """x'=a*x+b*y+c; y'=d*x+e*y+f. Homogeneous final row is (0,0,1)."""

    a: float = 1.0
    b: float = 0.0
    c: float = 0.0
    d: float = 0.0
    e: float = 1.0
    f: float = 0.0

    def __post_init__(self):
        if not all(finite_number(v) for row in self.matrix[:2] for v in row):
            raise PreprocessingError("affine coefficients must be finite numbers")

    @property
    def matrix(self) -> tuple[tuple[float, ...], ...]:
        return ((self.a, self.b, self.c), (self.d, self.e, self.f), (0.0, 0.0, 1.0))

    def then(self, other: "Affine2D") -> "Affine2D":
        """Apply self first, then other (matrix product other @ self)."""
        p, q = self.matrix, other.matrix
        rows = tuple(sum(q[i][k] * p[k][j] for k in range(3)) for i in range(2) for j in range(3))
        return Affine2D(*rows)

    def inverse(self) -> "Affine2D":
        determinant = self.a * self.e - self.b * self.d
        if determinant == 0 or not math.isfinite(determinant):
            raise PreprocessingError("affine transform is singular or nonfinite")
        return Affine2D(
            self.e / determinant,
            -self.b / determinant,
            (self.b * self.f - self.e * self.c) / determinant,
            -self.d / determinant,
            self.a / determinant,
            (self.d * self.c - self.a * self.f) / determinant,
        )


def transform_landmarks(points, transform: Affine2D) -> Landmarks:
    """Retain None missing points; reject malformed/nonfinite values, never zero-fill."""
    if not isinstance(points, (tuple, list)) or not points:
        raise PreprocessingError("landmarks must be a nonempty sequence")
    output = []
    for point in points:
        if point is None:
            output.append(None)
            continue
        if (
            not isinstance(point, (tuple, list))
            or len(point) != 2
            or not all(finite_number(v) for v in point)
        ):
            raise PreprocessingError("landmarks must contain finite pairs or explicit None")
        x, y = point
        mapped = (
            transform.a * x + transform.b * y + transform.c,
            transform.d * x + transform.e * y + transform.f,
        )
        if not all(math.isfinite(v) for v in mapped):
            raise PreprocessingError("transformed coordinates overflowed")
        output.append(mapped)
    return tuple(output)


def bounds_mask(points: Landmarks, size: tuple[int, int]) -> tuple[bool, ...]:
    width, height = image_size(size)
    return tuple(p is not None and 0 <= p[0] < width and 0 <= p[1] < height for p in points)


def exif_geometry(size: tuple[int, int], orientation: int | None):
    """Stored edges -> display edges. EXIF mirrors retain index i -> i.

    These are raster metadata corrections, NOT left/right augmentation. No
    anatomical index swaps are applied; the same annotated vertex moves with
    its pixel. A future augmentation needs a separately verified 68-index map.
    """
    width, height = image_size(size)
    if orientation is not None and (type(orientation) is not int or orientation not in range(1, 9)):
        raise PreprocessingError("invalid EXIF orientation; normalization/frame conversion refused")
    transforms = {
        1: Affine2D(),
        2: Affine2D(-1, 0, width, 0, 1, 0),
        3: Affine2D(-1, 0, width, 0, -1, height),
        4: Affine2D(1, 0, 0, 0, -1, height),
        5: Affine2D(0, 1, 0, 1, 0, 0),
        6: Affine2D(0, -1, height, 1, 0, 0),
        7: Affine2D(0, -1, height, -1, 0, width),
        8: Affine2D(0, 1, 0, -1, 0, width),
    }
    value = 1 if orientation is None else orientation
    return transforms[value], (height, width) if value >= 5 else (width, height)


def crop_geometry(box, size):
    """Integer half-open crop in the explicitly chosen processing raster frame."""
    width, height = image_size(size)
    if box is None:
        box = (0, 0, width, height)
    if not isinstance(box, (tuple, list)) or len(box) != 4 or any(type(v) is not int for v in box):
        raise PreprocessingError("crop must be four integer edges or None for the full raster")
    left, top, right, bottom = box
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise PreprocessingError("crop must have positive area inside the processing raster")
    return tuple(box), Affine2D(c=-left, f=-top), (right - left, bottom - top)


def letterbox_geometry(size, output_size):
    width, height = image_size(size)
    out_width, out_height = image_size(output_size)
    scale = min(out_width / width, out_height / height)
    padding = ((out_width - width * scale) / 2, (out_height - height * scale) / 2)
    # Continuous padding avoids independent width/height rounding and anisotropy.
    return Affine2D(scale, 0, padding[0], 0, scale, padding[1]), scale, padding
