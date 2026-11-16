"""Explicit, dependency-free adapters for the existing pixel-edge geometry.

Choosing an origin is a hypothesis, not evidence about dataset annotations.
The caller's affine must include all declared origin/frame/crop/padding steps.
These adapters do not inspect pixels, infer EXIF, clamp coordinates or swap slots.
"""

from infant_movement.datasets.landmark_geometry import (
    Affine2D,
    PreprocessingError,
    image_size,
    transform_landmarks,
)


def source_centres_to_edges(*, source_index_base):
    """Base-0 centre (0,0) and base-1 centre (1,1) both map to edge (0.5,0.5)."""
    if type(source_index_base) is not int or source_index_base not in (0, 1):
        raise PreprocessingError("declare source_index_base as integer 0 or 1")
    offset = 0.5 - source_index_base
    return Affine2D(c=offset, f=offset)


def encode_normalized_edges(points, *, forward, output_size):
    """Declared source numbers -> normalized edge pairs x/W,y/H; retain None/OOB."""
    width, height = image_size(output_size)
    if not isinstance(forward, Affine2D):
        raise PreprocessingError("an explicit source-to-output-edge affine is required")
    normalization = Affine2D(a=1 / width, e=1 / height)
    return transform_landmarks(points, forward.then(normalization))


def decode_normalized_edges(points, *, inverse, output_size):
    """x/W,y/H -> original source numbers using the caller's recorded inverse.

    Add no extra half pixel: the inverse already accounts for the declared origin.
    """
    width, height = image_size(output_size)
    if not isinstance(inverse, Affine2D):
        raise PreprocessingError("an explicit output-edge-to-source affine is required")
    denormalization = Affine2D(a=width, e=height)
    return transform_landmarks(points, denormalization.then(inverse))
