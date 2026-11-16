"""Opt-in reference metric and narrow legacy arithmetic; synthetic verification.

Reference: official InfAnFace commit 329abb24a858e3591d2b63f3bc0f3f5f41ac0b56,
lib/core/evaluation.py and lib/utils/transforms.py. Independent implementation
of the documented arithmetic; no upstream code is imported or vendored.
Source/raw frame and independently confirmed local slot anatomy stay UNVERIFIED.
"""

import math

from infant_movement.datasets.landmark_geometry import (
    Affine2D,
    PreprocessingError,
    finite_number,
    transform_landmarks,
)
from infant_movement.evaluation.landmark_metrics import interpoint_normalizers, landmark_errors

REFERENCE_REVISION = "329abb24a858e3591d2b63f3bc0f3f5f41ac0b56"
ASSUMED_SLOT_ORDER = "assumed-multipie68"
REFERENCE_EYE_SLOTS = (36, 45)  # Zero-based array slots observed in pinned compute_nme.


def infanface_reference_nme(predictions, targets, masks, *, slot_order, units):
    """Image-macro NME with explicit assumed slot order and common pixel units.

    Only fully eligible images reproduce the upstream all-68-point formula.
    Partial masks use the project's explicit extension: average eligible points
    per image; anchors must both be eligible with positive finite target distance.
    Undefined normalizers remain undefined without suppressing pixel coverage.
    This function does not certify dataset semantics or permission to evaluate.
    """
    if slot_order != ASSUMED_SLOT_ORDER:
        raise ValueError("explicit assumed-multipie68 slot order is required")
    if units not in ("source-pixels", "input-pixels"):
        raise ValueError("declare common source-pixels or input-pixels for all coordinates")
    if not targets or len(predictions) != len(targets) or len(masks) != len(targets):
        raise ValueError("matching nonempty image sequences required")
    for prediction, target, mask in zip(predictions, targets, masks, strict=True):
        if not (len(prediction) == len(target) == len(mask) == 68):
            raise ValueError("reference convention requires exactly 68 slots")
        if any(type(value) is not bool for value in mask):
            raise ValueError("validity masks must contain booleans")
    normalizers = interpoint_normalizers(targets, masks, index_pair=REFERENCE_EYE_SLOTS)
    report = landmark_errors(predictions, targets, masks, normalizers=normalizers)
    if any(
        value is not None and not math.isfinite(value)
        for value in (report["mean_error"], report["normalized_mean_error"])
    ):
        raise ValueError("metric arithmetic overflowed")
    fraction = report["normalized_mean_error"]
    percent = 100 * fraction if fraction is not None else None
    if percent is not None and not math.isfinite(percent):
        raise ValueError("metric percentage overflowed")
    full = sum(
        all(mask) and denominator is not None
        for mask, denominator in zip(masks, normalizers, strict=True)
    )
    report.update(
        {
            "nme_fraction": fraction,
            "nme_percent": percent,
            "units": units,
            "normalizer": "ground-truth distance between zero-based slots 36 and 45",
            "normalizer_index_pair": list(REFERENCE_EYE_SLOTS),
            "slot_order": slot_order,
            "upstream_revision": REFERENCE_REVISION,
            "fully_eligible_reference_images": full,
            "normalizer_defined_images": sum(value is not None for value in normalizers),
            "undefined_normalizer_images_total": sum(value is None for value in normalizers),
            "no_eligible_point_images": len(targets) - report["eligible_images"],
            "masked_extension_normalized_images": report["normalized_eligible_images"] - full,
            "source_coordinate_evidence": "UNVERIFIED",
            "local_anatomical_order_evidence": "UNVERIFIED",
        }
    )
    return report


def legacy_hrnet_pixel_reference(point, *, center, scale, resolution, inverse):
    """Audited subset: zero rotation, square resolution, one-based point interface.

    The upstream pixel transform subtracts 1, applies a centre/scale affine,
    truncates toward zero, then adds 1. Quantization prevents general inversion.
    This is NOT the reversible preprocessing adapter, a crop implementation or
    complete HRNet decoding parity. The upstream positive-y label guard is not
    adopted: raw y=0 can be eligible under our explicitly chosen base-0 policy.
    """
    if type(resolution) is not int or resolution <= 0:
        raise PreprocessingError("reference supports a positive integer square resolution")
    if type(inverse) is not bool:
        raise PreprocessingError("declare a boolean transform direction")
    if not finite_number(scale) or scale <= 0:
        raise PreprocessingError("reference scale must be positive and finite")
    # Existing strict pair validation rejects missing/nonfinite/bool coordinates.
    checked_point = transform_landmarks((point,), Affine2D())[0]
    checked_center = transform_landmarks((center,), Affine2D())[0]
    if checked_point is None or checked_center is None:
        raise PreprocessingError("reference point and center must be finite pairs")
    extent = 200 * scale
    if not math.isfinite(extent):
        raise PreprocessingError("reference extent overflowed")
    factor = resolution / extent
    affine = Affine2D(
        a=factor,
        e=factor,
        c=resolution / 2 - checked_center[0] * factor,
        f=resolution / 2 - checked_center[1] * factor,
    )
    if inverse:
        affine = affine.inverse()
    shifted = (checked_point[0] - 1, checked_point[1] - 1)
    transformed = transform_landmarks((shifted,), affine)[0]
    return tuple(math.trunc(value) + 1 for value in transformed)
