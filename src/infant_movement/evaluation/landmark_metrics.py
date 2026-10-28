"""Independent Python arithmetic for image-macro landmark errors and coverage.

No anatomical eye indices are inferred. An optional denominator must be supplied
explicitly per image; undefined/nonpositive denominators have no NME, no fallback.
"""

import math


def _finite_pair(point):
    return (
        isinstance(point, (tuple, list))
        and len(point) == 2
        and all(type(v) in (int, float) and math.isfinite(v) for v in point)
    )


def landmark_errors(predictions, targets, masks, *, normalizers=None, indices=None):
    """Pixels/other declared common units in; image-macro Euclidean error out.

    Supports any fixed K, including all 0..67 baseline slots or explicit subsets.
    Each eligible image has equal weight, NOT each point or unidentified patient.
    Nonfinite predictions at an eligible point fail rather than disappearing.
    """
    if not predictions or not (len(predictions) == len(targets) == len(masks)):
        raise ValueError("metric requires matching nonempty image sequences")
    k = len(targets[0])
    selected = tuple(range(k)) if indices is None else tuple(indices)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(type(i) is not int or not 0 <= i < k for i in selected)
    ):
        raise ValueError("metric indices must be distinct valid landmark indices")
    if normalizers is None:
        normalizers = [None] * len(targets)
    if len(normalizers) != len(targets):
        raise ValueError("one explicit normalizer per image is required")
    image_errors, normalized = [], []
    point_count = undefined = 0
    for pred, target, mask, denominator in zip(
        predictions, targets, masks, normalizers, strict=True
    ):
        if not (len(pred) == len(target) == len(mask) == k) or any(
            type(v) is not bool for v in mask
        ):
            raise ValueError("metric shapes/mask types disagree")
        distances = []
        for i in selected:
            if mask[i]:
                if not _finite_pair(pred[i]) or not _finite_pair(target[i]):
                    raise ValueError("eligible predictions/targets must be finite pairs")
                distances.append(math.hypot(pred[i][0] - target[i][0], pred[i][1] - target[i][1]))
        point_count += len(distances)
        if not distances:
            continue
        error = sum(distances) / len(distances)
        image_errors.append(error)
        if type(denominator) in (int, float) and math.isfinite(denominator) and denominator > 0:
            normalized.append(error / denominator)
        else:
            undefined += 1
    return {
        "mean_error": sum(image_errors) / len(image_errors) if image_errors else None,
        "normalized_mean_error": sum(normalized) / len(normalized) if normalized else None,
        "image_count": len(targets),
        "eligible_images": len(image_errors),
        "eligible_points": point_count,
        "selected_point_slots": len(targets) * len(selected),
        "point_coverage": point_count / (len(targets) * len(selected)),
        "image_coverage": len(image_errors) / len(targets),
        "normalized_eligible_images": len(normalized),
        "undefined_normalizer_images": undefined,
    }


def interpoint_normalizers(targets, masks, *, index_pair):
    """Caller-declared pair, not an established InfAnFace eye convention."""
    if (
        len(index_pair) != 2
        or index_pair[0] == index_pair[1]
        or any(type(i) is not int or i < 0 for i in index_pair)
    ):
        raise ValueError("normalization requires two distinct explicit indices")
    if len(targets) != len(masks):
        raise ValueError("normalizer shapes disagree")
    result = []
    for points, mask in zip(targets, masks, strict=True):
        if len(points) != len(mask) or max(index_pair) >= len(points):
            raise ValueError("normalizer indices/shapes disagree")
        left, right = index_pair
        if not (
            mask[left]
            and mask[right]
            and _finite_pair(points[left])
            and _finite_pair(points[right])
        ):
            result.append(None)
            continue
        value = math.dist(points[left], points[right])
        result.append(value if value > 0 else None)
    return result
