"""Matched training-fit arithmetic and descriptive paired differences."""

import math
import statistics

import torch

from infant_movement.datasets.torch_landmarks import decode_coordinates
from infant_movement.evaluation.landmark_metrics import landmark_errors
from infant_movement.models.mean_shape import validate_targets


def evaluate_predictions(prediction, target, valid, inverse, source_target):
    """Global scalar MSE, image-macro pixel error, and private per-image rows."""
    validate_targets(target, valid)
    if prediction.shape != target.shape or not torch.isfinite(prediction[valid]).all():
        raise ValueError("nonfinite/mismatched predictions on eligible observations")
    if source_target.shape != target.shape or not torch.isfinite(source_target[valid]).all():
        raise ValueError("invalid source targets")
    prediction, target = prediction.double(), target.double()
    source_prediction = decode_coordinates(prediction, (32, 32), inverse)
    masks = valid.tolist()
    pixel = landmark_errors((prediction * 32).tolist(), (target * 32).tolist(), masks)
    source = landmark_errors(source_prediction.tolist(), source_target.tolist(), masks)
    count = int(valid.sum())
    mse = float((prediction[valid] - target[valid]).square().mean()) if count else None
    rows = []
    for p, t, m, s, st in zip(
        prediction, target, valid, source_prediction, source_target, strict=True
    ):
        n = int(m.sum())
        rows.append(
            {
                "eligible_points": n,
                "coordinate_scalars": 2 * n,
                "normalized_coordinate_mse": float((p[m] - t[m]).square().mean()) if n else None,
                "input_pixel_error": float(
                    torch.linalg.vector_norm((p[m] - t[m]) * 32, dim=-1).mean()
                )
                if n
                else None,
                "source_pixel_error": float(torch.linalg.vector_norm(s[m] - st[m], dim=-1).mean())
                if n
                else None,
            }
        )
    return {
        "normalized_coordinate_mse": mse,
        "mse_coordinate_scalars": 2 * count,
        "input_pixel_error": pixel,
        "source_pixel_error": source,
    }, rows


def relative_reduction(control, cnn):
    """Fractional error reduction; a zero control denominator is undefined."""
    if control is None or cnn is None or control == 0:
        return None
    if not all(math.isfinite(x) and x >= 0 for x in (control, cnn)):
        raise ValueError("errors must be finite and nonnegative")
    return (control - cnn) / control


def paired_differences(cnn_rows, control_rows, *, tie_atol=1e-12):
    if not cnn_rows or len(cnn_rows) != len(control_rows):
        raise ValueError("matching nonempty per-image rows required")
    private = []
    fields = ("normalized_coordinate_mse", "input_pixel_error", "source_pixel_error")
    for cnn, control in zip(cnn_rows, control_rows, strict=True):
        if cnn["eligible_points"] != control["eligible_points"]:
            raise ValueError("paired observation denominators differ")
        row = {
            "eligible_points": cnn["eligible_points"],
            "coordinate_scalars": cnn["coordinate_scalars"],
        }
        for key in fields:
            if (cnn[key] is None) != (control[key] is None):
                raise ValueError("unmatched undefined metric")
            row[key] = {
                "cnn_minus_control": cnn[key] - control[key] if cnn[key] is not None else None,
                "relative_reduction": relative_reduction(control[key], cnn[key]),
            }
        private.append(row)
    summary = {}
    for key in fields:
        deltas = [
            r[key]["cnn_minus_control"] for r in private if r[key]["cnn_minus_control"] is not None
        ]
        summary[key] = {
            "paired_images": len(deltas),
            "undefined_images": len(private) - len(deltas),
            "mean_cnn_minus_control": statistics.mean(deltas) if deltas else None,
            "median_cnn_minus_control": statistics.median(deltas) if deltas else None,
            "min_cnn_minus_control": min(deltas) if deltas else None,
            "max_cnn_minus_control": max(deltas) if deltas else None,
            "cnn_lower_images": sum(d < -tie_atol for d in deltas),
            "control_lower_images": sum(d > tie_atol for d in deltas),
            "tied_images": sum(abs(d) <= tie_atol for d in deltas),
            "undefined_relative_images": sum(r[key]["relative_reduction"] is None for r in private),
        }
    return summary, private


def check_reproduction(new, original, *, mse_atol=1e-7, pixel_atol=1e-6, rtol=1e-6):
    """Return discrepancies for retention; never widen tolerances or replace results."""
    discrepancies = []
    for key, atol in (
        ("normalized_coordinate_mse", mse_atol),
        ("input_pixel_error", pixel_atol),
        ("source_pixel_error", pixel_atol),
    ):
        a, b = new[key], original[key]
        if key != "normalized_coordinate_mse":
            for name in (
                "image_count",
                "eligible_images",
                "eligible_points",
                "selected_point_slots",
                "point_coverage",
                "image_coverage",
                "undefined_normalizer_images",
            ):
                if a[name] != b[name]:
                    discrepancies.append(f"{key}.{name}")
            a, b = a["mean_error"], b["mean_error"]
        if (
            a is None
            or b is None
            or not math.isfinite(a)
            or not math.isfinite(b)
            or abs(a - b) > atol + rtol * abs(b)
        ):
            discrepancies.append(key)
    return discrepancies
