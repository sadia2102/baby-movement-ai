"""Optional CPU Torch heatmap adapter; no models, image access or downloads.

All sizes are (width,height), map axes are [B,K,Hh,Wh], and coordinates are
pixel edges. Explicit inverses contain the caller's origin/frame/crop/padding.
Import this module separately; the dependency-free core never imports Torch.
"""

from dataclasses import dataclass

import torch

from infant_movement.datasets.coordinate_conventions import source_centres_to_edges
from infant_movement.datasets.landmark_geometry import finite_number, image_size
from infant_movement.datasets.torch_landmarks import decode_coordinates


@dataclass(frozen=True)
class HeatmapConfig:
    input_size: tuple[int, int] = (32, 32)
    heatmap_size: tuple[int, int] = (8, 8)
    sigma: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "input_size", image_size(self.input_size))
        object.__setattr__(self, "heatmap_size", image_size(self.heatmap_size))
        if not finite_number(self.sigma) or self.sigma <= 0:
            raise ValueError("sigma must be positive finite heatmap pixels")


DEFAULT_HEATMAP_CONFIG = HeatmapConfig()


@dataclass(frozen=True)
class HeatmapTargets:
    heatmaps: torch.Tensor
    input_edges: torch.Tensor
    heatmap_edges: torch.Tensor
    requested_eligible: torch.Tensor
    eligible: torch.Tensor
    present: torch.Tensor
    in_input_footprint: torch.Tensor
    coverage: dict


@dataclass(frozen=True)
class DecodedHeatmaps:
    heatmap_edges: torch.Tensor
    input_edges: torch.Tensor
    source_coordinates: torch.Tensor
    requested_eligible: torch.Tensor
    supported: torch.Tensor
    peak_scores: torch.Tensor
    peak_multiplicity: torch.Tensor
    ambiguous: torch.Tensor
    flat: torch.Tensor
    coverage: dict
    source_assumptions: dict


@dataclass(frozen=True)
class HeatmapLoss:
    loss: torch.Tensor | None
    denominator: int
    coverage: dict


def _cpu_float(value, name):
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype not in (torch.float32, torch.float64)
    ):
        raise ValueError(f"{name} must be a CPU float32/float64 tensor")


def _mask(mask, shape):
    if (
        not isinstance(mask, torch.Tensor)
        or mask.device.type != "cpu"
        or mask.dtype != torch.bool
        or tuple(mask.shape) != tuple(shape)
    ):
        raise ValueError("explicit CPU boolean [B,K] eligibility must match")


def _coverage(mask, cells):
    slots, points = mask.numel(), int(mask.sum())
    images = mask.shape[0]
    eligible_images = int(mask.any(dim=1).sum())
    return {
        "selected_point_slots": slots,
        "eligible_point_slots": points,
        "image_count": images,
        "eligible_images": eligible_images,
        "point_coverage": points / slots,
        "image_coverage": eligible_images / images,
        "cells_per_heatmap": cells,
        "eligible_cells": points * cells,
    }


def _map_shape(maps):
    _cpu_float(maps, "heatmaps")
    if maps.ndim != 4 or any(n == 0 for n in maps.shape):
        raise ValueError("heatmaps must have nonempty [B,K,Hh,Wh] axes")


def generate_heatmaps(input_edges, eligibility, *, config=DEFAULT_HEATMAP_CONFIG):
    """Unnormalized full-grid Gaussians; masks describe geometry, never visibility.

    Missing tensor pairs must be NaN/NaN with a false caller mask. Original None
    metadata stays with the caller. Finite OOB values are retained, not clamped;
    their effective mask is false. Border tails outside the finite grid are lost.
    Coordinates are detached targets, not differentiable model predictions.
    """
    if not isinstance(config, HeatmapConfig):
        raise ValueError("an explicit HeatmapConfig is required")
    _cpu_float(input_edges, "input_edges")
    if input_edges.ndim != 3 or input_edges.shape[-1] != 2 or 0 in input_edges.shape:
        raise ValueError("input edges must have nonempty [B,K,2] axes")
    _mask(eligibility, input_edges.shape[:2])
    coordinates = input_edges.detach().clone()
    present = torch.isfinite(coordinates).all(dim=-1)
    missing = torch.isnan(coordinates).all(dim=-1)
    if not (present | missing).all() or not present[eligibility].all():
        raise ValueError("coordinates require finite pairs or masked NaN/NaN missing pairs")
    width, height = config.input_size
    wh, hh = config.heatmap_size
    footprint = present & (coordinates[..., 0] >= 0) & (coordinates[..., 0] < width)
    footprint &= (coordinates[..., 1] >= 0) & (coordinates[..., 1] < height)
    valid = eligibility & footprint
    centres = coordinates.double() * coordinates.new_tensor(
        [wh / width, hh / height], dtype=torch.float64
    )
    if not torch.isfinite(centres[present]).all():
        raise ValueError("heatmap coordinate conversion overflowed")
    maps = torch.zeros((*coordinates.shape[:2], hh, wh), dtype=coordinates.dtype)
    if valid.any():
        centre = centres[valid]
        x = torch.arange(wh, dtype=torch.float64) + 0.5
        y = torch.arange(hh, dtype=torch.float64) + 0.5
        # Divide before squaring to avoid sigma**2 under/overflow. No normalization.
        dx = (x[None, None, :] - centre[:, 0, None, None]) / config.sigma
        dy = (y[None, :, None] - centre[:, 1, None, None]) / config.sigma
        values = torch.exp(-0.5 * (dx.square() + dy.square())).to(maps.dtype)
        if not torch.isfinite(values).all() or not (values.flatten(1).amax(dim=1) > 0).all():
            raise ValueError("eligible Gaussian lost finite representable support")
        maps[valid] = values
    coverage = {
        **_coverage(valid, hh * wh),
        "requested_eligible_slots": int(eligibility.sum()),
        "out_of_footprint_slots": int((present & ~footprint).sum()),
        "requested_out_of_footprint_slots": int((eligibility & ~footprint).sum()),
        "missing_slots": int(missing.sum()),
        "border_policy": "finite grid only; no peak/sum normalization or radius cutoff",
    }
    return HeatmapTargets(
        maps, coordinates, centres, eligibility.clone(), valid.clone(), present, footprint, coverage
    )


def _inverse_contract(inverse, batch_size):
    _cpu_float(inverse, "inverse")
    if (
        inverse.dtype != torch.float64
        or tuple(inverse.shape) != (batch_size, 3, 3)
        or not torch.isfinite(inverse).all()
        or not torch.equal(inverse[:, 2], inverse.new_tensor([0, 0, 1]).expand(batch_size, -1))
    ):
        raise ValueError("recorded inverse requires finite float64 [B,3,3] affine matrices")
    determinant = inverse[:, 0, 0] * inverse[:, 1, 1] - inverse[:, 0, 1] * inverse[:, 1, 0]
    if not torch.isfinite(determinant).all() or (determinant == 0).any():
        raise ValueError("recorded affine inverse must be nonsingular")


@torch.no_grad()
def decode_argmax(
    heatmaps,
    eligibility,
    *,
    inverse,
    source_index_base,
    annotation_frame,
    config=DEFAULT_HEATMAP_CONFIG,
):
    """Cell-centre argmax; first exact row-major maximum wins; flat -> undefined.

    Nonflat exact ties remain supported but ambiguous. Nonflat negative scores
    are permitted; raw peaks are not probabilities. No extra source half pixel
    is added: the recorded inverse already encodes the explicit source hypothesis.
    Argmax is quantized and nondifferentiable, unlike the continuous affine.
    """
    if not isinstance(config, HeatmapConfig):
        raise ValueError("an explicit HeatmapConfig is required")
    _map_shape(heatmaps)
    _mask(eligibility, heatmaps.shape[:2])
    wh, hh = config.heatmap_size
    if tuple(heatmaps.shape[2:]) != (hh, wh):
        raise ValueError("map axes disagree with configured height,width")
    source_centres_to_edges(source_index_base=source_index_base)  # Validate declaration only.
    if annotation_frame not in ("stored", "display"):
        raise ValueError("declare source annotation_frame as stored or display")
    _inverse_contract(inverse, heatmaps.shape[0])
    shape = heatmaps.shape[:2]
    coordinates = torch.full((*shape, 2), float("nan"), dtype=torch.float64)
    peaks = torch.full(shape, float("nan"), dtype=heatmaps.dtype)
    multiplicity = torch.zeros(shape, dtype=torch.int64)
    flat = torch.zeros(shape, dtype=torch.bool)
    supported = torch.zeros_like(flat)
    if eligibility.any():
        selected = heatmaps[eligibility].flatten(1)
        if not torch.isfinite(selected).all():
            raise ValueError("nonfinite eligible heatmap predictions")
        maximum, index = selected.max(dim=1)
        nonflat = maximum != selected.amin(dim=1)
        xy = torch.stack((index % wh, index // wh), dim=-1).double() + 0.5
        coordinates[eligibility] = torch.where(nonflat[:, None], xy, float("nan"))
        peaks[eligibility] = maximum
        multiplicity[eligibility] = (selected == maximum[:, None]).sum(dim=1)
        flat[eligibility] = ~nonflat
        supported[eligibility] = nonflat
    width, height = config.input_size
    inputs = coordinates * coordinates.new_tensor([width / wh, height / hh])
    normalized = inputs / inputs.new_tensor([width, height])
    source = decode_coordinates(normalized, config.input_size, inverse)
    if not torch.isfinite(source[supported]).all():
        raise ValueError("source coordinate inversion overflowed")
    ambiguous = eligibility & (multiplicity > 1)
    coverage = {
        **_coverage(supported, hh * wh),
        "requested_eligible_slots": int(eligibility.sum()),
        "unsupported_flat_slots": int(flat.sum()),
        "ambiguous_slots": int(ambiguous.sum()),
        "tie_policy": "first exact row-major maximum; multiplicity reported",
    }
    return DecodedHeatmaps(
        coordinates,
        inputs,
        source,
        eligibility.clone(),
        supported,
        peaks,
        multiplicity,
        ambiguous,
        flat,
        coverage,
        {
            "source_index_base": source_index_base,
            "annotation_frame": annotation_frame,
            "source_coordinate_evidence": "UNVERIFIED; caller-declared inverse/hypotheses",
        },
    )


def masked_heatmap_mse(prediction, target, eligibility):
    """Select before arithmetic; denominator = eligible slots * Hh * Wh.

    Float64 accumulation preserves autograd to predictions; invalid values are
    never supervised negatives. Zero eligible cells returns loss=None, never 0.
    """
    _map_shape(prediction)
    _map_shape(target)
    if prediction.shape != target.shape:
        raise ValueError("prediction/target heatmap shapes must match")
    _mask(eligibility, prediction.shape[:2])
    cells = prediction.shape[-2] * prediction.shape[-1]
    coverage = _coverage(eligibility, cells)
    denominator = coverage["eligible_cells"]
    if not denominator:
        return HeatmapLoss(None, 0, coverage)
    pred, truth = prediction[eligibility].double(), target[eligibility].double()
    if not torch.isfinite(pred).all() or not torch.isfinite(truth).all():
        raise ValueError("eligible heatmap loss values must be finite")
    loss = (pred - truth).square().sum() / denominator
    if not torch.isfinite(loss):
        raise ValueError("heatmap loss overflowed")
    return HeatmapLoss(loss, denominator, coverage)
