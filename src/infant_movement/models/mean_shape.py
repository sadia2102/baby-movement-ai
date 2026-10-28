"""Constant landmark predictor fitted from targets/masks only, never pixels."""

from dataclasses import dataclass

import torch


def validate_targets(targets, valid):
    if (
        targets.ndim != 3
        or targets.shape[-1] != 2
        or not targets.is_floating_point()
        or targets.device.type != "cpu"
        or valid.device.type != "cpu"
        or valid.dtype != torch.bool
        or valid.shape != targets.shape[:-1]
        or not targets.shape[0]
        or not targets.shape[1]
    ):
        raise ValueError("expected nonempty CPU floating [N,K,2] targets and boolean [N,K] mask")
    if not torch.isfinite(targets[valid]).all():
        raise ValueError("eligible targets must be finite")


@dataclass(frozen=True)
class MeanShape:
    coordinates: torch.Tensor
    support: torch.Tensor

    def predict(self, count):
        if type(count) is not int or count <= 0:
            raise ValueError("prediction count must be positive")
        return self.coordinates.unsqueeze(0).expand(count, -1, -1).clone()


def fit_mean_shape(targets, valid):
    """Float64 means of eligible x/y scalars; unsupported slots remain NaN.

    Optimizes point-weighted coordinate MSE among constant predictors on this
    fitting set, not Euclidean error. No images/features can enter this API.
    """
    validate_targets(targets, valid)
    support = valid.sum(dim=0)
    coordinates = torch.full((targets.shape[1], 2), float("nan"), dtype=torch.float64)
    for landmark in range(targets.shape[1]):
        if support[landmark]:
            coordinates[landmark] = targets[valid[:, landmark], landmark].double().mean(dim=0)
    return MeanShape(coordinates, support)
