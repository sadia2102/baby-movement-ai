"""Manufactured moving stripes with 68 translated ellipse points, not infant faces."""

import math

import torch
from torch.utils.data import Dataset


class SyntheticLandmarks(Dataset):
    """Fixed local generator; train/validation seeds manufacture different inputs.

    RGB channels show Gaussian vertical/horizontal stripes and their intersection.
    All 68 target points are a fixed ellipse translated with that intersection.
    This tests spatial regression, not infant landmark semantics/generalization.
    """

    def __init__(self, count: int, *, seed: int, size: int = 32):
        if type(count) is not int or not 1 <= count <= 64 or size != 32:
            raise ValueError("bounded synthetic task requires 1..64 images at 32x32")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        centres = 0.3 + 0.4 * torch.rand(count, 2, generator=generator)
        axis = (torch.arange(size, dtype=torch.float32) + 0.5) / size
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        angle = torch.arange(68, dtype=torch.float32) * (2 * math.pi / 68)
        offsets = torch.stack((0.12 * angle.cos(), 0.10 * angle.sin()), dim=-1)
        self.examples = []
        for centre in centres:
            red = torch.exp(-(((xx - centre[0]) / 0.06) ** 2))
            green = torch.exp(-(((yy - centre[1]) / 0.06) ** 2))
            points = centre + offsets
            self.examples.append(
                {
                    "image": torch.stack((red, green, red * green)),
                    "target": points,
                    "valid": torch.ones(68, dtype=torch.bool),
                    "inverse": torch.tensor(
                        [[1, 0, -0.5], [0, 1, -0.5], [0, 0, 1]], dtype=torch.float64
                    ),
                    "source_landmarks": points.double() * size - 0.5,
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return {key: value.clone() for key, value in self.examples[index].items()}
