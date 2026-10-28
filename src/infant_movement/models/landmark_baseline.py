"""Randomly initialized coordinate regressor; no pretrained assets or downloads."""

import torch
from torch import nn


class SmallLandmarkCNN(nn.Module):
    """32x32 RGB -> 68 normalized edge-coordinate pairs, unconstrained linear head.

    Flattening retains position; no global-average pooling that discards stripe
    location. Output coordinates may fall outside the image and are not clamped.
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 68 * 2),
        )

    def forward(self, image):
        if image.ndim != 4 or tuple(image.shape[1:]) != (3, 32, 32):
            raise ValueError("baseline requires [B,3,32,32] images")
        if image.dtype != torch.float32 or not torch.isfinite(image).all():
            raise ValueError("image tensor must be finite float32")
        if (image < 0).any() or (image > 1).any():
            raise ValueError("RGB tensor values must be in [0,1]")
        return self.layers(image).reshape(-1, 68, 2)
