"""Optional CPU tensor adapter; explicit preprocessing, no splits or downloads."""

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, get_worker_info

from infant_movement.datasets.infanface_images import preprocess_face
from infant_movement.datasets.landmark_geometry import image_size


@dataclass(frozen=True)
class LandmarkInputConfig:
    source_index_base: int
    annotation_frame: str
    orientation_policy: str
    output_size: tuple[int, int]
    crop_box: tuple[int, int, int, int] | None


def prepared_to_tensors(prepared):
    """CHW RGB float32 [0,1]; targets are x/W,y/H output edge coordinates.

    Missing targets are NaN plus a false mask, never zeros. Present but out-of-
    bounds targets are retained numerically. Validity is geometric eligibility,
    not visibility: present AND source/crop/output footprint checks.
    """
    width, height = prepared.image.size
    pixels = torch.frombuffer(bytearray(prepared.image.tobytes()), dtype=torch.uint8)
    image = pixels.reshape(height, width, 3).permute(2, 0, 1).contiguous().float() / 255
    points = [(float("nan"), float("nan")) if p is None else p for p in prepared.landmarks]
    target = torch.tensor(points, dtype=torch.float32) / torch.tensor([width, height])
    raw = [(float("nan"), float("nan")) if p is None else p for p in prepared.source.landmarks]
    masks = {
        key: torch.tensor(getattr(prepared, key), dtype=torch.bool)
        for key in ("present", "source_in_bounds", "crop_in_bounds", "output_in_bounds")
    }
    valid = (
        masks["present"]
        & masks["source_in_bounds"]
        & masks["crop_in_bounds"]
        & masks["output_in_bounds"]
    )
    return {
        "image": image,
        "target": target,
        "valid": valid,
        **masks,
        "forward": torch.tensor(prepared.forward.matrix, dtype=torch.float64),
        "inverse": torch.tensor(prepared.inverse.matrix, dtype=torch.float64),
        "source_landmarks": torch.tensor(raw, dtype=torch.float64),
        "official_partition": prepared.source.official_split,
        "source_row": prepared.source.source_row,
        "source_index_base": prepared.source_index_base,
        "annotation_frame": prepared.annotation_frame,
        "orientation_policy": prepared.orientation_policy,
        "exif_orientation": "absent" if prepared.orientation is None else str(prepared.orientation),
        "crop_kind": prepared.crop_kind,
        "source_coordinate_evidence": prepared.source_coordinate_evidence,
    }


class InfAnFaceTensorDataset(Dataset):
    """Wrap an open verified image context; num_workers=0 is required.

    The explicit official partition filters metadata only. It is NOT a project
    train/validation split. No models are called by this adapter.
    """

    def __init__(self, images, *, official_partition: str, config: LandmarkInputConfig):
        if official_partition not in ("infanface-train", "infanface-test"):
            raise ValueError("an explicit official InfAnFace partition is required")
        image_size(config.output_size)
        self.images, self.config = images, config
        self.samples = tuple(s for s in images.samples if s.official_split == official_partition)
        if not self.samples:
            raise ValueError("requested official partition contains no records")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if get_worker_info() is not None:
            raise ValueError("the context-owned archive adapter requires num_workers=0")
        face = self.images.load(self.samples[index])
        try:
            prepared = preprocess_face(
                face,
                source_index_base=self.config.source_index_base,
                annotation_frame=self.config.annotation_frame,
                orientation_policy=self.config.orientation_policy,
                output_size=self.config.output_size,
                crop_box=self.config.crop_box,
            )
            try:
                return prepared_to_tensors(prepared)
            finally:
                prepared.image.close()
        finally:
            face.image.close()


def decode_coordinates(normalized, output_size, inverse):
    """Normalized edge predictions -> original annotation numbers, no clamping.

    Supports [K,2] or [B,K,2], with matching [3,3] or [B,3,3] inverse matrices.
    The working frame already uses edges, so no extra 0.5 is added here.
    """
    size = image_size(output_size)
    if normalized.ndim not in (2, 3) or normalized.shape[-1] != 2:
        raise ValueError("coordinates must have shape [K,2] or [B,K,2]")
    expected = (3, 3) if normalized.ndim == 2 else (normalized.shape[0], 3, 3)
    if tuple(inverse.shape) != expected or not torch.isfinite(inverse).all():
        raise ValueError("inverse matrices must be finite with matching batch dimensions")
    edge = normalized.to(torch.float64) * normalized.new_tensor(size, dtype=torch.float64)
    homogeneous = torch.cat((edge, torch.ones_like(edge[..., :1])), dim=-1)
    return torch.matmul(homogeneous, inverse.transpose(-1, -2))[..., :2]
