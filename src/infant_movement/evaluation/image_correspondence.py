"""One global image-only cyclic reassignment and matched descriptive arithmetic."""

import hashlib

import torch
from torch.utils.data import Dataset

from infant_movement.evaluation.mean_shape import paired_differences

FIELDS = ("normalized_coordinate_mse", "input_pixel_error", "source_pixel_error")


def cyclic_mapping(count):
    if type(count) is not int or count != 32:
        raise ValueError("this control requires the exact 32-position usable cohort")
    return tuple((i + 1) % count for i in range(count))


def validate_mapping(mapping, count):
    if (
        any(type(i) is not int for i in mapping)
        or tuple(mapping) != cyclic_mapping(count)
        or sorted(mapping) != list(range(count))
        or any(i == j for i, j in enumerate(mapping))
    ):
        raise ValueError("only the predeclared global cyclic bijection is authorized")


class ReassignedImages(Dataset):
    """Replace only image; every other field is cloned from its destination."""

    def __init__(self, original, mapping):
        validate_mapping(mapping, len(original))
        self.original, self.mapping = original, tuple(mapping)

    def __len__(self):
        return len(self.original)

    def __getitem__(self, index):
        destination = self.original[index]
        destination["image"] = self.original[self.mapping[index]]["image"]
        return destination


def tensor_digest(value):
    """Exact CPU tensor bytes, shape and dtype; supports masked NaNs without JSON."""
    if value.device.type != "cpu":
        raise ValueError("only CPU prepared tensors are authorized")
    data = value.detach().contiguous()
    digest = hashlib.sha256(f"{data.dtype}|{tuple(data.shape)}|".encode())
    digest.update(bytes(data.reshape(-1).view(torch.uint8).tolist()))
    return digest.hexdigest()


def paired_comparison(paired_rows, reassigned_rows, *, tie_atol=1e-12):
    for a, b in zip(paired_rows, reassigned_rows, strict=True):
        if a["coordinate_scalars"] != b["coordinate_scalars"]:
            raise ValueError("paired coordinate-scalar denominators differ")
    # Reuse the independently tested arithmetic with reassigned as the first
    # operand. Rename every field so the saved scientific sign is unambiguous.
    aggregate, rows = paired_differences(reassigned_rows, paired_rows, tie_atol=tie_atol)
    renames = {
        "mean_cnn_minus_control": "mean_reassigned_minus_paired",
        "median_cnn_minus_control": "median_reassigned_minus_paired",
        "min_cnn_minus_control": "min_reassigned_minus_paired",
        "max_cnn_minus_control": "max_reassigned_minus_paired",
        "cnn_lower_images": "reassigned_lower_images",
        "control_lower_images": "paired_lower_images",
    }
    summary = {
        field: {renames.get(k, k): v for k, v in values.items()}
        for field, values in aggregate.items()
    }
    private = []
    for row in rows:
        result = {k: row[k] for k in ("eligible_points", "coordinate_scalars")}
        for field in FIELDS:
            value = row[field]
            relative = value["relative_reduction"]
            result[field] = {
                "reassigned_minus_paired": value["cnn_minus_control"],
                "relative_increase": -relative if relative is not None else None,
            }
        private.append(result)
    return summary, private
