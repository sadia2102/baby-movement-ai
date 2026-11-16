"""Independent synthetic heatmap arithmetic; no infant data or trained assets."""

# ruff: noqa: E402
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from infant_movement.datasets.coordinate_conventions import source_centres_to_edges
from infant_movement.datasets.landmark_geometry import crop_geometry, letterbox_geometry
from infant_movement.datasets.landmark_heatmaps import (
    DEFAULT_HEATMAP_CONFIG,
    HeatmapConfig,
    decode_argmax,
    generate_heatmaps,
    masked_heatmap_mse,
)


def inverse(batch=1):
    return torch.eye(3, dtype=torch.float64).repeat(batch, 1, 1)


def decode(
    maps, mask, *, config=DEFAULT_HEATMAP_CONFIG, recorded_inverse=None, base=0, frame="stored"
):
    return decode_argmax(
        maps,
        mask,
        config=config,
        inverse=inverse(maps.shape[0]) if recorded_inverse is None else recorded_inverse,
        source_index_base=base,
        annotation_frame=frame,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gaussian_cell_centres_values_layout_and_no_normalization(dtype):
    points = torch.tensor([[[10.0, 14.0]] * 68], dtype=dtype, requires_grad=True)
    original = points.detach().clone()
    target = generate_heatmaps(points, torch.ones((1, 68), dtype=torch.bool))
    assert target.heatmaps.shape == (1, 68, 8, 8)
    assert target.heatmaps.dtype == dtype and not target.heatmaps.requires_grad
    assert not target.input_edges.requires_grad
    assert torch.equal(points.detach(), original)
    assert target.heatmap_edges[0, 0].tolist() == [2.5, 3.5]
    assert float(target.heatmaps[0, 0, 3, 2]) == 1
    assert float(target.heatmaps[0, 0, 3, 3]) == pytest.approx(math.exp(-0.5), abs=1e-7)
    assert float(target.heatmaps[0, 0, 4, 3]) == pytest.approx(math.exp(-1), abs=1e-7)
    assert float(target.heatmaps[0, 0].sum()) > 1
    assert target.coverage["eligible_cells"] == 68 * 64
    decoded = decode(target.heatmaps, target.eligible)
    assert decoded.heatmap_edges[0, 0].tolist() == [2.5, 3.5]
    assert decoded.input_edges[0, 0].tolist() == [10, 14]
    assert decoded.source_coordinates[0, 0].tolist() == [10, 14]
    assert decoded.supported.all() and not decoded.ambiguous.any()


def test_fractional_gaussian_peak_score_and_expected_argmax_quantization():
    points = torch.tensor([[[11.2, 14.8]]], dtype=torch.float64)
    target = generate_heatmaps(points, torch.tensor([[True]]))
    assert target.heatmap_edges[0, 0].tolist() == pytest.approx([2.8, 3.7])
    result = decode(target.heatmaps, target.eligible)
    assert result.input_edges[0, 0].tolist() == [10, 14]
    assert float(result.peak_scores[0, 0]) == pytest.approx(math.exp(-0.065), abs=1e-12)
    delta = result.input_edges[0, 0] - points[0, 0]
    assert delta.tolist() == pytest.approx([-1.2, -0.8], abs=1e-12)
    assert float(torch.linalg.vector_norm(delta)) == pytest.approx(math.sqrt(2.08), abs=1e-12)


def test_non_square_noninteger_unequal_axis_scales_and_slot_order():
    config = HeatmapConfig(input_size=(30, 18), heatmap_size=(7, 5))
    points = torch.tensor([[[15.0, 9.0], [75 / 7, 1.8]]], dtype=torch.float64)
    targets = generate_heatmaps(points, torch.ones((1, 2), dtype=torch.bool), config=config)
    assert targets.heatmaps.shape == (1, 2, 5, 7)
    assert targets.heatmap_edges[0, 0].tolist() == pytest.approx([3.5, 2.5], abs=1e-12)
    assert targets.heatmap_edges[0, 1].tolist() == pytest.approx([2.5, 0.5], abs=1e-12)
    result = decode(targets.heatmaps, targets.eligible, config=config)
    assert result.input_edges[0, 0].tolist() == [15, 9]
    assert result.input_edges[0, 1].tolist() == pytest.approx([75 / 7, 1.8], abs=1e-12)


@pytest.mark.parametrize("base", [0, 1])
def test_existing_crop_padding_and_declared_source_inverse_no_extra_half_pixel(base):
    _, crop, cropped = crop_geometry((4, 2, 16, 8), (20, 10))
    fit, scale, padding = letterbox_geometry(cropped, (30, 18))
    assert scale == 2.5 and padding == (0, 1.5)
    forward = source_centres_to_edges(source_index_base=base).then(crop).then(fit)
    recorded = torch.tensor([forward.inverse().matrix], dtype=torch.float64)
    config = HeatmapConfig(input_size=(30, 18), heatmap_size=(7, 5))
    target = generate_heatmaps(
        torch.tensor([[[15.0, 9.0]]], dtype=torch.float64), torch.tensor([[True]]), config=config
    )
    result = decode(
        target.heatmaps, target.eligible, config=config, recorded_inverse=recorded, base=base
    )
    # Input x/2.5 + (4 - 0.5 + base); y=(9-1.5)/2.5 + (2-0.5+base).
    assert result.source_coordinates[0, 0].tolist() == pytest.approx([9.5 + base, 4.5 + base])
    assert result.source_assumptions["source_index_base"] == base


def test_inverse_arbitrary_affine_shear_and_batch_specific_offsets():
    maps = torch.zeros((2, 1, 8, 8), dtype=torch.float64)
    maps[:, :, 3, 2] = 1
    matrices = torch.tensor(
        [
            [[2.0, 0.5, 10.0], [-1.0, 3.0, -5.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, -7.0], [0.0, 2.0, 4.0], [0.0, 0.0, 1.0]],
        ]
    )
    result = decode(maps, torch.ones((2, 1), dtype=torch.bool), recorded_inverse=matrices.double())
    assert result.source_coordinates[:, 0].tolist() == [[37, 27], [3, 32]]


def test_missing_masks_oob_numeric_values_and_originals_are_retained():
    points = torch.tensor(
        [[[float("nan"), float("nan")], [-1.0, 5.0], [32.0, 4.0], [2.0, 2.0], [3.0, 3.0]]],
        dtype=torch.float64,
    )
    mask = torch.tensor([[False, True, True, True, False]])
    result = generate_heatmaps(points, mask)
    assert torch.isnan(result.input_edges[0, 0]).all()
    assert result.input_edges[0, 1].tolist() == [-1, 5]
    assert result.input_edges[0, 2].tolist() == [32, 4]
    assert result.heatmap_edges[0, 1].tolist() == [-0.25, 1.25]
    assert result.eligible.tolist() == [[False, False, False, True, False]]
    assert torch.equal(result.requested_eligible, mask)
    assert not result.heatmaps[~result.eligible].any()
    assert result.coverage["missing_slots"] == 1
    assert result.coverage["out_of_footprint_slots"] == 2
    assert result.coverage["point_coverage"] == 0.2
    result.input_edges[0, 1, 0] = 100
    assert points[0, 1, 0] == -1 and mask[0, 1]


def test_border_truncation_is_unrenormalized_and_footprint_is_half_open():
    points = torch.tensor([[[0.0, 0.0], [32.0, 0.0], [31.999, 31.999]]], dtype=torch.float64)
    result = generate_heatmaps(points, torch.ones((1, 3), dtype=torch.bool))
    assert result.eligible.tolist() == [[True, False, True]]
    assert float(result.heatmaps[0, 0, 0, 0]) == pytest.approx(math.exp(-0.25), abs=1e-12)
    assert float(result.heatmaps[0, 0].sum()) < 2
    assert not result.heatmaps[0, 1].any()
    decoded = decode(result.heatmaps, result.eligible)
    assert decoded.input_edges[0, 0].tolist() == [2, 2]
    assert float(torch.linalg.vector_norm(decoded.input_edges[0, 0])) == pytest.approx(
        2 * math.sqrt(2)
    )
    assert not decoded.supported[0, 1] and torch.isnan(decoded.input_edges[0, 1]).all()


def test_exact_ties_use_first_row_major_maximum_and_expose_ambiguity():
    maps = torch.zeros((1, 2, 8, 8), dtype=torch.float64)
    maps[0, 0, 1, 7] = maps[0, 0, 2, 0] = 9
    # Raw unconstrained negative scores have no arbitrary positivity threshold.
    maps[0, 1].fill_(-5)
    maps[0, 1, 6, 4] = -2
    result = decode(maps, torch.tensor([[True, True]]))
    assert result.input_edges[0, 0].tolist() == [30, 6]
    assert result.peak_multiplicity.tolist() == [[2, 1]]
    assert result.ambiguous.tolist() == [[True, False]]
    assert result.peak_scores.tolist() == [[9, -2]]
    assert result.supported.all()


@pytest.mark.parametrize("value", [0.0, 2.0, -3.0])
def test_flat_maps_unsupported_not_valid_top_left(value):
    maps = torch.full((1, 2, 8, 8), value, dtype=torch.float64)
    maps[0, 1].fill_(float("nan"))  # Entirely ineligible map is never inspected.
    result = decode(maps, torch.tensor([[True, False]]))
    assert result.supported.tolist() == [[False, False]]
    assert torch.isnan(result.heatmap_edges).all() and torch.isnan(result.source_coordinates).all()
    assert result.flat.tolist() == [[True, False]]
    assert result.peak_multiplicity.tolist() == [[64, 0]]
    assert result.coverage["unsupported_flat_slots"] == 1
    assert result.coverage["eligible_point_slots"] == 0


def test_single_cell_map_and_all_masked_maps_are_explicitly_unsupported():
    config = HeatmapConfig(heatmap_size=(1, 1))
    maps = torch.ones((1, 1, 1, 1))
    result = decode(maps, torch.tensor([[True]]), config=config)
    assert result.flat.all() and not result.supported.any()
    result = decode(torch.full((1, 1, 8, 8), float("nan")), torch.tensor([[False]]))
    assert not result.supported.any() and result.coverage["requested_eligible_slots"] == 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_eligible_predictions_fail_without_coverage_attrition(bad):
    maps = torch.zeros((1, 1, 8, 8))
    maps[0, 0, 7, 7] = bad
    with pytest.raises(ValueError, match="nonfinite eligible"):
        decode(maps, torch.tensor([[True]]))


@pytest.mark.parametrize(
    "point,mask",
    [
        ([float("nan"), float("nan")], True),
        ([1.0, float("nan")], False),
        ([float("inf"), 2.0], False),
    ],
)
def test_malformed_or_nonfinite_target_coordinates_rejected(point, mask):
    with pytest.raises(ValueError):
        generate_heatmaps(torch.tensor([[point]]), torch.tensor([[mask]]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_size": (0, 32)},
        {"input_size": (True, 32)},
        {"heatmap_size": (8.0, 8)},
        {"heatmap_size": (8,)},
        {"sigma": 0},
        {"sigma": -1},
        {"sigma": True},
        {"sigma": float("inf")},
        {"sigma": float("nan")},
    ],
)
def test_invalid_configurations_rejected(kwargs):
    with pytest.raises(ValueError):
        HeatmapConfig(**kwargs)


@pytest.mark.parametrize(
    "violation", ["mask_type", "mask_shape", "point_shape", "point_dtype", "empty", "config_type"]
)
def test_target_shape_type_and_mask_contracts(violation):
    points = torch.ones((1, 2, 2))
    mask = torch.ones((1, 2), dtype=torch.bool)
    kwargs = {}
    if violation == "mask_type":
        mask = mask.float()
    elif violation == "mask_shape":
        mask = mask[:, :1]
    elif violation == "point_shape":
        points = points[..., :1]
    elif violation == "point_dtype":
        points = points.long()
    elif violation == "empty":
        points = points[:, :0]
    else:
        kwargs["config"] = {}
    with pytest.raises(ValueError):
        generate_heatmaps(points, mask, **kwargs)


@pytest.mark.parametrize(
    "violation",
    [
        "shape",
        "matrix_dtype",
        "matrix_shape",
        "singular",
        "projective",
        "nonfinite",
        "base",
        "frame",
    ],
)
def test_decoder_requires_explicit_complete_inverse_and_conventions(violation):
    maps = torch.zeros((1, 1, 8, 8))
    matrix, base, frame = inverse(), 0, "stored"
    if violation == "shape":
        maps = maps[..., :7]
    elif violation == "matrix_dtype":
        matrix = matrix.float()
    elif violation == "matrix_shape":
        matrix = matrix[0]
    elif violation == "singular":
        matrix[:, 0] = 0
    elif violation == "projective":
        matrix[:, 2, 0] = 1
    elif violation == "nonfinite":
        matrix[:, 0, 0] = float("inf")
    elif violation == "base":
        base = True
    else:
        frame = "inferred"
    with pytest.raises(ValueError):
        decode(maps, torch.tensor([[True]]), recorded_inverse=matrix, base=base, frame=frame)


def test_mask_before_loss_arithmetic_unequal_support_hand_values_and_gradients():
    pred = torch.tensor(
        [[[[1.0, 3.0]], [[float("nan"), float("nan")]]], [[[2.0, 4.0]], [[0.0, 2.0]]]],
        requires_grad=True,
    )
    target = torch.zeros_like(pred)
    target[0, 1] = float("nan")
    mask = torch.tensor([[True, False], [True, True]])
    result = masked_heatmap_mse(pred, target, mask)
    assert result.denominator == 6
    assert float(result.loss.detach()) == pytest.approx(34 / 6, abs=1e-12)
    assert result.loss.dtype == torch.float64
    assert result.coverage["eligible_images"] == 2
    assert result.coverage["point_coverage"] == 0.75
    result.loss.backward()
    expected = torch.tensor([[[[1 / 3, 1.0]], [[0.0, 0.0]]], [[[2 / 3, 4 / 3]], [[0.0, 2 / 3]]]])
    torch.testing.assert_close(pred.grad, expected, atol=1e-7, rtol=1e-6)
    assert torch.isfinite(pred.grad).all()


def test_no_supervision_is_no_loss_and_does_not_create_gradients():
    pred = torch.full((2, 3, 4, 5), float("nan"), requires_grad=True)
    result = masked_heatmap_mse(pred, pred.detach(), torch.zeros((2, 3), dtype=torch.bool))
    assert result.loss is None and result.denominator == 0
    assert result.coverage["eligible_images"] == 0
    assert pred.grad is None


@pytest.mark.parametrize(
    "violation", ["prediction_nan", "target_inf", "overflow", "shape", "mask", "integer", "empty"]
)
def test_loss_rejects_invalid_eligible_values_shapes_and_overflow(violation):
    pred = torch.zeros((1, 1, 1, 2), dtype=torch.float64)
    target = torch.zeros_like(pred)
    mask = torch.tensor([[True]])
    if violation == "prediction_nan":
        pred[0, 0, 0, 0] = float("nan")
    elif violation == "target_inf":
        target[0, 0, 0, 0] = float("inf")
    elif violation == "overflow":
        pred[0, 0, 0, 0] = 1e308
    elif violation == "shape":
        target = target[..., :1]
    elif violation == "mask":
        mask = mask.long()
    elif violation == "integer":
        pred = pred.long()
    else:
        pred = pred[:, :0]
    with pytest.raises(ValueError):
        masked_heatmap_mse(pred, target, mask)


def test_extreme_sigma_underflow_fails_instead_of_valid_zero_target():
    with pytest.raises(ValueError, match="support"):
        generate_heatmaps(
            torch.tensor([[[12.0, 12.0]]]),
            torch.tensor([[True]]),
            config=HeatmapConfig(sigma=1e-200),
        )


def test_core_import_stays_dependency_free():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import infant_movement; "
            "import infant_movement.datasets.coordinate_conventions; "
            "assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0


def test_gaussian_midpoint_has_four_exact_ties_and_declared_quantization():
    target = generate_heatmaps(
        torch.tensor([[[12.0, 12.0]]], dtype=torch.float64), torch.tensor([[True]])
    )
    result = decode(target.heatmaps, target.eligible)
    assert result.heatmap_edges[0, 0].tolist() == [2.5, 2.5]
    assert result.peak_multiplicity.tolist() == [[4]]
    assert result.ambiguous.all() and result.supported.all()
    assert float(result.peak_scores[0, 0]) == pytest.approx(math.exp(-0.25), abs=1e-12)


def test_decoder_is_detached_and_does_not_mutate_prediction_mask_or_inverse():
    maps = torch.arange(64).float().reshape(1, 1, 8, 8).requires_grad_()
    mask, recorded = torch.tensor([[True]]), inverse()
    before, before_matrix = maps.detach().clone(), recorded.clone()
    result = decode(maps, mask, recorded_inverse=recorded)
    assert not result.input_edges.requires_grad and not result.peak_scores.requires_grad
    assert torch.equal(maps.detach(), before) and torch.equal(recorded, before_matrix)
    result.requested_eligible[0, 0] = False
    assert mask[0, 0]


def test_offline_verifier_with_network_and_research_artifact_reads_denied():
    root = Path(__file__).resolve().parents[1]
    code = """
import pathlib, runpy, sys
root=pathlib.Path.cwd()
def guard(event,args):
    if event in ('socket.connect','socket.getaddrinfo'):
        raise AssertionError('network forbidden')
    if event=='open' and isinstance(args[0],(str,bytes)):
        p=pathlib.Path(args[0]).resolve()
        sealed=('data','runs','manifests','artifacts','reports/private')
        if any(p.is_relative_to(root/f) for f in sealed):
            raise AssertionError('research artifacts forbidden')
sys.addaudithook(guard)
sys.argv=['scripts/verify_heatmap_adapter.py']
runpy.run_path(sys.argv[0],run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=root, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads(result.stdout)
    assert record["passed"] and all(record["checks"].values())
    assert record["model_operations"] == record["dataset_samples_used"] == 0


@pytest.mark.parametrize(
    "arguments",
    [
        ["--dataset", "forbidden"],
        ["--archive", "forbidden"],
        ["--checkpoint", "forbidden"],
        ["--run-dir", "/private/tmp/heatmap-public-output"],
    ],
)
def test_verifier_rejects_asset_arguments_and_public_output(arguments):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/verify_heatmap_adapter.py", *arguments],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
