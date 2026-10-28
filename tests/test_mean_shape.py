"""Synthetic hand arithmetic: masked constant fit, inversion and pairing."""

# ruff: noqa: E402
import pytest

torch = pytest.importorskip("torch")

from infant_movement.evaluation.mean_shape import (
    check_reproduction,
    evaluate_predictions,
    paired_differences,
    relative_reduction,
)
from infant_movement.models.mean_shape import fit_mean_shape


def test_unequal_support_arithmetic_and_undefined_mean():
    target = torch.tensor(
        [
            [[0.0, 0.0], [0.25, 0.5], [float("nan")] * 2],
            [[0.5, 1.0], [999.0, 999.0], [float("nan")] * 2],
        ],
        dtype=torch.float64,
    )
    valid = torch.tensor([[True, True, False], [True, False, False]])
    model = fit_mean_shape(target, valid)
    assert model.support.tolist() == [2, 1, 0]
    assert model.coordinates[:2].tolist() == [[0.25, 0.5], [0.25, 0.5]]
    assert torch.isnan(model.coordinates[2]).all()
    predicted = model.predict(2)
    assert torch.equal(predicted[0, :2], predicted[1, :2])
    assert torch.isnan(predicted[:, 2]).all()
    # Only 3 points / 6 scalars contribute: (.25²+.5²)*2 / 6 = 5/48.
    identity = torch.eye(3, dtype=torch.float64).repeat(2, 1, 1)
    metrics, rows = evaluate_predictions(predicted, target, valid, identity, target * 32)
    assert metrics["normalized_coordinate_mse"] == pytest.approx(5 / 48)
    assert metrics["mse_coordinate_scalars"] == 6
    # Distances: sqrt(320),0 in image 1; sqrt(320) in image 2.
    assert metrics["input_pixel_error"]["mean_error"] == pytest.approx((320**0.5) * 0.75)
    assert metrics["input_pixel_error"]["eligible_points"] == 3
    assert metrics["input_pixel_error"]["selected_point_slots"] == 6
    assert [r["eligible_points"] for r in rows] == [2, 1]


def test_source_inverse_and_independent_pixel_units():
    target = torch.tensor([[[0.25, 0.25]], [[0.5, 0.5]]], dtype=torch.float64)
    valid = torch.ones(2, 1, dtype=torch.bool)
    predicted = target + torch.tensor([3 / 32, 4 / 32], dtype=torch.float64)
    inverse = torch.tensor(
        [
            [[2.0, 0.0, -7.0], [0.0, 2.0, 9.0], [0.0, 0.0, 1.0]],
            [[0.5, 0.0, 3.0], [0.0, 0.5, -4.0], [0.0, 0.0, 1.0]],
        ],
        dtype=torch.float64,
    )
    source = torch.tensor([[[9.0, 25.0]], [[11.0, 4.0]]], dtype=torch.float64)
    metrics, rows = evaluate_predictions(predicted, target, valid, inverse, source)
    assert metrics["normalized_coordinate_mse"] == pytest.approx(12.5 / 1024)
    assert metrics["input_pixel_error"]["mean_error"] == 5
    assert [r["source_pixel_error"] for r in rows] == [10, 2.5]
    assert metrics["source_pixel_error"]["mean_error"] == 6.25
    assert metrics["source_pixel_error"]["normalized_mean_error"] is None


def test_all_missing_stays_undefined_no_fake_zero():
    targets = torch.full((2, 3, 2), float("nan"))
    valid = torch.zeros(2, 3, dtype=torch.bool)
    model = fit_mean_shape(targets, valid)
    assert model.support.tolist() == [0, 0, 0] and torch.isnan(model.predict(2)).all()
    metrics, rows = evaluate_predictions(
        model.predict(2),
        targets,
        valid,
        torch.eye(3, dtype=torch.float64).repeat(2, 1, 1),
        targets.double(),
    )
    assert metrics["normalized_coordinate_mse"] is None
    assert metrics["mse_coordinate_scalars"] == 0
    assert metrics["input_pixel_error"]["eligible_images"] == 0
    paired, private = paired_differences(rows, rows)
    assert paired["input_pixel_error"]["paired_images"] == 0
    assert paired["input_pixel_error"]["undefined_relative_images"] == 2
    assert private[0]["input_pixel_error"]["cnn_minus_control"] is None


def test_paired_differences_zero_denominators_and_ties():
    def row(value):
        return dict(
            eligible_points=1,
            coordinate_scalars=2,
            normalized_coordinate_mse=value,
            input_pixel_error=value,
            source_pixel_error=value,
        )

    summary, private = paired_differences([row(1), row(0), row(4)], [row(2), row(0), row(2)])
    metric = summary["input_pixel_error"]
    assert metric["mean_cnn_minus_control"] == pytest.approx(1 / 3)
    assert metric["median_cnn_minus_control"] == 0
    assert metric["min_cnn_minus_control"] == -1 and metric["max_cnn_minus_control"] == 2
    assert (
        metric["cnn_lower_images"] == metric["control_lower_images"] == metric["tied_images"] == 1
    )
    assert metric["undefined_relative_images"] == 1
    assert private[0]["input_pixel_error"]["relative_reduction"] == 0.5
    assert relative_reduction(0, 0) is relative_reduction(0, 1) is None
    assert relative_reduction(2, 4) == -1
    with pytest.raises(ValueError, match="denominators"):
        paired_differences([row(1)], [{**row(2), "eligible_points": 2}])


@pytest.mark.parametrize("kind", ["mask_dtype", "eligible_nan", "shape", "empty"])
def test_invalid_fit_input_rejected(kind):
    target = torch.zeros(2, 3, 2)
    valid = torch.ones(2, 3, dtype=torch.bool)
    if kind == "mask_dtype":
        valid = valid.float()
    elif kind == "eligible_nan":
        target[0, 0, 0] = float("nan")
    elif kind == "shape":
        target = target[:, :, :1]
    else:
        target, valid = target[:0], valid[:0]
    with pytest.raises(ValueError):
        fit_mean_shape(target, valid)


def test_nonfinite_cnn_predictions_fail_not_reduce_coverage():
    target = torch.zeros(1, 1, 2)
    with pytest.raises(ValueError, match="predictions"):
        evaluate_predictions(
            torch.full_like(target, float("nan")),
            target,
            torch.ones(1, 1, dtype=torch.bool),
            torch.eye(3).unsqueeze(0),
            target,
        )


def test_reproduction_predeclared_tolerance_and_counts():
    target = torch.zeros(1, 1, 2, dtype=torch.float64)
    original, _ = evaluate_predictions(
        target + 1,
        target,
        torch.ones(1, 1, dtype=torch.bool),
        torch.eye(3, dtype=torch.float64).unsqueeze(0),
        target,
    )
    near = {**original, "normalized_coordinate_mse": 1 + 1e-8}
    assert check_reproduction(near, original) == []
    far = {**original, "normalized_coordinate_mse": 1.001}
    assert check_reproduction(far, original) == ["normalized_coordinate_mse"]
    bad = {**original, "input_pixel_error": {**original["input_pixel_error"], "eligible_points": 0}}
    assert check_reproduction(bad, original) == ["input_pixel_error.eligible_points"]
