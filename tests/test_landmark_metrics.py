"""Hand-computed errors independent of PyTorch, infant data and model outputs."""

import math

import pytest

from infant_movement.evaluation.landmark_metrics import interpoint_normalizers, landmark_errors


def test_image_macro_is_not_pooled_point_mean():
    report = landmark_errors(
        [[(6, 8), (0, 0)], [(0, 0), (0, 0)]],
        [[(0, 0), (99, 99)], [(0, 0), (0, 0)]],
        [[True, False], [True, True]],
        normalizers=[10, 20],
    )
    assert report["mean_error"] == 5  # (10 from image 1 + 0 from image 2)/2.
    assert report["normalized_mean_error"] == 0.5
    assert report["point_coverage"] == 0.75 and report["image_coverage"] == 1
    assert report["eligible_points"] == 3


@pytest.mark.parametrize("denominator", [None, 0, -1, float("nan"), float("inf"), "unknown"])
def test_undefined_normalizer_has_no_fallback(denominator):
    report = landmark_errors([[(3, 4)]], [[(0, 0)]], [[True]], normalizers=[denominator])
    assert report["mean_error"] == 5 and report["normalized_mean_error"] is None
    assert report["undefined_normalizer_images"] == 1


def test_missing_points_have_coverage_without_zero_errors():
    report = landmark_errors([[(math.nan, math.nan)]], [[None]], [[False]])
    assert report["mean_error"] is report["normalized_mean_error"] is None
    assert report["point_coverage"] == report["image_coverage"] == 0


def test_explicit_index_subset_and_interpoint_denominators():
    points, masks = [[(0, 0), (3, 4), (10, 0)]], [[True, True, False]]
    assert interpoint_normalizers(points, masks, index_pair=(0, 1)) == [5]
    assert interpoint_normalizers(points, masks, index_pair=(0, 2)) == [None]
    assert interpoint_normalizers([[(1, 1), (1, 1)]], [[True, True]], index_pair=(0, 1)) == [None]
    report = landmark_errors([[(6, 8), (0, 0), (99, 99)]], points, masks, indices=(1,))
    assert report["mean_error"] == 5 and report["eligible_points"] == 1


@pytest.mark.parametrize("indices", [[], [True], [0, 0], [-1], [68]])
def test_invalid_indices_rejected(indices):
    with pytest.raises(ValueError):
        landmark_errors([[(0, 0)] * 68], [[(0, 0)] * 68], [[True] * 68], indices=indices)


def test_nonfinite_eligible_predictions_and_bad_shapes_fail():
    with pytest.raises(ValueError, match="finite"):
        landmark_errors([[(float("nan"), 0)]], [[(0, 0)]], [[True]])
    with pytest.raises(ValueError):
        landmark_errors([[(0, 0)]], [[(0, 0)]], [[1]])
    with pytest.raises(ValueError):
        landmark_errors([], [], [])
    with pytest.raises(ValueError):
        interpoint_normalizers([[(0, 0)]], [[True]], index_pair=(0, 0))
