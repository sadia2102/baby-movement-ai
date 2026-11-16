"""Independent hand-calculated synthetic geometry/NME; no images or models."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from infant_movement.datasets.coordinate_conventions import (
    decode_normalized_edges,
    encode_normalized_edges,
    source_centres_to_edges,
)
from infant_movement.datasets.landmark_geometry import (
    Affine2D,
    PreprocessingError,
    crop_geometry,
    exif_geometry,
    letterbox_geometry,
    transform_landmarks,
)
from infant_movement.evaluation.infanface_conventions import (
    ASSUMED_SLOT_ORDER,
    infanface_reference_nme,
    legacy_hrnet_pixel_reference,
)


def nme(prediction, target, mask, **changes):
    arguments = {"slot_order": ASSUMED_SLOT_ORDER, "units": "source-pixels", **changes}
    return infanface_reference_nme(prediction, target, mask, **arguments)


def points68():
    target = [(0.0, 0.0)] * 68
    target[45] = (3.0, 4.0)
    target[42] = (50.0, 0.0)  # Incorrect inner/other-corner selection must fail gold arithmetic.
    return target


def shifted(points, x=3, y=4):
    return [(px + x, py + y) for px, py in points]


def test_nme_uses_target_slots_36_45_and_explicit_fraction_percent():
    target = points68()
    report = nme([shifted(target)], [target], [[True] * 68])
    assert report["normalizer_index_pair"] == [36, 45]
    assert report["mean_error"] == 5
    assert report["nme_fraction"] == 1 and report["nme_percent"] == 100
    assert report["fully_eligible_reference_images"] == 1
    assert report["masked_extension_normalized_images"] == 0
    assert report["eligible_points"] == report["selected_point_slots"] == 68
    assert (
        report["source_coordinate_evidence"]
        == report["local_anatomical_order_evidence"]
        == "UNVERIFIED"
    )
    larger = nme([shifted(target, 6, 8)], [target], [[True] * 68])
    assert larger["nme_fraction"] == 2 and larger["nme_percent"] == 200  # Never clamp to 100%.


def test_masked_extension_is_image_macro_with_explicit_denominators():
    a = points68()
    b = [(2 * x, 2 * y) for x, y in a]
    first = [False] * 68
    for i in (0, 36, 45):
        first[i] = True
    report = nme([shifted(a), shifted(b)], [a, b], [first, [True] * 68])
    # Image A: 5/5=1; image B: 5/10=.5. Give each image equal weight.
    assert report["nme_fraction"] == 0.75
    assert report["eligible_points"] == 71 and report["selected_point_slots"] == 136
    assert report["fully_eligible_reference_images"] == 1
    assert report["masked_extension_normalized_images"] == 1
    assert report["normalized_eligible_images"] == 2


@pytest.mark.parametrize("failure", ["missing_eye", "zero_distance", "nonfinite_masked_eye"])
def test_undefined_eye_denominator_keeps_pixel_coverage(failure):
    target = points68()
    pred = shifted(target)
    mask = [True] * 68
    if failure == "zero_distance":
        target[45] = target[36]
    else:
        mask[36] = False
    if failure == "nonfinite_masked_eye":
        target[36] = (math.nan, math.nan)
        pred[36] = None
    report = nme([pred], [target], [mask])
    assert report["nme_fraction"] is report["nme_percent"] is None
    assert report["normalizer_defined_images"] == 0
    assert report["undefined_normalizer_images"] == report["undefined_normalizer_images_total"] == 1
    assert report["eligible_points"] == (68 if failure == "zero_distance" else 67)
    assert report["mean_error"] is not None


def test_all_missing_points_stay_missing_and_counts_remain_explicit():
    report = nme([[None] * 68], [[None] * 68], [[False] * 68])
    assert report["mean_error"] is report["nme_fraction"] is None
    assert report["eligible_points"] == report["eligible_images"] == 0
    assert report["no_eligible_point_images"] == report["undefined_normalizer_images_total"] == 1
    assert report["undefined_normalizer_images"] == 0  # Existing field counts eligible images only.


def test_common_translation_and_uniform_scale_preserve_nme():
    target = points68()
    prediction = shifted(target)

    def altered(points):
        return [(x * 2 + 7, y * 2 - 11) for x, y in points]

    report = nme([altered(prediction)], [altered(target)], [[True] * 68], units="input-pixels")
    assert report["mean_error"] == 10 and report["nme_fraction"] == 1
    assert nme([shifted(target)], [target], [[True] * 68])["nme_fraction"] == 1


def test_finite_inputs_cannot_silently_yield_infinite_metric():
    with pytest.raises(ValueError, match="overflowed"):
        nme([[(1e308, 1e308)] * 68], [points68()], [[True] * 68])


def test_percentage_overflow_is_not_returned_as_a_valid_number():
    target = [(0.0, 0.0)] * 68
    target[45] = (1e-308, 0.0)
    with pytest.raises(ValueError, match="percentage overflowed"):
        nme([[(1.0, 0.0)] * 68], [target], [[True] * 68])


@pytest.mark.parametrize(
    "change",
    [{"slot_order": "verified"}, {"slot_order": None}, {"units": "normalized-xy"}, {"units": ""}],
)
def test_unsupported_or_implicit_convention_rejected(change):
    target = points68()
    with pytest.raises(ValueError):
        nme([target], [target], [[True] * 68], **change)


def test_explicit_declarations_and_finite_eligible_targets_required():
    target = points68()
    with pytest.raises(TypeError):
        infanface_reference_nme([target], [target], [[True] * 68])
    with pytest.raises(ValueError):
        nme([target[:-1]], [target[:-1]], [[True] * 67])
    with pytest.raises(ValueError):
        nme([target], [target], [[1] * 68])
    target[36] = (math.nan, 0)
    with pytest.raises(ValueError, match="finite"):
        nme([points68()], [target], [[True] * 68])


def test_origin_hypotheses_are_explicit_and_equivalent_only_after_conversion():
    zero = source_centres_to_edges(source_index_base=0)
    one = source_centres_to_edges(source_index_base=1)
    assert transform_landmarks(((0, 0),), zero) == ((0.5, 0.5),)
    assert transform_landmarks(((1, 1),), one) == ((0.5, 0.5),)
    assert transform_landmarks(((0, 0),), one) == ((-0.5, -0.5),)  # No silent repair.
    with pytest.raises(TypeError):
        source_centres_to_edges()


@pytest.mark.parametrize("base", [True, None, -1, 2, 0.0])
def test_invalid_origin_rejected(base):
    with pytest.raises(PreprocessingError):
        source_centres_to_edges(source_index_base=base)


def test_non_square_crop_resize_padding_known_values_and_roundtrip():
    _, crop, size = crop_geometry((2, 1, 8, 5), (10, 6))
    resize, scale, padding = letterbox_geometry(size, (12, 12))
    assert scale == 2 and padding == (0, 2)
    forward = source_centres_to_edges(source_index_base=0).then(crop).then(resize)
    points = ((3, 2), (-2, 9), None)
    encoded = encode_normalized_edges(points, forward=forward, output_size=(12, 12))
    assert encoded[0] == pytest.approx((3 / 12, 5 / 12), abs=1e-12)
    assert encoded[1] == pytest.approx((-7 / 12, 19 / 12), abs=1e-12)
    recovered = decode_normalized_edges(encoded, inverse=forward.inverse(), output_size=(12, 12))
    assert recovered[0] == pytest.approx(points[0], abs=1e-10)
    assert recovered[1] == pytest.approx(points[1], abs=1e-10) and recovered[2] is None


def test_nonsquare_output_wh_is_not_a_hidden_isotropic_xy_normalization():
    encoded = encode_normalized_edges(
        ((3.5, 1.5),), forward=source_centres_to_edges(source_index_base=0), output_size=(8, 4)
    )
    assert encoded == ((0.5, 0.5),)
    result = decode_normalized_edges(
        encoded, inverse=source_centres_to_edges(source_index_base=0).inverse(), output_size=(8, 4)
    )
    assert result == ((3.5, 1.5),)


def test_explicit_exif_geometry_preserves_slot_and_source_inverse():
    orientation, size = exif_geometry((6, 4), 6)
    forward = source_centres_to_edges(source_index_base=0).then(orientation)
    source = ((1, 0), (3, 2), None)
    normalized = encode_normalized_edges(source, forward=forward, output_size=size)
    # Stored centre (1.5,.5) -> orientation-6 edge (3.5,1.5).
    assert normalized[0] == pytest.approx((3.5 / 4, 1.5 / 6), abs=1e-12)
    recovered = decode_normalized_edges(normalized, inverse=forward.inverse(), output_size=size)
    assert recovered[:2] == source[:2] and recovered[2] is None
    with pytest.raises(PreprocessingError):
        exif_geometry((6, 4), 0)


@pytest.mark.parametrize("point", [[(math.nan, 0)], [(True, 1)], [(1,)], []])
def test_invalid_coordinate_pairs_rejected(point):
    with pytest.raises(PreprocessingError):
        encode_normalized_edges(point, forward=Affine2D(), output_size=(8, 4))


def test_legacy_square_reference_matches_hand_arithmetic_and_loses_fractions():
    common = {"center": (50, 50), "scale": 1, "resolution": 100}
    assert legacy_hrnet_pixel_reference((11, 21), **common, inverse=False) == (31, 36)
    assert legacy_hrnet_pixel_reference((31, 36), **common, inverse=True) == (11, 21)
    quantized = legacy_hrnet_pixel_reference((10.25, 20.75), **common, inverse=False)
    assert quantized == (30, 35)
    back = legacy_hrnet_pixel_reference(quantized, **common, inverse=True)
    assert back == (9, 19) and back != (10.25, 20.75)
    # int(-50.5) is -50, not floor(-50.5)=-51.
    assert legacy_hrnet_pixel_reference(
        (0, 0), center=(200, 200), scale=1, resolution=100, inverse=False
    ) == (-49, -49)


@pytest.mark.parametrize(
    "change",
    [
        {"scale": 0},
        {"scale": math.inf},
        {"center": None},
        {"resolution": True},
        {"resolution": (100, 50)},
        {"inverse": 1},
    ],
)
def test_legacy_reference_rejects_unsupported_inputs(change):
    kwargs = {"center": (50, 50), "scale": 1, "resolution": 100, "inverse": False, **change}
    with pytest.raises(PreprocessingError):
        legacy_hrnet_pixel_reference((1, 2), **kwargs)


def test_offline_verifier_import_and_cli_have_no_optional_or_network_access(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/verify_infanface_conventions.py"
    # Deny network connections and attempts to open data/model artifact paths.
    code = """
import runpy,sys
def audit(event,args):
    if event in ('socket.connect','socket.getaddrinfo'): raise AssertionError('network forbidden')
    if event=='open' and isinstance(args[0],str):
        path=args[0].replace('\\\\','/')
        if any(part in path for part in ('/data/raw/','/runs/','/manifests/','/reports/private/')):
            raise AssertionError('research artifacts forbidden')
sys.addaudithook(audit)
sys.argv=[sys.argv[1]]
runpy.run_path(sys.argv[0],run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(script)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0, result.stderr + result.stdout
    summary = json.loads(result.stdout)
    assert summary["passed"] and summary["dataset_samples_used"] == 0
    assert summary["images_opened"] == summary["model_or_checkpoint_operations"] == 0
    invalid = subprocess.run(
        [sys.executable, str(script), "--output", str(tmp_path / "public.json")],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert invalid.returncode == 2 and not (tmp_path / "public.json").exists()
