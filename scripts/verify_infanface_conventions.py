"""Offline manufactured-coordinate checks; never reads dataset/model artifacts."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import subprocess
import sys
from pathlib import Path

from infant_movement.datasets.coordinate_conventions import (
    decode_normalized_edges,
    encode_normalized_edges,
    source_centres_to_edges,
)
from infant_movement.datasets.landmark_geometry import crop_geometry, letterbox_geometry
from infant_movement.evaluation.infanface_conventions import (
    ASSUMED_SLOT_ORDER,
    REFERENCE_EYE_SLOTS,
    REFERENCE_REVISION,
    infanface_reference_nme,
    legacy_hrnet_pixel_reference,
)
from infant_movement.evaluation.provenance_readiness import locked_pins


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def synthetic_checks():
    target = [(0.0, 0.0)] * 68
    target[45] = (3.0, 4.0)
    prediction = [(x + 3, y + 4) for x, y in target]
    report = infanface_reference_nme(
        [prediction], [target], [[True] * 68], slot_order=ASSUMED_SLOT_ORDER, units="source-pixels"
    )
    checks = {
        "nme_fraction_1": math.isclose(report["nme_fraction"], 1, abs_tol=1e-12, rel_tol=0),
        "nme_percent_100": math.isclose(report["nme_percent"], 100, abs_tol=1e-12, rel_tol=0),
        "nme_points_68": report["eligible_points"] == 68,
    }
    bad_mask = [True] * 68
    bad_mask[36] = False
    missing = infanface_reference_nme(
        [prediction], [target], [bad_mask], slot_order=ASSUMED_SLOT_ORDER, units="source-pixels"
    )
    checks["missing_anchor_undefined"] = (
        missing["nme_fraction"] is None and missing["eligible_points"] == 67
    )
    points = ((3.0, 2.0), (-2.0, 9.0), None)
    _, crop, size = crop_geometry((2, 1, 8, 5), (10, 6))
    resize, _, _ = letterbox_geometry(size, (12, 12))
    forward = source_centres_to_edges(source_index_base=0).then(crop).then(resize)
    normalized = encode_normalized_edges(points, forward=forward, output_size=(12, 12))
    recovered = decode_normalized_edges(normalized, inverse=forward.inverse(), output_size=(12, 12))
    error = max(
        abs(a - b)
        for p, q in zip(points, recovered, strict=True)
        if p is not None
        for a, b in zip(p, q, strict=True)
    )
    checks["crop_padding_expected"] = all(
        math.isclose(a, b, abs_tol=1e-12, rel_tol=0)
        for a, b in zip(normalized[0], (0.25, 5 / 12), strict=True)
    )
    checks["geometry_roundtrip"] = error <= 1e-10 and recovered[-1] is None
    base1 = source_centres_to_edges(source_index_base=1)
    checks["origin_explicit"] = encode_normalized_edges(
        ((1, 1),), forward=base1, output_size=(1, 1)
    ) == ((0.5, 0.5),)
    checks["legacy_square_forward"] = legacy_hrnet_pixel_reference(
        (11, 21), center=(50, 50), scale=1, resolution=100, inverse=False
    ) == (31, 36)
    checks["legacy_truncates_toward_zero"] = legacy_hrnet_pixel_reference(
        (0, 0), center=(200, 200), scale=1, resolution=100, inverse=False
    ) == (-49, -49)
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "max_roundtrip_error": error,
        "synthetic_reference_nme": report,
        "dataset_samples_used": 0,
        "images_opened": 0,
        "model_or_checkpoint_operations": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="optional unused JSON path under project reports/private")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve() if args.output else None
    if output and (
        output.suffix != ".json"
        or not output.is_relative_to(root / "reports/private")
        or output.exists()
    ):
        parser.error("choose an unused JSON path under project reports/private")
    try:
        registry_path = root / "configs/datasets/infanface_conventions.json"
        registry = json.loads(registry_path.read_text())
        if (
            registry["upstream_revision"] != REFERENCE_REVISION
            or tuple(registry["implementation_observation"]["normalizer_index_pair_zero_based"])
            != REFERENCE_EYE_SLOTS
            or any(
                not re.fullmatch("[0-9a-f]{64}", v["sha256"]) for v in registry["files"].values()
            )
        ):
            raise ValueError("source evidence registry mismatch")
        # Resolve the already-tested local includes; no package or network resolution.
        baseline = Path(sys.prefix).resolve() == (root / ".venv-baseline").resolve()
        lock = (
            root
            / "environments"
            / (
                "macos-arm64-py313-baseline.lock"
                if baseline
                else "macos-arm64-py313-image-audit.lock"
            )
        )
        pins = locked_pins(lock, root / "environments")
        installed = {
            re.sub("[-_.]+", "-", d.metadata["Name"]).lower(): d.version
            for d in importlib.metadata.distributions()
        }
        expected = {re.sub("[-_.]+", "-", name).lower(): version for name, version in pins.items()}
        expected["infant-movement"] = "0.1.0"
        if installed != expected:
            raise ValueError("installed inventory differs from existing lock plus project")
        result = synthetic_checks()
        if not result["passed"]:
            raise ValueError("manufactured arithmetic verification failed")

        def git(*args):
            return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

        paths = list((root / "src").rglob("*.py")) + list((root / "tests").glob("*.py"))
        paths += list((root / "scripts").glob("*.py")) + [
            registry_path,
            lock,
            root / "reports/infanface-conventions-protocol-2026-11-16.md",
        ]
        evidence = {
            **result,
            "python": sys.version,
            "platform": platform.platform(),
            "installed_versions": installed,
            "lock_sha256": digest(lock),
            "source_registry": registry,
            "source_registry_sha256": digest(registry_path),
            "public_file_sha256": {str(p.relative_to(root)): digest(p) for p in paths},
            "git_head": git("rev-parse", "HEAD"),
            "git_branch": git("branch", "--show-current"),
            "git_dirty": bool(git("status", "--porcelain")),
            "upstream_source_executed": False,
            "network_operations": 0,
            "real_data_nme_computed": False,
            "historical_results_changed": False,
            "optional_modules_loaded": sorted(
                n for n in ("torch", "PIL", "numpy", "torchvision") if n in sys.modules
            ),
        }
        if evidence["optional_modules_loaded"]:
            raise ValueError("verification must not load image/model/numeric extensions")
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x") as stream:
                json.dump(evidence, stream, indent=2, allow_nan=False)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError) as error:
        print(json.dumps({"passed": False, "error_type": type(error).__name__}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
