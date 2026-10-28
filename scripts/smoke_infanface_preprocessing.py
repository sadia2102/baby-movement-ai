"""Explicit offline local-data smoke check; aggregate output only, no saved images."""

import argparse
import json
from collections import Counter
from pathlib import Path

from infant_movement.datasets.infanface_images import (
    METHOD_VERSION,
    open_infanface_images,
    preprocess_face,
)
from infant_movement.datasets.landmark_geometry import PreprocessingError, transform_landmarks
from infant_movement.evaluation.infanface_duplicates import private_output_path, save_private_audit


def smoke(
    archive,
    expected,
    *,
    source_index_base,
    annotation_frame,
    orientation_policy,
    output_size,
    crop_box,
):
    processed = decoded = rejected = decode_failures = outside = changed = 0
    max_error = 0.0
    orientations, rejections = Counter(), Counter()
    with open_infanface_images(archive, expected_sha256=expected) as reader:
        official = dict(Counter(sample.official_split for sample in reader.samples))
        total = len(reader.samples)
        for sample in reader.samples:
            original = sample.landmarks
            try:
                face = reader.load(sample)
            except PreprocessingError:
                decode_failures += 1
                continue
            decoded += 1
            orientations[str(face.orientation)] += 1
            try:
                result = preprocess_face(
                    face,
                    source_index_base=source_index_base,
                    annotation_frame=annotation_frame,
                    orientation_policy=orientation_policy,
                    crop_box=crop_box,
                    output_size=output_size,
                )
                try:
                    recovered = transform_landmarks(result.landmarks, result.inverse)
                    for source, restored in zip(original, recovered, strict=True):
                        if source is not None:
                            max_error = max(
                                max_error,
                                *(abs(a - b) for a, b in zip(source, restored, strict=True)),
                            )
                    outside += sum(not valid for valid in result.source_in_bounds)
                    changed += int(sample.landmarks != original or result.source is not sample)
                    processed += 1
                finally:
                    result.image.close()
            except PreprocessingError:
                rejected += 1
                invalid_exif = face.orientation is not None and (
                    type(face.orientation) is not int or face.orientation not in range(1, 9)
                )
                rejections[
                    "invalid-exif-orientation" if invalid_exif else "preprocessing-input"
                ] += 1
            finally:
                face.image.close()
    return {
        "method": METHOD_VERSION,
        "configuration": {
            "source_index_base": source_index_base,
            "annotation_frame": annotation_frame,
            "orientation_policy": orientation_policy,
            "crop_box": crop_box,
            "crop_kind": "full-raster" if crop_box is None else "caller-supplied",
            "output_size_wh": output_size,
            "source_coordinate_evidence": "UNVERIFIED",
            "resampling": "uniform affine BICUBIC",
            "padding_rgb": [0, 0, 0],
            "alpha_background_rgb": [255, 255, 255],
        },
        "archive_sha256_verified_before_after": expected,
        "sample_count": total,
        "decoded_count": decoded,
        "processed_count": processed,
        "decode_failure_count": decode_failures,
        "preprocessing_rejected_count": rejected,
        "rejection_reason_counts": dict(rejections),
        "orientation_counts": dict(orientations),
        "official_partitions": official,
        "source_out_of_bounds_landmarks_in_processed": outside,
        "original_annotation_changes": changed,
        "max_roundtrip_error_source_pixels": max_error,
        "project_splits_created": False,
        "limitations": [
            "Source origin and annotation raster frame are unverified caller hypotheses.",
            "Round trips validate arithmetic; anatomical alignment/conventions remain unverified.",
            "Bounds are geometric footprint flags, not visibility/clinical annotations.",
            "No subject/video independence or clinical/model validity is established.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--source-index-base", type=int, choices=(0, 1), required=True)
    parser.add_argument("--annotation-frame", choices=("stored", "display"), required=True)
    parser.add_argument("--orientation-policy", choices=("stored", "normalize"), required=True)
    parser.add_argument(
        "--output-size", type=int, nargs=2, default=(256, 256), metavar=("WIDTH", "HEIGHT")
    )
    parser.add_argument("--crop-box", type=int, nargs=4, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"))
    parser.add_argument("--output", required=True, help="new aggregate JSON under reports/private")
    args = parser.parse_args(argv)
    try:
        output = private_output_path(args.output, Path.cwd())
        if output.exists():
            raise PreprocessingError("refusing to overwrite an existing smoke record")
        expected = json.loads(Path(args.receipt).read_text())["sha256"]
        report = smoke(
            args.archive,
            expected,
            source_index_base=args.source_index_base,
            annotation_frame=args.annotation_frame,
            orientation_policy=args.orientation_policy,
            output_size=tuple(args.output_size),
            crop_box=args.crop_box,
        )
        save_private_audit(report, output, Path.cwd())
    except (ValueError, KeyError, TypeError, OSError):
        print(
            json.dumps(
                {"ok": False, "error": "cannot complete the local preprocessing smoke check"}
            )
        )
        return 2
    print(json.dumps(report, indent=2))
    return int(
        bool(
            report["decode_failure_count"]
            or report["preprocessing_rejected_count"]
            or report["original_annotation_changes"]
            or report["max_roundtrip_error_source_pixels"] > 1e-8
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
