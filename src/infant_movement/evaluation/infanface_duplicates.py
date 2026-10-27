"""Explicit local InfAnFace audit orchestration; private results, no splits."""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from infant_movement.datasets.infanface import load_infanface_archive
from infant_movement.evaluation.image_duplicates import (
    AHASH_MAX,
    DHASH_MAX,
    METHOD_VERSION,
    ImageAuditError,
    candidate_pairs,
    exact_groups,
    fingerprint_image,
)


def file_sha256(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_archive(path: str | Path, expected: str) -> str:
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ImageAuditError("receipt must supply a lowercase SHA-256 digest")
    digest = file_sha256(path)
    if digest != expected:
        raise ImageAuditError("archive checksum differs from receipt; no image decoding performed")
    return digest


def private_output_path(path: str | Path, root: str | Path) -> Path:
    """Restrict audit output to root/reports/private, resolving symlinks first."""
    root = Path(root).resolve()
    destination = Path(path).resolve()
    private = root / "reports" / "private"
    if destination == private or not destination.is_relative_to(private):
        raise ImageAuditError(
            "sample-level output must be under reports/private in the project root"
        )
    return destination


def audit_infanface_duplicates(archive_path: str | Path, receipt_path: str | Path) -> dict:
    """Read only local files, verifying the receipt before any pixel processing."""
    try:
        receipt = json.loads(Path(receipt_path).read_text())
        expected = receipt["sha256"]
    except (ValueError, KeyError, TypeError) as error:
        raise ImageAuditError("invalid local acquisition receipt") from error
    digest = verify_archive(archive_path, expected)
    samples = load_infanface_archive(archive_path)
    fingerprints, records, failures = {}, {}, []
    with ZipFile(archive_path) as archive:
        names = [item.filename for item in archive.infolist()]
        for index, sample in enumerate(samples, 1):
            key = f"sample-{index:04d}"
            member = "infanface-public/" + sample.image_relative_path.as_posix()
            records[key] = {
                "source_row": sample.source_row,
                "archive_image_path": member,
                "source_batch": sample.image_set,
                "official_partition": sample.official_split,
                "subject_id": None,
                "source_video_id": None,
                "source_group_id": None,
                "project_split": None,
            }
            try:
                if names.count(member) != 1:
                    raise ImageAuditError("image reference must resolve to one archive member")
                info = archive.getinfo(member)
                if info.flag_bits & 1 or info.file_size > 32 * 1024**2:
                    raise ImageAuditError("encrypted or oversized image is unsupported")
                fingerprints[key] = fingerprint_image(archive.read(info))
            except (ImageAuditError, BadZipFile, KeyError, OSError) as error:
                # Never expose original names or embedded metadata on stdout.
                failures.append({"sample": key, "code": type(error).__name__})
    pairs = candidate_pairs(fingerprints)
    for pair in pairs:
        pair["cross_official_partition"] = (
            records[pair["left"]]["official_partition"]
            != records[pair["right"]]["official_partition"]
        )
    stored = exact_groups(fingerprints, oriented=False)
    display = exact_groups(fingerprints)
    verify_archive(archive_path, expected)
    source_tag_counts = Counter(
        label for fp in fingerprints.values() for label, _ in fp.source_metadata
    )
    summary = {
        "sample_count": len(samples),
        "decoded_count": len(fingerprints),
        "decode_failure_count": len(failures),
        "all_pairs_compared": len(fingerprints) * (len(fingerprints) - 1) // 2,
        "exact_stored_pixel_groups": len(stored),
        "exact_display_pixel_groups": len(display),
        "candidate_pair_count": len(pairs),
        "cross_official_partition_candidates": sum(p["cross_official_partition"] for p in pairs),
        "decoded_mode_counts": dict(Counter(fp.mode for fp in fingerprints.values())),
        "source_metadata_tag_counts": dict(source_tag_counts),
        "exif_orientation_counts": dict(
            Counter(str(fp.orientation) for fp in fingerprints.values())
        ),
        "confirmed_subject_ids": 0,
        "confirmed_video_ids": 0,
        "project_partitions_created": False,
    }
    return {
        "method": METHOD_VERSION,
        "configuration": {
            "dhash_max": DHASH_MAX,
            "ahash_max": AHASH_MAX,
            "candidate_rule": "same view pair passes both inclusive Hamming thresholds",
            "views": ["full", "full-mirror", "center80", "center80-mirror"],
            "ranking": "minimum (dHash+aHash, RGB thumbnail MAE, distances, view names)",
            "exact": "dimensions plus RGBA8 pixels; no ICC colour conversion",
        },
        "archive_sha256_before_and_after": digest,
        "summary": summary,
        "records": records,
        "fingerprints": {key: fp.private_dict() for key, fp in fingerprints.items()},
        "exact_stored_pixel_groups": stored,
        "exact_display_pixel_groups": display,
        "candidates": pairs,
        "decode_failures": failures,
        "limitations": [
            "Similarity matches require review and do not establish identity or source video.",
            "No match is not evidence of independent infants/videos; recall is uncalibrated.",
            "Pixel equality is under a declared RGBA8 conversion, not file equality.",
            "No project splits or patient/source-video assignments are made.",
        ],
    }


def save_private_audit(report: dict, output: str | Path, root: str | Path) -> None:
    destination = private_output_path(output, root)
    if destination.exists():
        raise ImageAuditError(
            "refusing to overwrite an existing private audit; choose a new run path"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
