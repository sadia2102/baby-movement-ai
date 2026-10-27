"""Inspect local source/lineage metadata; no URL resolution or identity inference."""

import argparse
import io
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

from infant_movement.evaluation.image_duplicates import ImageAuditError
from infant_movement.evaluation.image_provenance import ProvenanceError, extract_xmp_references
from infant_movement.evaluation.infanface_duplicates import (
    private_output_path,
    save_private_audit,
    verify_archive,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--audit-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        output = private_output_path(args.output, Path.cwd())
        if output.exists():
            raise ImageAuditError("refusing to overwrite an existing private source audit")
        report = json.loads(Path(args.audit_report).read_text())
        expected = report["archive_sha256_before_and_after"]
        verify_archive(args.archive, expected)
        samples, groups, failures = {}, defaultdict(lambda: defaultdict(list)), []
        metadata_keys = defaultdict(int)
        with ZipFile(args.archive) as archive:
            for sample, record in report["records"].items():
                refs, original_date = {}, None
                member = archive.getinfo(record["archive_image_path"])
                if member.file_size > 32 * 1024**2 or member.flag_bits & 1:
                    raise ImageAuditError("unsupported source image container")
                with Image.open(io.BytesIO(archive.read(member))) as image:
                    for key in image.info:
                        metadata_keys[key] += 1
                    exif = image.getexif()
                    if 34665 in exif:
                        value = exif.get_ifd(34665).get(36867)
                        if isinstance(value, str):
                            original_date = value[:4096]
                    for field in ("xmp", "XML:com.adobe.xmp"):
                        if field not in image.info:
                            continue
                        try:
                            extracted = extract_xmp_references(image.info[field])
                        except ProvenanceError as error:
                            failures.append({"sample": sample, "field": field, "error": str(error)})
                            continue
                        for key, values in extracted.items():
                            refs.setdefault(key, []).extend(values)
                refs = {key: sorted(set(values)) for key, values in refs.items()}
                for key, values in refs.items():
                    for value in values:
                        groups[key][value].append(sample)
                samples[sample] = {
                    "xmp_references": refs,
                    "exif_datetime_original": original_date,
                    "source_video_id": None,
                    "subject_id": None,
                }
        documents = defaultdict(set)
        for sample, fields in samples.items():
            for key, values in fields["xmp_references"].items():
                if "documentid" in key.lower():
                    for value in values:
                        documents[value].add(sample)
        lineage_pairs = sorted(
            {
                tuple(sorted(pair))
                for members in documents.values()
                for pair in combinations(members, 2)
            }
        )
        summary = {
            "sample_count": len(samples),
            "samples_with_xmp_source_or_lineage_refs": sum(
                bool(s["xmp_references"]) for s in samples.values()
            ),
            "samples_with_nested_exif_datetime_original": sum(
                s["exif_datetime_original"] is not None for s in samples.values()
            ),
            "embedded_source_http_url_values": sum(
                "http://" in value.lower() or "https://" in value.lower()
                for groups_by_value in groups.values()
                for value in groups_by_value
            ),
            "cross_field_lineage_candidate_pairs": len(lineage_pairs),
            "cross_official_partition_lineage_pairs": sum(
                report["records"][left]["official_partition"]
                != report["records"][right]["official_partition"]
                for left, right in lineage_pairs
            ),
            "xml_failure_count": len(failures),
            "gps_values_inspected": False,
            "subject_ids_assigned": 0,
            "source_video_ids_assigned": 0,
            "project_partitions_created": False,
        }
        verify_archive(args.archive, expected)
        result = {
            "method": "local-source-metadata-v1",
            "archive_sha256_before_and_after": expected,
            "summary": summary,
            "samples": samples,
            "metadata_key_counts": dict(metadata_keys),
            "groups": {key: dict(value) for key, value in groups.items()},
            "lineage_candidate_pairs": lineage_pairs,
            "xml_failures": failures,
            "limitations": [
                "Document lineage and timestamps are not infant/video identity evidence.",
                "Unparsed XMP and uninspected fields remain UNVERIFIED; no URLs are resolved.",
            ],
        }
        save_private_audit(result, output, Path.cwd())
    except (ValueError, OSError, KeyError) as error:
        message = (
            str(error)
            if isinstance(error, ImageAuditError)
            else "cannot complete local source audit"
        )
        print(json.dumps({"ok": False, "error": message}))
        return 2
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
