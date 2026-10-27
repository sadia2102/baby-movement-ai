"""Local metadata audits and explicit private whole-image duplicate reports."""

import argparse
import json
from collections.abc import Sequence

from infant_movement.datasets.metadata import MetadataError, load_metadata
from infant_movement.evaluation.splits import check_grouped_splits


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local metadata and grouped split checks")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="audit an existing local JSON metadata manifest")
    audit.add_argument("manifest", help="local JSON path; keep actual manifests private")
    duplicates = commands.add_parser(
        "duplicate-audit", help="local InfAnFace image audit; private output only"
    )
    duplicates.add_argument("--archive", required=True)
    duplicates.add_argument("--receipt", required=True)
    duplicates.add_argument("--output", required=True, help="new JSON path under reports/private")
    arguments = parser.parse_args(argv)
    if arguments.command == "duplicate-audit":
        from pathlib import Path

        from infant_movement.evaluation.image_duplicates import ImageAuditError
        from infant_movement.evaluation.infanface_duplicates import (
            audit_infanface_duplicates,
            private_output_path,
            save_private_audit,
        )

        try:
            destination = private_output_path(arguments.output, Path.cwd())
            if destination.exists():
                raise ImageAuditError("refusing to overwrite an existing private audit")
            report = audit_infanface_duplicates(arguments.archive, arguments.receipt)
            save_private_audit(report, destination, Path.cwd())
        except (ValueError, OSError) as error:
            message = (
                str(error)
                if isinstance(error, ImageAuditError)
                else "cannot complete local duplicate audit"
            )
            print(json.dumps({"ok": False, "error": message}))
            return 2
        print(json.dumps(report["summary"], indent=2))
        return 0 if report["summary"]["decode_failure_count"] == 0 else 1
    try:
        report = check_grouped_splits(load_metadata(arguments.manifest))
    except (MetadataError, OSError) as error:
        # Filesystem error text may contain private paths. Do not echo those.
        message = str(error) if isinstance(error, MetadataError) else "cannot read local manifest"
        print(json.dumps({"ok": False, "error": message}))
        return 2
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.ok else 1
