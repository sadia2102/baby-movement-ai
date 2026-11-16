"""One local metadata-only review. Never opens archive members or model/image payloads."""

import argparse
import importlib.metadata
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from infant_movement.evaluation.provenance_readiness import (
    aggregate_review,
    digest,
    evidence_matrix,
    frozen_cohort,
    locked_pins,
    match_cohort,
    metadata_path,
    new_review_directory,
    relevant_decisions,
)

PROTOCOL = "reports/infanface-provenance-review-protocol-2026-11-16.md"
PILOT = "runs/infanface-train-pilot-2026-10-28"
MEAN = "runs/infanface-mean-control-2026-10-28-verified"
CORRESPONDENCE = "runs/infanface-correspondence-2026-11-16"
AUDIT = "reports/private/infanface-duplicate-audit-2026-10-27.json"
SOURCE = "reports/private/infanface-source-audit-2026-10-27.json"
STRUCTURED = (
    "reports/private/infanface-duplicate-review-2026-10-27/structured-source-provenance.json"
)
DECISIONS = "reports/private/infanface-duplicate-review-2026-10-27/review-decisions.json"
CONFIG = {
    "protocol_id": "infanface-provenance-readiness-v1",
    "cohort": PILOT,
    "cohort_images": 32,
    "official_partition": "infanface-train",
    "new_data": False,
    "model_execution": False,
    "image_inspection": False,
    "archive_members_opened": False,
    "network": False,
    "create_splits": False,
    "environment": ".venv",
    "protocol": PROTOCOL,
}


def write_new(path, obj):
    with path.open("x") as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)


def review(root, output):
    root = Path(root).resolve()
    if Path(sys.prefix).resolve() != (root / ".venv").resolve():
        raise ValueError("use the existing metadata .venv")
    if not (root / PROTOCOL).is_file():
        raise ValueError("write the dated protocol before classifying evidence")
    path = new_review_directory(output, root)
    access, verified, skipped = {}, {}, {}

    def read(name):
        p = metadata_path(root / name, root)
        access[str(p)] = digest(p)
        return json.loads(p.read_text())

    def verify(hashes, label):
        for name, expected in hashes.items():
            p = (root / name).resolve()
            try:
                metadata_path(p, root)
            except ValueError:
                skipped[str(p)] = "sealed payload; recorded digest retained without reopening"
                continue
            actual = digest(p)
            if actual != expected:
                raise ValueError(f"recorded metadata integrity mismatch: {label}")
            verified[str(p)] = actual

    def status(value, **fields):
        (path / "status.json").write_text(
            json.dumps(
                {"status": value, "recorded_at_utc": datetime.now(UTC).isoformat(), **fields},
                indent=2,
            )
        )

    status("running", review_kind="metadata-only")
    try:
        write_new(path / "config.json", CONFIG)
        initial_files = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=root, text=True
        ).splitlines()
        preserved = {}
        for name in initial_files:
            try:
                p = metadata_path(root / name, root)
            except ValueError:
                continue
            preserved[str(p)] = digest(p)
        # Hash only metadata/docs in historical runs; sealed payloads get stat
        # preservation checks without opening or semantically inspecting them.
        sealed_stats = {}
        for folder in (root / "runs").iterdir():
            if folder == path:
                continue
            for p in folder.rglob("*"):
                if not p.is_file():
                    continue
                try:
                    metadata_path(p, root)
                except ValueError:
                    s = p.stat()
                    sealed_stats[str(p)] = [s.st_size, s.st_mtime_ns, s.st_ino]
                else:
                    preserved[str(p)] = digest(p)
        starting = {
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "git_status": subprocess.check_output(
                ["git", "status", "--short", "--branch"], cwd=root, text=True
            ),
            "preserved_metadata_hashes": preserved,
            "sealed_payload_stats": sealed_stats,
        }
        write_new(path / "starting-state.json", starting)
        (path / "starting-tracked.diff").write_bytes(
            subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=root)
        )
        pins = locked_pins(
            root / "environments/macos-arm64-py313-image-audit.lock", root / "environments"
        )
        installed = {
            re.sub(r"[-_.]+", "-", d.metadata["Name"]).lower(): d.version
            for d in importlib.metadata.distributions()
        }
        assert len(pins) == 10
        assert set(installed) == {re.sub(r"[-_.]+", "-", k).lower() for k in pins} | {
            "infant-movement"
        }
        assert all(importlib.metadata.version(k) == v for k, v in pins.items())
        commands = [
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "tests/test_provenance_readiness.py",
                "tests/test_metadata.py",
                "tests/test_splits.py",
                "tests/test_cli.py",
            ],
            [sys.executable, "-m", "ruff", "check", "--no-cache", "."],
            [sys.executable, "-m", "ruff", "format", "--no-cache", "--check", "."],
            [sys.executable, "-m", "pip", "--disable-pip-version-check", "--no-cache-dir", "check"],
            [
                sys.executable,
                "-m",
                "infant_movement",
                "audit",
                "tests/fixtures/synthetic_metadata.json",
            ],
        ]
        checks = []
        for command in commands:
            result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
            checks.append(
                {
                    "command": command,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            )
            print(result.stdout.strip(), flush=True)
        write_new(
            path / "checks.json",
            {
                "commands": checks,
                "locked_distributions": pins,
                "python": sys.version,
                "installed": installed,
            },
        )
        if any(c["returncode"] for c in checks):
            raise ValueError("metadata-only software checks failed")
        prepared, provenance = read(f"{PILOT}/prepared.json"), read(f"{PILOT}/provenance.json")
        verify(
            {
                f"{PILOT}/{name}": prepared[key]
                for name, key in (
                    ("selection.json", "selection_file_sha256"),
                    ("config.json", "config_sha256"),
                    ("coverage.json", "coverage_sha256"),
                )
            },
            "prepared cohort",
        )
        verify(prepared["runtime_hashes"], "original pilot software")
        verify(provenance["source_config_hashes"], "original pilot source/config")
        for name in (
            "selection_sha256",
            "config_sha256",
            "coverage_sha256",
            "archive_sha256",
            "protocol_sha256",
        ):
            assert prepared[name] == provenance[name]
        rows = frozen_cohort(read(f"{PILOT}/selection.json"), prepared)
        coverage, config = read(f"{PILOT}/coverage.json"), read(f"{PILOT}/config.json")
        assert config == read("configs/experiments/infanface_train_pilot.json")
        assert coverage["usable_images"] == 32 and coverage["eligible_point_slots"] == 2176
        assert read(f"{PILOT}/status.json")["status"] == "complete"
        for reference in (MEAN, CORRESPONDENCE):
            frozen = read(f"{reference}/frozen-reference.json")
            verify(frozen["original_files"], "later frozen pilot metadata")
            verify(frozen["historically_recorded_hashes"], "historical pilot references")
            assert frozen["selection_sha256"] == prepared["selection_sha256"]
            assert (
                frozen["checkpoint_sha256"]
                == read(f"{PILOT}/summary.json")["final_checkpoint_sha256"]
            )
            manifest = read(f"{reference}/output-sha256.json")
            verify(
                {f"{reference}/{k}": v for k, v in manifest["files"].items()},
                "later metadata output manifest",
            )
            prior_provenance = read(f"{reference}/provenance.json")
            verify(prior_provenance["source_config_hashes"], "later source/config")
            assert read(f"{reference}/status.json")["status"] == "complete"
        # Verify recorded retained-evidence hashes without opening their imagery.
        for name, fields in (
            ("reports/private/baseline-preservation-2026-10-28.json", ["retained_file_hashes"]),
            ("reports/private/infanface-duplicate-method-2026-10-27.json", ["file_sha256"]),
        ):
            record = read(name)
            for field in fields:
                verify(record[field], "retained source evidence")
        archive = root / "data/raw/infanface/InfAnFace.zip"
        receipt = read("data/raw/infanface/acquisition-receipt-2026-10-27.json")
        assert digest(archive) == receipt["sha256"] == prepared["archive_sha256"]
        verified[str(archive)] = receipt["sha256"]
        verify(
            {
                "data/raw/infanface/author-readme-received-2026-10-27.md": receipt[
                    "author_readme_sha256"
                ],
                "data/raw/infanface/author-index-received-2026-10-27.html": receipt[
                    "author_index_sha256"
                ],
                "reports/private/infanface-bundled-readme-2026-10-27.md": receipt[
                    "author_readme_sha256"
                ],
            },
            "retained access documents",
        )
        audit, source, structured, decisions = (
            read(p) for p in (AUDIT, SOURCE, STRUCTURED, DECISIONS)
        )
        assert (
            audit["archive_sha256_before_and_after"]
            == source["archive_sha256_before_and_after"]
            == receipt["sha256"]
        )
        keys = match_cohort(rows, audit["records"])
        relevant = relevant_decisions(
            keys, audit["records"], audit["candidates"], decisions["pairs"]
        )
        evidence = [
            {"path": p, "sha256": access[str(root / p)]}
            for p in (f"{PILOT}/selection.json", AUDIT, SOURCE, STRUCTURED, DECISIONS)
        ]
        matrix = evidence_matrix(
            rows, keys, audit["records"], source, structured, relevant, evidence
        )
        summary = aggregate_review(matrix, relevant, source)
        write_new(path / "private-evidence-matrix.json", {"ordered_examples": matrix})
        write_new(path / "private-relevant-prior-decisions.json", {"pairs": relevant})
        write_new(path / "summary.json", summary)
        for name in initial_files:
            # All pre-existing user changes remain byte-for-byte intact at this stage.
            if str(root / name) in preserved:
                assert digest(root / name) == preserved[str(root / name)]
        for name, expected in preserved.items():
            assert digest(name) == expected
        for name, expected in sealed_stats.items():
            s = Path(name).stat()
            assert [s.st_size, s.st_mtime_ns, s.st_ino] == expected
        assert not any(k == "torch" or k.startswith("PIL") for k in sys.modules)
        write_new(
            path / "evidence-integrity.json",
            {
                "metadata_read_and_hashed": access,
                "verified_recorded_hashes": verified,
                "sealed_recorded_references_not_reopened": skipped,
                "archive_bytes_hashed_without_opening_members": True,
                "sealed_payload_preservation": "size/mtime/inode only; no fresh content-hash claim",
                "new_metadata_snapshots_are_not_retrospective_authentication": True,
            },
        )
        status(
            "complete",
            metadata_only=True,
            historical_metadata_unchanged=True,
            preexisting_user_changes_unchanged=True,
            new_relationships=0,
            no_model_or_image_imports=True,
            no_partitions_created=True,
        )
        write_new(
            path / "output-sha256.json",
            {
                "algorithm": "sha256",
                "files": {p.name: digest(p) for p in path.iterdir() if p.is_file()},
                "scope": "all review outputs except this manifest",
            },
        )
        return summary
    except Exception as error:
        status("failed", error_type=type(error).__name__, error=str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = review(Path.cwd(), args.output)
    except (ValueError, OSError, KeyError, AssertionError) as error:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "message": "review stopped; inspect private records",
                }
            )
        )
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
