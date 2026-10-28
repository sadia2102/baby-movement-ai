"""Offline software-only verification gate; never opens research assets."""

import argparse
import importlib.metadata
import json
import re
import subprocess
import sys
from pathlib import Path

from infant_movement.engine.infanface_pilot import runtime_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    output = Path(args.output).resolve()
    if not output.is_relative_to(root / "reports/private") or output.exists():
        parser.error("choose an unused private report path")
    if Path(sys.prefix).resolve() != (root / ".venv-baseline").resolve():
        parser.error("use the project's existing .venv-baseline")
    lock = root / "environments/macos-arm64-py313-baseline.lock"
    pins = dict(
        re.findall(
            r"^([\w.-]+)==([^\s]+) --hash=sha256:[0-9a-f]{64}$", lock.read_text(), re.MULTILINE
        )
    )
    if len(pins) != 19:
        parser.error("unexpected tested baseline lock schema")
    mismatches = [
        name for name, version in pins.items() if importlib.metadata.version(name) != version
    ]
    installed = {
        re.sub(r"[-_.]+", "-", d.metadata["Name"]).lower()
        for d in importlib.metadata.distributions()
    }
    allowed = {re.sub(r"[-_.]+", "-", name).lower() for name in pins}
    allowed.add("infant-movement")
    if mismatches or installed != allowed:
        parser.error("installed environment does not exactly match the tested lock plus project")
    commands = [
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        [sys.executable, "-m", "ruff", "check", "--no-cache", "."],
        [sys.executable, "-m", "ruff", "format", "--no-cache", "--check", "."],
        [sys.executable, "-m", "pip", "--disable-pip-version-check", "--no-cache-dir", "check"],
        [str(root / ".venv/bin/python"), "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        [
            str(root / ".venv/bin/python"),
            "-m",
            "pip",
            "--disable-pip-version-check",
            "--no-cache-dir",
            "check",
        ],
        [
            sys.executable,
            "-m",
            "infant_movement",
            "audit",
            "tests/fixtures/synthetic_metadata.json",
        ],
    ]
    results = []
    for command in commands:
        result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
        results.append(
            {
                "command": command,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
        print(result.stdout.strip(), flush=True)
        if result.returncode:
            print(result.stderr.strip(), flush=True)
    record = {
        "checks_passed": all(r["returncode"] == 0 for r in results),
        "runtime_hashes": runtime_hashes(
            root, root / "reports/infanface-training-pilot-protocol-2026-10-28.md"
        ),
        "locked_distributions_verified": len(pins),
        "dependency_changes": False,
        "commands": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(record, stream, indent=2)
    return 0 if record["checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
