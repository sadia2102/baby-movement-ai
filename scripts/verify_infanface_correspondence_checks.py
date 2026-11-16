"""Offline full verification plus protocol/config/preservation binding; no research data."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from infant_movement.engine.image_correspondence import comparison_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, output = Path.cwd().resolve(), Path(args.output).resolve()
    base = output.with_suffix(".base.json")
    if not output.is_relative_to(root / "reports/private") or output.exists() or base.exists():
        parser.error("choose unused ignored software verification paths")
    result = subprocess.run(
        [sys.executable, "scripts/verify_infanface_pilot_checks.py", "--output", str(base)],
        check=False,
    )
    if not base.is_file():
        return result.returncode or 1
    record = json.loads(base.read_text())
    record["comparison_hashes"] = comparison_hashes(root)
    record["base_record"] = str(base)
    with output.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    return 0 if result.returncode == 0 and record["checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
