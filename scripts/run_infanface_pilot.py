"""Explicit local preparation and one reviewed, bounded training-only pilot."""

import argparse
import json
from pathlib import Path

from infant_movement.datasets.infanface_pilot import PilotConfig
from infant_movement.engine.infanface_pilot import execute_pilot, prepare_pilot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="record training-only selection and annotation QC; no model"
    )
    for name in ("config", "archive", "receipt", "protocol", "run-dir"):
        prepare.add_argument("--" + name, required=True)
    run = commands.add_parser(
        "run", help="execute once after full synthetic checks and private alignment review"
    )
    run.add_argument("--run-dir", required=True)
    run.add_argument(
        "--checks", required=True, help="passing synthetic verification record for current code"
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            config = PilotConfig(**json.loads(Path(args.config).read_text()))
            expected = json.loads(Path(args.receipt).read_text())["sha256"]
            result = prepare_pilot(
                config,
                args.archive,
                expected,
                args.run_dir,
                project_root=Path.cwd(),
                protocol_path=args.protocol,
            )
        else:
            report = execute_pilot(
                args.run_dir, project_root=Path.cwd(), verification_path=args.checks
            )
            result = {
                key: value for key, value in report.items() if key != "final_checkpoint_sha256"
            }
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "message": "pilot rejected or failed; inspect private status/review records",
                }
            )
        )
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
