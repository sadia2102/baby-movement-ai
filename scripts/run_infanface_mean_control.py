"""One local, frozen-reference mean-shape control; no training or downloads."""

import argparse
import json
from pathlib import Path

from infant_movement.engine.mean_shape_control import run_control


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checks", required=True)
    args = parser.parse_args()
    try:
        report = run_control(
            args.config, args.run_dir, project_root=Path.cwd(), verification_path=args.checks
        )
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "message": "comparison failed; inspect private records",
                }
            )
        )
        return 2
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
