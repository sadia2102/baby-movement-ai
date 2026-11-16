"""Explicit single frozen-checkpoint correspondence comparison; no training."""

import argparse
import json
from pathlib import Path

from infant_movement.engine.image_correspondence import run_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checks", required=True)
    args = parser.parse_args()
    try:
        summary = run_comparison(
            args.config, args.run_dir, project_root=Path.cwd(), verification_path=args.checks
        )
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "message": "comparison failed; inspect private failure evidence",
                }
            )
        )
        return 2
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
