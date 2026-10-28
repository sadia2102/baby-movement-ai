"""Explicit bounded synthetic-only CPU training command; no real-data option."""

import argparse
import json
from pathlib import Path

from infant_movement.engine.landmark_baseline import TrainConfig, run_synthetic_baseline


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="local TrainConfig JSON")
    parser.add_argument("--run-dir", required=True, help="new directory below project runs/")
    args = parser.parse_args(argv)
    config = TrainConfig(**json.loads(Path(args.config).read_text()))
    summary = run_synthetic_baseline(config, args.run_dir, project_root=Path.cwd())
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0 if summary["acceptance"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
