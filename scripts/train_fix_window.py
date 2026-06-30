#!/usr/bin/env python3
"""CLI for resumable fix-window training (safe to run with nohup/tmux)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.fix_window_train import TrainConfig, run_training


def _load_config(path: Path) -> TrainConfig:
    return TrainConfig.from_dict(json.loads(path.read_text()))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resumable fix-window training. Survives disconnect; re-run to resume."
    )
    parser.add_argument("--config", type=Path, help="JSON config from notebook or prior run")
    parser.add_argument(
        "--run-name",
        type=str,
        help="Run name; loads Models/fix_window/train_config_{run_name}.json",
    )
    parser.add_argument("--no-resume", action="store_true", help="Start from epoch 1 (deletes resume checkpoint)")
    args = parser.parse_args()

    if args.config:
        config = _load_config(args.config)
    elif args.run_name:
        config_path = Path(f"Models/fix_window/train_config_{args.run_name}.json")
        if not config_path.exists():
            parser.error(f"Config not found: {config_path}. Run notebook launch cell first.")
        config = _load_config(config_path)
    else:
        parser.error("Provide --config or --run-name")

    run_training(config, resume=not args.no_resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
