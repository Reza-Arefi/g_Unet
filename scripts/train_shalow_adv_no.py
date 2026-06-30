#!/usr/bin/env python3
"""CLI for Shalow adv-NO / L1 training (background-safe, resumable)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHALOW_DIR = ROOT / "Shalow"
for p in (str(ROOT), str(SHALOW_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from train.adv_no_train import AdvNOTrainConfig, run_adv_no_training


def _load_config(path: Path) -> AdvNOTrainConfig:
    return AdvNOTrainConfig.from_dict(json.loads(path.read_text()))


def _config_path_for_run(run_name: str) -> Path | None:
    candidates = [
        SHALOW_DIR / f"Models/adv_no/{run_name}/train_config_{run_name}.json",
        ROOT / f"Models/adv_no/{run_name}/train_config_{run_name}.json",
        SHALOW_DIR / f"Models/adv_no/train_config_{run_name}.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Shalow random-window training. Safe for nohup/tmux; re-run with --resume."
    )
    parser.add_argument("--config", type=Path, help="train_config_{RUN_NAME}.json path")
    parser.add_argument(
        "--run-name",
        type=str,
        help="Run name; loads Shalow/Models/adv_no/{run_name}/train_config_{run_name}.json",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from resume_{run_name}.pt after interrupt",
    )
    args = parser.parse_args()

    if args.config:
        config = _load_config(args.config)
    elif args.run_name:
        config_path = _config_path_for_run(args.run_name)
        if config_path is None:
            parser.error(
                f"Config not found for run {args.run_name!r}. "
                "Run the notebook train-config cell first."
            )
        config = _load_config(config_path)
    else:
        parser.error("Provide --config or --run-name")

    run_adv_no_training(config, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
