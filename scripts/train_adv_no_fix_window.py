#!/usr/bin/env python3
"""CLI for adv-NO fix-window training (always starts from epoch 1)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.adv_no_train import AdvNOTrainConfig, run_adv_no_training


def _load_config(path: Path) -> AdvNOTrainConfig:
    return AdvNOTrainConfig.from_dict(json.loads(path.read_text()))


def _config_path_for_run(run_name: str) -> Path | None:
    candidates = [
        Path(f"Models/adv_no/{run_name}/train_config_{run_name}.json"),
        Path(f"Models/adv_no/train_config_{run_name}.json"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="adv-NO training (L1 + perceptual + adversarial). Always starts fresh."
    )
    parser.add_argument("--config", type=Path, help="JSON config path")
    parser.add_argument(
        "--run-name",
        type=str,
        help="Run name; loads Models/adv_no/{run_name}/train_config_{run_name}.json",
    )
    args = parser.parse_args()

    if args.config:
        config = _load_config(args.config)
    elif args.run_name:
        config_path = _config_path_for_run(args.run_name)
        if config_path is None:
            parser.error(
                f"Config not found for run {args.run_name!r}. "
                "Run the notebook config cell first."
            )
        config = _load_config(config_path)
    else:
        parser.error("Provide --config or --run-name")

    run_adv_no_training(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
