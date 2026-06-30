#!/usr/bin/env python3
"""CLI for resumable random-window export (safe to run with nohup/tmux)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.export.random_window_export import (
    RandomWindowExportConfig,
    export_train_epoch_windows,
    merge_export_chunks,
    run_export,
)


def _load_config(path: Path) -> RandomWindowExportConfig:
    data = json.loads(path.read_text())
    return RandomWindowExportConfig.from_dict(data)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resumable random-window export. Survives disconnect; re-run to resume."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON config written by Shalow/01_export_random_window.ipynb",
    )
    parser.add_argument("--output-prefix", type=str, help="Override output_prefix in config")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh (requires new prefix or delete state)",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Only merge existing chunks into final .npy",
    )
    parser.add_argument(
        "--epoch-windows-only",
        action="store_true",
        help="Only pre-export per-epoch train windows (pool .npy must exist)",
    )
    args = parser.parse_args()

    if args.config:
        config = _load_config(args.config)
    elif args.output_prefix:
        config_path = Path(f"{args.output_prefix}_export_config.json")
        if not config_path.exists():
            parser.error(f"Config not found: {config_path}. Run notebook setup cells first.")
        config = _load_config(config_path)
    else:
        parser.error("Provide --config or --output-prefix")

    if args.output_prefix:
        config.output_prefix = args.output_prefix

    if args.merge_only:
        merge_export_chunks(config)
        return 0

    if args.epoch_windows_only:
        export_train_epoch_windows(config)
        return 0

    run_export(config, resume=not args.no_resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
