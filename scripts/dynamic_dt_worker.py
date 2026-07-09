#!/usr/bin/env python3
"""Run one dynamic_dt export / train / rollout job (used by multi-GPU orchestration)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MW_DIR = ROOT / "Magic_window"
DT_DIR = MW_DIR / "dynamic_dt"
for p in (ROOT, MW_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dynamic_dt.rollout_core import evaluate_rollout  # noqa: E402
from dynamic_dt.run_registry import (  # noqa: E402
    save_rollout_results,
    update_comparison_index,
)
from dynamic_dt.window_catalog import get_window_spec, primary_export_component  # noqa: E402
from src.train.adv_no_train import AdvNOTrainConfig, run_adv_no_training  # noqa: E402


def cmd_train(args: argparse.Namespace) -> int:
    config = AdvNOTrainConfig.from_dict(json.loads(Path(args.config).read_text()))
    config.device = "cuda:0"
    run_adv_no_training(config, resume=args.resume)
    return 0


def cmd_rollout(args: argparse.Namespace) -> int:
    dt_dir = Path(args.dt_dir)
    job = json.loads(Path(args.job_config).read_text())
    export_key = job["export_key"]
    dt_days = int(job["dt_days"])
    window = get_window_spec(export_key)
    channel_names = window["channel_names_rollout"]

    from dynamic_dt.run_registry import resolve_model_checkpoint, run_dir, run_name, weight_decay_run_tag, RUN_TAG
    from dynamic_dt.dt_config import dt_hours
    from dynamic_dt.window_paths import test_prefix

    dt = dt_hours(dt_days)
    models_dir = dt_dir / "Models" / "adv_no"
    run_tag = weight_decay_run_tag(RUN_TAG, float(job.get("weight_decay", 1e-4)))
    run_name_value = run_name(export_key, dt_hours=dt, run_tag=run_tag)
    run_dir_path = Path(job.get("model_run_dir") or run_dir(
        models_dir, export_key, dt_hours=dt, run_tag=run_tag
    ))
    export_component = primary_export_component(export_key)
    data_prefix = test_prefix(dt_dir, export_key, component=export_component, dt_hours=dt)
    test_rollout_path = dt_dir / "Test&Rollout" / run_name_value

    train_cfg, model_path, stats_path, _, _, _ = resolve_model_checkpoint(
        run_dir_path, expected_run_name=run_name_value
    )
    results = evaluate_rollout(
        data_prefix=data_prefix,
        model_path=model_path,
        stats_path=stats_path,
        dt_days=dt_days,
        train_cfg=train_cfg,
        output_dir=test_rollout_path,
        device="cuda:0",
        plot_indices=job.get("plot_indices") or None,
        channel_names=channel_names,
    )
    save_rollout_results(test_rollout_path, results)
    from dynamic_dt.rollout_metrics_plots import plot_rollout_metrics_from_results

    results["_run_dir"] = str(test_rollout_path)
    results["run_name"] = run_name_value
    plot_rollout_metrics_from_results(results, out_dir=test_rollout_path)
    update_comparison_index(
        models_dir,
        run_name_value=run_name_value,
        export_key=export_key,
        run_dir_path=run_dir_path,
        rollout_results=results,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="dynamic_dt single-job worker")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Run one adv-NO training job")
    train_p.add_argument("--config", type=Path, required=True)
    train_p.add_argument("--resume", action="store_true")

    rollout_p = sub.add_parser("rollout", help="Run one rollout evaluation job")
    rollout_p.add_argument("--job-config", type=Path, required=True)
    rollout_p.add_argument("--dt-dir", type=Path, default=DT_DIR)

    args = parser.parse_args()
    if args.command == "train":
        return cmd_train(args)
    if args.command == "rollout":
        return cmd_rollout(args)
    parser.error(f"Unknown command {args.command!r}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
