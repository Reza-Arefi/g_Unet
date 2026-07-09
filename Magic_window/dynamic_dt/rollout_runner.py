"""Launch rollout evaluation for one dt."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from dynamic_dt.device_plan import cuda_visible_devices, device_string
from dynamic_dt.dt_config import dt_hours
from dynamic_dt.rollout_core import evaluate_rollout
from dynamic_dt.run_registry import (
    RUN_TAG,
    resolve_model_checkpoint,
    run_dir,
    run_name,
    save_rollout_results,
    test_rollout_dir,
    update_comparison_index,
    weight_decay_run_tag,
)
from dynamic_dt.window_catalog import get_window_spec, primary_export_component
from dynamic_dt.window_paths import test_prefix


def rollout_one_dt(
    *,
    dt_dir: Path,
    root: Path,
    export_key: str,
    dt_days: int,
    log_dir: Path,
    weight_decay: float = 1e-4,
    plot_indices: Sequence[int] | None = None,
    detached: bool = True,
    model_run_dir: Path | str | None = None,
) -> dict[str, Any]:
    window = get_window_spec(export_key)
    export_component = primary_export_component(export_key)
    channel_names = window["channel_names_rollout"]
    dt = dt_hours(dt_days)
    device_id = cuda_visible_devices(dt_days)

    models_dir = dt_dir / "Models" / "adv_no"
    run_tag = weight_decay_run_tag(RUN_TAG, weight_decay)
    run_name_value = run_name(export_key, dt_hours=dt, run_tag=run_tag)
    run_dir_path = Path(model_run_dir) if model_run_dir else run_dir(
        models_dir, export_key, dt_hours=dt, run_tag=run_tag
    )
    test_rollout_path = test_rollout_dir(dt_dir, export_key, dt_hours=dt, run_tag=run_tag)
    data_prefix = test_prefix(dt_dir, export_key, component=export_component, dt_hours=dt)

    train_cfg, model_path, stats_path, _, _, _ = resolve_model_checkpoint(
        run_dir_path, expected_run_name=run_name_value
    )
    log_path = log_dir / f"rollout_{export_key}_dt{dt_days}d_gpu{device_id}.log"

    job = {
        "dt_days": dt_days,
        "device_id": device_id,
        "run_name": run_name_value,
        "run_dir": str(run_dir_path),
        "rollout_dir": str(test_rollout_path),
        "data_prefix": data_prefix,
        "model_path": str(model_path),
    }

    if detached:
        rollout_cfg = {
            "export_key": export_key,
            "dt_days": dt_days,
            "weight_decay": weight_decay,
            "plot_indices": list(plot_indices or []),
            "model_run_dir": str(run_dir_path),
        }
        cfg_path = test_rollout_path / f"rollout_job_dt{dt_days}.json"
        test_rollout_path.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(rollout_cfg, indent=2))

        env = {**os.environ, "CUDA_VISIBLE_DEVICES": device_id}
        cmd = [
            sys.executable,
            str(root / "scripts/dynamic_dt_worker.py"),
            "rollout",
            "--job-config",
            str(cfg_path),
            "--dt-dir",
            str(dt_dir),
        ]
        with open(log_path, "a") as log_file:
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                cwd=root,
                env=env,
            )
        job["pid"] = proc.pid
        job["log_path"] = str(log_path)
        job["job_config"] = str(cfg_path)
        return job

    os.environ["CUDA_VISIBLE_DEVICES"] = device_id
    results = evaluate_rollout(
        data_prefix=data_prefix,
        model_path=model_path,
        stats_path=stats_path,
        dt_days=dt_days,
        train_cfg=train_cfg,
        output_dir=test_rollout_path,
        device=device_string(dt_days),
        plot_indices=plot_indices,
        channel_names=channel_names,
    )
    save_rollout_results(test_rollout_path, results)
    from dynamic_dt.rollout_metrics_plots import plot_rollout_metrics_from_results

    results["run_name"] = run_name_value
    results["_run_dir"] = str(test_rollout_path)
    plot_rollout_metrics_from_results(results, out_dir=test_rollout_path)
    update_comparison_index(
        models_dir,
        run_name_value=run_name_value,
        export_key=export_key,
        run_dir_path=run_dir_path,
        rollout_results=results,
    )
    job["log_path"] = str(log_path)
    job["results"] = results
    return job
