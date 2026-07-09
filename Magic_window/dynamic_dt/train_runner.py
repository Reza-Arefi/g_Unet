"""Build adv-NO train configs and launch training for one dt."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dynamic_dt.device_plan import cuda_visible_devices, device_string
from dynamic_dt.dt_config import dt_hours
from dynamic_dt.run_registry import (
    RUN_TAG,
    collect_dataset_stats,
    run_dir,
    run_name,
    save_dataset_stats,
    update_comparison_index,
    weight_decay_run_tag,
)
from dynamic_dt.sequence_config import CONDITIONING_STEPS
from dynamic_dt.window_catalog import get_window_spec, primary_export_component
from dynamic_dt.window_paths import test_prefix, train_prefix, val_prefix
from src.train.adv_no_train import AdvNOTrainConfig, run_adv_no_training


def build_train_config(
    *,
    dt_dir: Path,
    export_key: str,
    dt_days: int,
    weight_decay: float = 1e-4,
    epochs: int = 100,
    optimizer: str = "adamw",
    max_samples: int | None = None,
    val_split: float = 0.3,
    val_split_mode: str = "random",
    val_split_seed: int = 42,
    val_reconstruction_metric: str = "l1",
    merge_val_export: bool = True,
    device: str | None = None,
) -> tuple[AdvNOTrainConfig, Path, str]:
    window = get_window_spec(export_key)
    export_component = primary_export_component(export_key)
    channel_names = window["channel_names_train"]
    iD = window["iD"]
    dt = dt_hours(dt_days)

    if val_split_mode == "export":
        merge_val_export = False

    models_dir = dt_dir / "Models" / "adv_no"
    run_tag = weight_decay_run_tag(RUN_TAG, weight_decay)
    run_name_value = run_name(export_key, dt_hours=dt, run_tag=run_tag)
    run_dir_path = run_dir(models_dir, export_key, dt_hours=dt, run_tag=run_tag)

    data_prefix = train_prefix(dt_dir, export_key, component=export_component, dt_hours=dt)
    val_data_prefix = val_prefix(dt_dir, export_key, component=export_component, dt_hours=dt)

    if device is None:
        device = device_string(dt_days)

    config = AdvNOTrainConfig(
        data_prefix=data_prefix,
        run_name=run_name_value,
        dt_hours=dt,
        epochs=epochs,
        batch_size=4,
        lr=2e-4,
        optimizer=optimizer,
        weight_decay=weight_decay,
        d_lr=1e-4,
        d_lr_follow_g=True,
        d_lr_ratio=0.5,
        use_cosine_lr=True,
        iD=iD,
        conditioning_steps=CONDITIONING_STEPS,
        device=device,
        loss_mode="adv_no",
        loss_metric="components",
        loss_spatial_mode="full",
        channel_names=channel_names,
        val_split=val_split,
        val_split_mode=val_split_mode if val_split_mode != "export" else "random",
        val_split_seed=val_split_seed,
        val_reconstruction_metric=val_reconstruction_metric,
        val_data_prefix=None if merge_val_export else val_data_prefix,
        merge_val_export_prefix=val_data_prefix if merge_val_export else None,
        max_samples=max_samples,
        checkpoint_subdir=str(models_dir),
        isolate_runs=True,
        warmup_epochs=30,
        perceptual_warmup_epochs=10,
        beta=0.1,
        lambda_perc=0.1,
        perceptual_backend="vgg",
    )
    return config, run_dir_path, export_component


def prepare_train_job(
    *,
    dt_dir: Path,
    export_key: str,
    dt_days: int,
    **kwargs: Any,
) -> dict[str, Any]:
    config, run_dir_path, export_component = build_train_config(
        dt_dir=dt_dir,
        export_key=export_key,
        dt_days=dt_days,
        **kwargs,
    )
    run_dir_path.mkdir(parents=True, exist_ok=True)
    config_path = run_dir_path / f"train_config_{config.run_name}.json"
    config_path.write_text(json.dumps(asdict(config), indent=2, default=str))

    stats = collect_dataset_stats(
        dt_dir, export_key, dt_hours=config.dt_hours, component=export_component
    )
    save_dataset_stats(run_dir_path, stats)
    update_comparison_index(
        dt_dir / "Models" / "adv_no",
        run_name_value=config.run_name,
        export_key=export_key,
        run_dir_path=run_dir_path,
        dataset_stats=stats,
    )
    return {
        "dt_days": dt_days,
        "device_id": cuda_visible_devices(dt_days),
        "run_name": config.run_name,
        "run_dir": str(run_dir_path),
        "config_path": str(config_path),
        "device": config.device,
    }


def train_one_dt(
    *,
    dt_dir: Path,
    root: Path,
    export_key: str,
    dt_days: int,
    log_dir: Path,
    detached: bool = True,
    resume: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    job = prepare_train_job(dt_dir=dt_dir, export_key=export_key, dt_days=dt_days, **kwargs)
    config_path = Path(job["config_path"])
    device_id = job["device_id"]
    log_path = log_dir / f"train_{export_key}_dt{dt_days}d_gpu{device_id}.log"

    if detached:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": device_id}
        cmd = [
            sys.executable,
            str(root / "scripts/dynamic_dt_worker.py"),
            "train",
            "--config",
            str(config_path),
        ]
        if resume:
            cmd.append("--resume")
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
        return job

    os.environ["CUDA_VISIBLE_DEVICES"] = device_id
    config = AdvNOTrainConfig.from_dict(json.loads(config_path.read_text()))
    config.device = "cuda:0"
    run_adv_no_training(config, resume=resume)
    job["log_path"] = str(log_path)
    return job
