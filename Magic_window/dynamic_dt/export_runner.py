"""Run dynamic_dt exports for one catalog window."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from dynamic_dt.dt_config import dt_hours
from dynamic_dt.export_setup import (
    candidate_indices_for_hour,
    component_export_settings,
    split_train_val_zarr_indices,
)
from dynamic_dt.sequence_config import CONDITIONING_STEPS, FORECAST_STEPS
from dynamic_dt.window_catalog import get_window_spec
from dynamic_dt.window_paths import split_manifest_path, test_prefix, train_prefix, val_prefix
from src.data.dataloaders.guiana_margem_tensor import GuianaDataloader
from src.data.useful_transforms import FixedCrop, RotatedFixedCrop
from window_selection_setup import resolve_window


def build_window_crop(window_spec: dict) -> FixedCrop | RotatedFixedCrop:
    h, w = window_spec["crop_height"], window_spec["crop_width"]
    top, left = window_spec["window_top"], window_spec["window_left"]
    angle = float(window_spec["window_angle"])
    if angle == 0.0:
        return FixedCrop(size=(h, w), top=top, left=left)
    return RotatedFixedCrop(size=(h, w), top=top, left=left, angle_deg=angle)


def export_one_window(
    *,
    dt_dir: Path,
    root: Path,
    export_key: str,
    dt_days: int,
    zarr_path: str,
    log_dir: Path,
    val_split: float = 0.3,
    val_split_mode: str = "random",
    val_split_seed: int = 42,
    sample_seed: int = 42,
    n_samples_to_gen: int = 3600,
    conditioning_steps: int = CONDITIONING_STEPS,
    forecast_steps: int = FORECAST_STEPS,
    hour_of_day: int = 12,
    start_time: str = "2013-01-01",
    end_time: str = "2022-12-31",
    test_start: str = "2023-01-01",
    test_end: str = "2023-12-31",
    max_mask_frac: float = 0.7,
    n_jobs: int = 30,
    chunk_size: int = 100,
    detached: bool = True,
    resume: bool = False,
    reuse_split: bool = False,
    device_slot: int | None = None,
) -> dict[str, Any]:
    from src.export.fix_window_export import ExportConfig, run_export

    entry = get_window_spec(export_key)
    window_spec = resolve_window(entry["window"])
    export_components = entry["export_components"]
    dt = dt_hours(dt_days)
    dt_dir = Path(dt_dir).resolve()
    split_path = split_manifest_path(dt_dir, export_key, dt_hours=dt)

    crop = build_window_crop(window_spec)
    base_kwargs = dict(
        path=zarr_path,
        dt=dt,
        conditioning_steps=conditioning_steps,
        forecast_steps=forecast_steps,
    )

    dataset = GuianaDataloader(
        **base_kwargs,
        start_time=start_time,
        end_time=end_time,
        transform=crop,
        device="cpu",
    )
    candidate_idx = candidate_indices_for_hour(dataset, hour_of_day)

    if reuse_split and split_path.is_file():
        prior = json.loads(split_path.read_text())
        same = (
            prior.get("export_key") == export_key
            and prior.get("dt_hours") == dt
            and prior.get("split_mode") == val_split_mode
            and prior.get("val_split") == val_split
            and prior.get("split_seed") == val_split_seed
            and prior.get("sample_seed") == sample_seed
        )
        if same and prior.get("train_zarr_indices"):
            train_idx = np.asarray(prior["train_zarr_indices"], dtype=np.int64)
            val_idx = np.asarray(prior["val_zarr_indices"], dtype=np.int64)
        else:
            train_idx, val_idx = split_train_val_zarr_indices(
                dataset, candidate_idx, n_samples_to_gen, val_split,
                split_mode=val_split_mode, sample_seed=sample_seed, split_seed=val_split_seed,
            )
    else:
        train_idx, val_idx = split_train_val_zarr_indices(
            dataset, candidate_idx, n_samples_to_gen, val_split,
            split_mode=val_split_mode, sample_seed=sample_seed, split_seed=val_split_seed,
        )

    manifest = {
        "export_key": export_key,
        "label": entry["label"],
        "dt_days": dt_days,
        "dt_hours": dt,
        "split_mode": val_split_mode,
        "val_split": val_split,
        "sample_seed": sample_seed,
        "split_seed": val_split_seed,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "train_zarr_indices": train_idx.tolist(),
        "val_zarr_indices": val_idx.tolist(),
        "window_top": window_spec["window_top"],
        "window_left": window_spec["window_left"],
        "window_angle": window_spec["window_angle"],
        "export_components": list(export_components),
        "conditioning_steps": conditioning_steps,
        "forecast_steps": forecast_steps,
    }
    split_path.write_text(json.dumps(manifest, indent=2))

    export_env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    if device_slot is not None:
        export_env["DYNAMIC_DT_DEVICE_SLOT"] = str(device_slot)
    jobs: list[str] = []
    pids: list[int] = []
    prefixes: list[str] = []
    slot_tag = f"_gpu{device_slot}" if device_slot is not None else ""

    def _export_finished(prefix: str) -> bool:
        state_path = Path(f"{prefix}_export_state.json")
        if not state_path.is_file():
            return False
        return json.loads(state_path.read_text()).get("status") == "finished"

    def _run(prefix: str, ch_names, tgt_idx, *, st, en, indices):
        prefix = str(Path(prefix).resolve())
        if resume and _export_finished(prefix):
            print(f"  skip (finished) → {prefix}")
            prefixes.append(prefix)
            return

        cfg = ExportConfig(
            zarr_path=zarr_path,
            dt=dt,
            conditioning_steps=conditioning_steps,
            forecast_steps=forecast_steps,
            crop_height=window_spec["crop_height"],
            crop_width=window_spec["crop_width"],
            window_top=window_spec["window_top"],
            window_left=window_spec["window_left"],
            window_angle=window_spec["window_angle"],
            seed=sample_seed,
            n_jobs=n_jobs,
            max_mask_frac=max_mask_frac,
            hour_of_day=hour_of_day,
            start_time=st,
            end_time=en,
            output_prefix=prefix,
            n_samples_to_gen=len(indices),
            chunk_size=chunk_size,
            channel_names=ch_names,
            target_channel_indices=tgt_idx,
            fixed_dataset_indices=tuple(int(x) for x in indices),
        )
        d = asdict(cfg)
        d["export_key"] = export_key
        d["dt_days"] = dt_days
        d["rotation_pivot"] = window_spec.get("rotation_pivot", "top_left")
        Path(f"{prefix}_export_config.json").write_text(json.dumps(d, indent=2, default=str))

        if detached:
            log_path = log_dir / f"export_{export_key}{slot_tag}_{Path(prefix).name}.log"
            cmd = [
                sys.executable,
                str(root / "scripts/export_fix_window_data.py"),
                "--output-prefix",
                prefix,
            ]
            if not resume:
                cmd.append("--no-resume")
            with open(log_path, "a") as log_file:
                proc = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    cwd=root,
                    env=export_env,
                )
            print(f"  Started PID {proc.pid} → {prefix}")
            jobs.append(str(log_path))
            pids.append(proc.pid)
            prefixes.append(prefix)
        else:
            run_export(cfg, resume=resume)
            prefixes.append(prefix)

    for component in export_components:
        tr_base = train_prefix(dt_dir, export_key, component=component, dt_hours=dt)
        va_base = val_prefix(dt_dir, export_key, component=component, dt_hours=dt)
        te_base = test_prefix(dt_dir, export_key, component=component, dt_hours=dt)
        tr_p, ch, tgt = component_export_settings(dt_dir, export_key, component, dt, base=tr_base)
        va_p, _, _ = component_export_settings(dt_dir, export_key, component, dt, base=va_base)
        te_p, _, _ = component_export_settings(dt_dir, export_key, component, dt, base=te_base)

        print(f"  train {component} n={len(train_idx)}")
        _run(tr_p, ch, tgt, st=start_time, en=end_time, indices=train_idx)
        print(f"  val {component} n={len(val_idx)}")
        _run(va_p, ch, tgt, st=start_time, en=end_time, indices=val_idx)

        test_ds = GuianaDataloader(
            **base_kwargs,
            start_time=test_start,
            end_time=test_end,
            transform=crop,
            device="cpu",
        )
        test_idx = tuple(int(x) for x in candidate_indices_for_hour(test_ds, hour_of_day))
        print(f"  test 2023 {component} n={len(test_idx)}")
        _run(te_p, ch, tgt, st=test_start, en=test_end, indices=test_idx)

    return {
        "export_key": export_key,
        "label": entry["label"],
        "dt_days": dt_days,
        "dt_hours": dt,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "split_manifest": str(split_path),
        "log_files": jobs,
        "pids": pids,
        "output_prefixes": prefixes,
        "device_slot": device_slot,
    }
