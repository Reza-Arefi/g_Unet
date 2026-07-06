"""Run naming and paths for dynamic_dt train / rollout notebooks."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

MW_DIR = Path(__file__).resolve().parents[1]
DT_DIR = Path(__file__).resolve().parent
if str(MW_DIR) not in sys.path:
    sys.path.insert(0, str(MW_DIR))
if str(DT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(DT_DIR.parent))

from dynamic_dt.window_paths import data_root, split_manifest_path, test_prefix, train_prefix, val_prefix

# Reuse helpers from parent Magic_window registry
from run_registry import (  # noqa: E402
    RUN_TAG,
    collect_dataset_stats as _collect_stats_parent,
    resolve_model_checkpoint,
    rollout_eval_name,
    save_dataset_stats,
    save_rollout_results,
    update_comparison_index,
    weight_decay_run_tag,
)


def run_name(export_key: str, *, dt_hours: int = 360, run_tag: str = RUN_TAG) -> str:
    return f"adv_no_uv_13_22_{export_key}_dt{dt_hours}_{run_tag}"


def run_dir(
    models_dir: Path | str,
    export_key: str,
    *,
    dt_hours: int = 360,
    run_tag: str = RUN_TAG,
) -> Path:
    return Path(models_dir) / run_name(export_key, dt_hours=dt_hours, run_tag=run_tag)


def test_rollout_dir(
    dt_dir: Path | str,
    export_key: str,
    *,
    dt_hours: int = 360,
    run_tag: str = RUN_TAG,
) -> Path:
    return Path(dt_dir) / "Test&Rollout" / run_name(export_key, dt_hours=dt_hours, run_tag=run_tag)


def test_rollout_dir_with_model(
    dt_dir: Path | str,
    export_key: str,
    model_run_name: str,
) -> Path:
    return Path(dt_dir) / "Test&Rollout" / rollout_eval_name(model_run_name, export_key)


def collect_dataset_stats(
    dt_dir: Path | str,
    export_key: str,
    *,
    dt_hours: int = 360,
    component: str = "uv",
) -> dict[str, Any]:
    """Dataset stats using dynamic_dt data paths."""
    data_dir = data_root(dt_dir)
    train_p = train_prefix(dt_dir, export_key, component=component, dt_hours=dt_hours)
    val_p = val_prefix(dt_dir, export_key, component=component, dt_hours=dt_hours)
    test_p = test_prefix(dt_dir, export_key, component=component, dt_hours=dt_hours)

    def _split_stats(prefix: str) -> dict[str, Any]:
        sims_path = Path(f"{prefix}_sims.npy")
        mask_path = Path(f"{prefix}_mask.npy")
        window_path = Path(f"{prefix}_window.npy")
        if not sims_path.exists():
            raise FileNotFoundError(f"Missing export: {sims_path}")
        sims = np.load(sims_path, mmap_mode="r")
        mask = np.load(mask_path, mmap_mode="r")
        window = np.load(window_path) if window_path.exists() else None
        return {
            "prefix": prefix,
            "n_sequences": int(len(sims)),
            "sims_shape": list(sims.shape),
            "land_fraction": float(mask.mean()),
            "window": window.tolist() if window is not None else None,
        }

    return {
        "export_key": export_key,
        "dt_hours": dt_hours,
        "component": component,
        "train": _split_stats(train_p),
        "val": _split_stats(val_p),
        "test": _split_stats(test_p),
        "split_manifest": str(
            split_manifest_path(dt_dir, export_key, dt_hours=dt_hours)
        ),
        "collected_utc": datetime.now(timezone.utc).isoformat(),
    }
