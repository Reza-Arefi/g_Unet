"""Shared run naming, dataset stats, and cross-dataset comparison index for adv-NO notebooks 04/05."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from window_paths import split_manifest_path, test_prefix, train_prefix, val_prefix

# Fixed model / training settings — change only EXPORT_KEY in notebooks 04 and 05.
RUN_TAG = "vgg_warm10_adv60"
GAUSS_RUN_TAG = "gauss_vgg_warm10_adv60"  # notebook 04/05 gaussian spatial loss variant
DEFAULT_DT = 360
TEST_ROLLOUT_DIR_NAME = "Test&Rollout"


def run_name(export_key: str, *, dt: int = DEFAULT_DT, run_tag: str = RUN_TAG) -> str:
    return f"adv_no_uv_13_22_{export_key}_dt{dt}_{run_tag}"


def weight_decay_run_tag(base_tag: str, weight_decay: float) -> str:
    """Suffix ``base_tag`` for weight-decay ablations, e.g. ``..._wd1e-4``."""
    wd = float(weight_decay)
    if wd == 0.0:
        return f"{base_tag}_wd0"
    suffix = f"{wd:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    return f"{base_tag}_wd{suffix}"


def weight_decay_ablation_run_tag(
    base_tag: str,
    weight_decay: float,
    *,
    epochs: int | None = None,
) -> str:
    """Run tag for one ablation arm, optionally suffixed with ``_ep{N}``."""
    tag = weight_decay_run_tag(base_tag, weight_decay)
    if epochs is not None:
        tag = f"{tag}_ep{int(epochs)}"
    return tag


def weight_decay_ablation_tags(
    base_tag: str,
    weight_decays: tuple[float, ...] = (0.0, 1e-4, 3e-4),
    *,
    epochs: int | None = None,
) -> list[tuple[float, str]]:
    """Return (weight_decay, run_tag) pairs for a standard ablation grid."""
    return [
        (wd, weight_decay_ablation_run_tag(base_tag, wd, epochs=epochs))
        for wd in weight_decays
    ]


def run_dir(
    models_dir: Path | str,
    export_key: str,
    *,
    dt: int = DEFAULT_DT,
    run_tag: str = RUN_TAG,
) -> Path:
    return Path(models_dir) / run_name(export_key, dt=dt, run_tag=run_tag)


def test_rollout_dir(
    mw_dir: Path | str,
    export_key: str,
    *,
    dt: int = DEFAULT_DT,
    run_tag: str = RUN_TAG,
) -> Path:
    """Notebook 05 rollout outputs: Magic_window/Test&Rollout/{run_name}/."""
    return Path(mw_dir) / TEST_ROLLOUT_DIR_NAME / run_name(export_key, dt=dt, run_tag=run_tag)


def rollout_eval_name(model_run_name: str, export_key: str) -> str:
    """Label for evaluating one fixed checkpoint on an export_key test set."""
    return f"{model_run_name}__on__{export_key}"


def test_rollout_dir_with_model(
    mw_dir: Path | str,
    export_key: str,
    model_run_name: str,
) -> Path:
    """Rollout dir when a shared model is evaluated on export_key data."""
    return Path(mw_dir) / TEST_ROLLOUT_DIR_NAME / rollout_eval_name(model_run_name, export_key)


def resolve_model_checkpoint(
    run_dir: Path | str,
    *,
    expected_run_name: str | None = None,
) -> tuple[dict[str, Any], Path, Path, int | None, float | None, str]:
    """Load train config + best_model.pth + norm_stats from an explicit run directory."""
    import torch

    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Model run directory not found: {run_dir}")

    cfg_paths = sorted(run_dir.glob("train_config_*.json"))
    if not cfg_paths:
        raise FileNotFoundError(
            f"Missing train_config_*.json under {run_dir}\n"
            "Point MODEL_RUN_DIR at a trained adv-NO checkpoint folder."
        )
    cfg_path = cfg_paths[0]
    if expected_run_name:
        preferred = run_dir / f"train_config_{expected_run_name}.json"
        if preferred.exists():
            cfg_path = preferred
    elif len(cfg_paths) > 1:
        raise ValueError(
            f"Multiple train configs in {run_dir}; set MODEL_RUN_NAME explicitly."
        )

    cfg = json.loads(cfg_path.read_text())
    run_name_value = str(cfg.get("run_name") or cfg_path.stem.removeprefix("train_config_"))
    model_path = Path(cfg["best_model_path"]) if cfg.get("best_model_path") else run_dir / "best_model.pth"
    if not str(model_path):
        model_path = run_dir / "best_model.pth"
    model_path = Path(model_path)
    stats_path = run_dir / "norm_stats.pt"
    for label, path in (("best_model.pth", model_path), ("norm_stats.pt", stats_path)):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {label}: {path}\n"
                f"Expected checkpoint artifacts under {run_dir}."
            )

    best_epoch = best_val = None
    hist_path = run_dir / "history" / f"training_history_{run_name_value}.pt"
    if hist_path.exists():
        hist = torch.load(hist_path, map_location="cpu", weights_only=False)
        best_epoch = int(hist.get("best_epoch") or 0)
        best_val = float(hist.get("best_val_loss") or float("nan"))

    return cfg, model_path, stats_path, best_epoch, best_val, run_name_value


def _split_stats(prefix: str) -> dict[str, Any]:
    sims_path = Path(f"{prefix}_sims.npy")
    mask_path = Path(f"{prefix}_mask.npy")
    window_path = Path(f"{prefix}_window.npy")
    if not sims_path.exists():
        raise FileNotFoundError(f"Missing export: {sims_path}")

    sims = np.load(sims_path, mmap_mode="r")
    mask = np.load(mask_path, mmap_mode="r")
    window = np.load(window_path) if window_path.exists() else None
    land_frac = float(mask.mean())
    static_mask = len(mask) <= 1 or np.all(mask == mask[0:1])

    ocean = mask == 0
    if static_mask and ocean.ndim >= 2:
        ocean = ocean[0]
    uv_stats: dict[str, float] = {}
    if sims.size and ocean.any():
        # subsample sequences for speed on large exports
        n = len(sims)
        idx = np.linspace(0, n - 1, num=min(n, 64), dtype=int)
        u_vals = sims[idx, :, 0][..., ocean].astype(np.float64)
        v_vals = sims[idx, :, 1][..., ocean].astype(np.float64)
        speed = np.sqrt(u_vals ** 2 + v_vals ** 2)
        uv_stats = {
            "u_mean": float(u_vals.mean()),
            "u_std": float(u_vals.std()),
            "v_mean": float(v_vals.mean()),
            "v_std": float(v_vals.std()),
            "speed_mean": float(speed.mean()),
            "speed_std": float(speed.std()),
        }

    return {
        "prefix": prefix,
        "n_sequences": int(len(sims)),
        "sims_shape": list(sims.shape),
        "land_fraction": land_frac,
        "static_mask": bool(static_mask),
        "window": window.tolist() if window is not None else None,
        "uv_ocean_stats": uv_stats,
    }


def collect_dataset_stats(
    data_dir: Path | str,
    export_key: str,
    *,
    dt: int = DEFAULT_DT,
    component: str = "uv",
) -> dict[str, Any]:
    data_dir = Path(data_dir)
    train_p = train_prefix(data_dir, export_key, component=component, dt=dt)
    val_p = val_prefix(data_dir, export_key, component=component, dt=dt)
    test_p = test_prefix(data_dir, export_key, component=component, dt=dt)
    split_path = split_manifest_path(data_dir, export_key)

    stats: dict[str, Any] = {
        "export_key": export_key,
        "dt_hours": dt,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir / export_key),
        "split_manifest": str(split_path) if split_path.exists() else None,
        "train": _split_stats(train_p),
        "val": _split_stats(val_p),
        "test": _split_stats(test_p),
    }
    if split_path.exists():
        stats["split_manifest_json"] = json.loads(split_path.read_text())
    return stats


def save_json(path: Path | str, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def save_dataset_stats(run_dir_path: Path | str, stats: dict[str, Any]) -> Path:
    return save_json(Path(run_dir_path) / "dataset_stats.json", stats)


def save_rollout_results(output_dir: Path | str, results: dict[str, Any]) -> Path:
    return save_json(Path(output_dir) / "rollout_results.json", results)


def _summary_from_rollout(results: dict[str, Any]) -> dict[str, Any]:
    def _skill(rows: list[dict], horizon: int) -> float | None:
        for row in rows:
            if int(row.get("horizon_days", -1)) == horizon:
                return row.get("skill_score")
        return None

    center = results.get("metrics_center") or []
    full = results.get("metrics_full") or []
    return {
        "n_test": results.get("n_test"),
        "best_val_loss": results.get("best_val_loss"),
        "best_epoch": results.get("best_epoch"),
        "skill_center_15d": _skill(center, 15),
        "skill_center_45d": _skill(center, 45),
        "skill_full_15d": _skill(full, 15),
        "skill_full_45d": _skill(full, 45),
        "model_rmse_full_45d": next(
            (r.get("model_rmse") for r in full if int(r.get("horizon_days", -1)) == 45),
            None,
        ),
    }


def update_comparison_index(
    models_dir: Path | str,
    *,
    run_name_value: str,
    export_key: str,
    run_dir_path: Path | str,
    dataset_stats: dict[str, Any] | None = None,
    rollout_results: dict[str, Any] | None = None,
    training_summary: dict[str, Any] | None = None,
    rollout_output_dir: Path | str | None = None,
) -> Path:
    models_dir = Path(models_dir)
    index_path = models_dir / "comparison_index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
    else:
        index = {"runs": [], "updated_at": None}

    run_dir_path = Path(run_dir_path)
    rollout_dir = Path(rollout_output_dir) if rollout_output_dir else run_dir_path
    entry: dict[str, Any] = {
        "run_name": run_name_value,
        "export_key": export_key,
        "run_dir": str(run_dir_path),
        "rollout_output_dir": str(rollout_dir),
        "dataset_stats_path": str(run_dir_path / "dataset_stats.json"),
        "rollout_results_path": str(rollout_dir / "rollout_results.json"),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if rollout_results and rollout_results.get("model_run_name"):
        entry["model_run_name"] = rollout_results["model_run_name"]
        entry["model_run_dir"] = rollout_results.get("model_run_dir")
    if dataset_stats:
        entry["dataset"] = {
            "train_n": dataset_stats["train"]["n_sequences"],
            "val_n": dataset_stats["val"]["n_sequences"],
            "test_n": dataset_stats["test"]["n_sequences"],
            "land_fraction": dataset_stats["train"]["land_fraction"],
            "window": dataset_stats["train"]["window"],
        }
    if training_summary:
        entry["training"] = training_summary
    if rollout_results:
        entry["rollout"] = _summary_from_rollout(rollout_results)

    runs = [r for r in index.get("runs", []) if r.get("run_name") != run_name_value]
    runs.append(entry)
    index["runs"] = sorted(runs, key=lambda r: r.get("export_key", ""))
    index["updated_at"] = datetime.now(timezone.utc).isoformat()
    return save_json(index_path, index)
