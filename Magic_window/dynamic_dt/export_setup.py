"""Train/val index sampling for dynamic_dt exports."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.train.fix_window_train import split_train_val_indices


def candidate_indices_for_hour(dataset, hour_of_day):
    if hour_of_day is None:
        return np.arange(len(dataset), dtype=np.int64)
    if isinstance(hour_of_day, (int, np.integer)):
        allowed = {int(hour_of_day)}
    else:
        allowed = {int(h) for h in hour_of_day}
    invalid = allowed - set(range(24))
    if invalid:
        raise ValueError(f"Invalid hour(s) {sorted(invalid)}; expected integers 0-23")
    return np.array(
        [i for i, t0 in enumerate(dataset.starts) if pd.Timestamp(t0).hour in allowed],
        dtype=np.int64,
    )


def _sort_indices_by_start_time(dataset, indices: np.ndarray) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64)
    starts = np.array([pd.Timestamp(dataset.starts[int(i)]).value for i in idx])
    return idx[np.argsort(starts, kind="stable")]


def split_train_val_zarr_indices(
    dataset,
    candidate_idx: np.ndarray,
    n_samples: int,
    val_split: float,
    *,
    split_mode: str = "random",
    sample_seed: int = 42,
    split_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample ``n_samples`` Zarr sequence indices and split into train / val.

    split_mode
    ----------
    ``random`` : random subsample, then random 70/30 (val_split) split
    ``chronological`` : chronologically sorted subsample (first n), last val_split → val
    """
    pool = np.asarray(candidate_idx, dtype=np.int64)
    n = min(int(n_samples), len(pool))
    if n <= 0:
        raise ValueError("No sequences to sample from candidate pool")

    if split_mode == "chronological":
        ordered = _sort_indices_by_start_time(dataset, pool)
        selected = ordered[:n]
        n_val = max(1, int(n * val_split)) if n > 1 else 0
        if n_val >= n:
            n_val = max(1, n // 5)
        n_train = n - n_val
        return selected[:n_train], selected[n_train:]

    if split_mode == "random":
        rng = np.random.default_rng(sample_seed)
        selected = rng.choice(pool, size=n, replace=False)
        train_rows, val_rows = split_train_val_indices(
            n, val_split, mode="random", seed=split_seed
        )
        return selected[train_rows], selected[val_rows]

    raise ValueError(f"split_mode must be 'random' or 'chronological', got {split_mode!r}")


def component_export_settings(
    dynamic_dt_dir,
    export_key: str,
    component: str,
    dt_hours: int,
    *,
    base: str | None = None,
):
    from dynamic_dt.window_paths import test_prefix, train_prefix, val_prefix

    if component == "uv":
        base = base or train_prefix(dynamic_dt_dir, export_key, component="uv", dt_hours=dt_hours)
        return base, ("u", "v"), (0, 1)
    if component == "uvbathy":
        base = base or train_prefix(
            dynamic_dt_dir, export_key, component="uvbathy", dt_hours=dt_hours
        )
        return base, ("u", "v", "bathymetry"), (0, 1)
    if component == "bathymetry":
        base = base or train_prefix(
            dynamic_dt_dir, export_key, component="bathymetry", dt_hours=dt_hours
        )
        return base, ("bathymetry",), ()
    if component == "u":
        base = base or train_prefix(dynamic_dt_dir, export_key, component="u", dt_hours=dt_hours)
        return base, ("u",), (0,)
    if component == "v":
        base = base or train_prefix(dynamic_dt_dir, export_key, component="v", dt_hours=dt_hours)
        return base, ("v",), (0,)
    raise ValueError(f"Unknown component {component!r}")
