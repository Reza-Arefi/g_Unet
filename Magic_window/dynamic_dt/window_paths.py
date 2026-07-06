"""Export paths under Magic_window/dynamic_dt/data/{export_key}/."""

from __future__ import annotations

from pathlib import Path

from dynamic_dt.dt_config import dt_suffix


def data_root(dynamic_dt_dir: Path | str) -> Path:
    return Path(dynamic_dt_dir) / "data"


def window_data_dir(dynamic_dt_dir: Path | str, export_key: str) -> Path:
    d = data_root(dynamic_dt_dir) / export_key
    d.mkdir(parents=True, exist_ok=True)
    return d


def train_prefix(
    dynamic_dt_dir: Path | str,
    export_key: str,
    *,
    component: str = "uv",
    dt_hours: int = 360,
) -> str:
    tag = dt_suffix(dt_hours)
    return str(
        window_data_dir(dynamic_dt_dir, export_key) / f"sample_13_22_train_{component}{tag}"
    )


def val_prefix(
    dynamic_dt_dir: Path | str,
    export_key: str,
    *,
    component: str = "uv",
    dt_hours: int = 360,
) -> str:
    tag = dt_suffix(dt_hours)
    return str(
        window_data_dir(dynamic_dt_dir, export_key) / f"sample_13_22_val_{component}{tag}"
    )


def test_prefix(
    dynamic_dt_dir: Path | str,
    export_key: str,
    *,
    component: str = "uv",
    dt_hours: int = 360,
) -> str:
    tag = dt_suffix(dt_hours)
    return str(
        window_data_dir(dynamic_dt_dir, export_key) / f"sample_23_{component}{tag}"
    )


def split_manifest_path(
    dynamic_dt_dir: Path | str,
    export_key: str,
    *,
    dt_hours: int = 360,
) -> Path:
    tag = dt_suffix(dt_hours)
    return window_data_dir(dynamic_dt_dir, export_key) / f"sample_13_22_split{tag}.json"
