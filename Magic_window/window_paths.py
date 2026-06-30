"""Per-window export paths under Magic_window/data/{export_key}/ (flat, no train/val/test subdirs)."""

from __future__ import annotations

from pathlib import Path


def window_data_dir(data_dir: Path | str, export_key: str) -> Path:
    d = Path(data_dir) / export_key
    d.mkdir(parents=True, exist_ok=True)
    return d


def _dt_suffix(dt: int) -> str:
    if dt == 360:
        return "_dt360"
    if dt == 96:
        return "_dt96"
    return ""


def train_prefix(
    data_dir: Path | str,
    export_key: str,
    *,
    component: str = "uv",
    dt: int = 360,
) -> str:
    tag = _dt_suffix(dt)
    return str(window_data_dir(data_dir, export_key) / f"sample_13_22_train_{component}{tag}")


def val_prefix(
    data_dir: Path | str,
    export_key: str,
    *,
    component: str = "uv",
    dt: int = 360,
) -> str:
    tag = _dt_suffix(dt)
    return str(window_data_dir(data_dir, export_key) / f"sample_13_22_val_{component}{tag}")


def test_prefix(
    data_dir: Path | str,
    export_key: str,
    *,
    component: str = "uv",
    dt: int = 360,
) -> str:
    tag = _dt_suffix(dt)
    return str(window_data_dir(data_dir, export_key) / f"sample_23_{component}{tag}")


def split_manifest_path(data_dir: Path | str, export_key: str) -> Path:
    return window_data_dir(data_dir, export_key) / "sample_13_22_split.json"
