"""Per-window export paths under ``{data_dir}/{export_key}/`` (Shalow notebooks use ``Shalow/data``)."""

from __future__ import annotations

import json
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


def pool_prefix(
    data_dir: Path | str,
    export_key: str,
    *,
    component: str = "uv",
    dt: int = 360,
) -> str:
    """Single 2013–2022 export pool (train/val split happens at training time)."""
    tag = _dt_suffix(dt)
    return str(window_data_dir(data_dir, export_key) / f"sample_13_22_{component}{tag}")


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


def export_manifest_path(data_dir: Path | str, export_key: str) -> Path:
    """Metadata for a single pool export (no export-time train/val split)."""
    return window_data_dir(data_dir, export_key) / "sample_13_22_export.json"


def _component_from_pool_basename(name: str, *, dt: int) -> str | None:
    """Parse ``uvsshbathy`` from ``sample_13_22_uvsshbathy_dt360``."""
    tag = _dt_suffix(dt)
    prefix = "sample_13_22_"
    if not name.startswith(prefix):
        return None
    rest = name[len(prefix) :]
    if tag and rest.endswith(tag):
        rest = rest[: -len(tag)]
    return rest or None


def resolve_export_component(
    data_dir: Path | str,
    export_key: str,
    *,
    dt: int = 360,
    component: str | None = None,
) -> str:
    """Infer export component tag (``uv``, ``uvsshbathy``, …) for path helpers."""
    if component is not None:
        return component

    export_path = export_manifest_path(data_dir, export_key)
    if export_path.exists():
        manifest = json.loads(export_path.read_text())
        components = manifest.get("export_components")
        if components:
            return str(components[0])
        pool_base = manifest.get("pool_data_prefix_base")
        if pool_base:
            parsed = _component_from_pool_basename(Path(pool_base).name, dt=dt)
            if parsed:
                return parsed

    wd = window_data_dir(data_dir, export_key)
    tag = _dt_suffix(dt)
    for path in sorted(wd.glob(f"sample_13_22_*{tag}_sims.npy")):
        parsed = _component_from_pool_basename(path.name.replace("_sims.npy", ""), dt=dt)
        if parsed:
            return parsed

    return "uv"
