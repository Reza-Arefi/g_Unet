"""Canonical 12-window catalog from ``Magic_window/window_selection.ipynb``."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

MW_DIR = Path(__file__).resolve().parent
CATALOG_ID = "magic_window_12"

# Display labels for windows 1..12 (matches window_selection.ipynb section titles).
WINDOW_SELECTION_LABELS: tuple[str, ...] = (
    "1 reference",
    "2 coast slide n2",
    "3 coast slide n1",
    "4 coast slide s1",
    "5 coast slide s2",
    "6 bathy mid",
    "7 bathy east",
    "8 bathy northeast",
    "9 rot 0°",
    "10 rot 23°",
    "11 rot 45°",
    "12 rot 68°",
)

# Mirrors window_selection.ipynb cells 2–13 (slot order = training window index).
WINDOW_CONFIGS: list[dict[str, Any]] = [
    {
        "export_key": "magicwin_reference",
        "crop_height": 128,
        "crop_width": 128,
        "window_top": None,
        "window_left": None,
        "window_angle": 0.0,
        "rotation_pivot": "top_left",
    },
    {
        "export_key": "magicwin_coast_slide_n2",
        "crop_height": 128,
        "crop_width": 128,
        "window_top": 300,
        "window_left": 1,
        "window_angle": 0.0,
        "rotation_pivot": "top_left",
    },
    {
        "export_key": "magicwin_coast_slide_n1",
        "crop_height": 128,
        "crop_width": 128,
        "window_top": 200,
        "window_left": 200,
        "window_angle": 0.0,
        "rotation_pivot": "top_left",
    },
    {
        "export_key": "magicwin_coast_slide_s1",
        "crop_height": 128,
        "crop_width": 128,
        "window_top": 100,
        "window_left": 300,
        "window_angle": 0.0,
        "rotation_pivot": "top_left",
    },
    {
        "export_key": "magicwin_coast_slide_s2",
        "crop_height": 128,
        "crop_width": 128,
        "window_top": 20,
        "window_left": 410,
        "window_angle": 0.0,
        "rotation_pivot": "top_left",
    },
    {
        "export_key": "magicwin_bathy_mid",
        "crop_height": 128,
        "crop_width": 128,
        "window_top": 150,
        "window_left": 400,
        "window_angle": 0.0,
        "rotation_pivot": "top_left",
    },
    {
        "export_key": "magicwin_bathy_east",
        "crop_height": 128,
        "crop_width": 128,
        "window_top": 300,
        "window_left": 400,
        "window_angle": 0.0,
        "rotation_pivot": "top_left",
    },
    {
        "export_key": "magicwin_bathy_northeast",
        "crop_height": 128,
        "crop_width": 128,
        "window_top": 300,
        "window_left": 250,
        "window_angle": 0.0,
        "rotation_pivot": "top_left",
    },
    {
        "export_key": "magicwin_rot_0deg",
        "crop_height": 128,
        "crop_width": 128,
        "window_top": 330,
        "window_left": 20,
        "window_angle": -35.0,
        "rotation_pivot": "top_left",
    },
    {
        "export_key": "magicwin_rot_23deg",
        "crop_height": 128,
        "crop_width": 128,
        "window_top": 100,
        "window_left": 400,
        "window_angle": 23.0,
        "rotation_pivot": "top_left",
    },
    {
        "export_key": "magicwin_rot_45deg",
        "crop_height": 128,
        "crop_width": 128,
        "window_top": 200,
        "window_left": 300,
        "window_angle": 45.0,
        "rotation_pivot": "top_left",
    },
    {
        "export_key": "magicwin_rot_68deg",
        "crop_height": 128,
        "crop_width": 128,
        "window_top": 250,
        "window_left": 200,
        "window_angle": 68.0,
        "rotation_pivot": "top_left",
    },
]


def _load_setup_module():
    setup_path = MW_DIR / "window_selection_setup.py"
    spec = importlib.util.spec_from_file_location("window_selection_setup", setup_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.MW_DIR = MW_DIR
    mod.ROOT = MW_DIR.parent
    mod.ensure_setup()
    return mod


def catalog_window_label(slot: int) -> str:
    """Human label for catalog slot (0-based index → windows 1..12)."""
    labels = WINDOW_SELECTION_LABELS
    if slot < 0 or slot >= len(labels):
        return f"{slot + 1}"
    return labels[slot]


def catalog_window_number(slot: int) -> str:
    """Window number 1..12 for plot annotations."""
    return str(int(slot) + 1)


def get_resolved_catalog(catalog_id: str = CATALOG_ID) -> list[dict[str, Any]]:
    """Return resolved window specs with stable slot indices 0..11."""
    if catalog_id != CATALOG_ID:
        raise ValueError(f"Unknown catalog_id={catalog_id!r}; expected {CATALOG_ID!r}")

    setup = _load_setup_module()
    catalog: list[dict[str, Any]] = []
    for cfg in WINDOW_CONFIGS:
        window_cfg = dict(cfg)
        if window_cfg["export_key"] == "magicwin_reference":
            window_cfg["window_top"] = setup.MAGIC_TOP
            window_cfg["window_left"] = setup.MAGIC_LEFT
        spec = setup.resolve_window(window_cfg)
        spec = dict(spec)
        spec["slot"] = len(catalog)
        catalog.append(spec)
    return catalog


def catalog_manifest(catalog: list[dict[str, Any]], *, catalog_id: str = CATALOG_ID) -> dict[str, Any]:
    return {
        "catalog_id": catalog_id,
        "source_notebook": "Magic_window/window_selection.ipynb",
        "n_windows": len(catalog),
        "windows": [
            {
                "slot": int(spec["slot"]),
                "export_key": spec["export_key"],
                "window_top": int(spec["window_top"]),
                "window_left": int(spec["window_left"]),
                "window_angle": float(spec["window_angle"]),
                "crop_height": int(spec["crop_height"]),
                "crop_width": int(spec["crop_width"]),
                "rotation_pivot": spec.get("rotation_pivot", "top_left"),
            }
            for spec in catalog
        ],
    }


def write_catalog_manifest(path: Path | str, catalog: list[dict[str, Any]] | None = None) -> Path:
    path = Path(path)
    if catalog is None:
        catalog = get_resolved_catalog()
    path.write_text(json.dumps(catalog_manifest(catalog), indent=2))
    return path
