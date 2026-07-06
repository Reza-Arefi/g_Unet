"""Fixed window catalog for dynamic_dt experiments."""

from __future__ import annotations

from typing import Any

from dynamic_dt.sequence_config import CONDITIONING_STEPS, FORECAST_STEPS

# Two windows — both export/predict u,v only; test metrics on speed (rollout).

DYNAMIC_DT_WINDOWS: dict[str, dict[str, Any]] = {
    "magicwin_reference": {
        "label": "Magic window (reference)",
        "window": {
            "export_key": "magicwin_reference",
            "crop_height": 128,
            "crop_width": 128,
            "window_top": 226,
            "window_left": 85,
            "window_angle": 0.0,
            "rotation_pivot": "top_left",
            "pivot_col": None,
            "pivot_row": None,
        },
        "export_components": ("uv",),
        "channel_names_train": ("u", "v"),
        "channel_names_rollout": ("U", "V"),
        "iD": 2,
    },
    "smallfixwin": {
        "label": "Small fix window (−45°)",
        "window": {
            "export_key": "smallfixwin",
            "crop_height": 128,
            "crop_width": 128,
            "window_top": 330,
            "window_left": 70,
            "window_angle": -45.0,
            "rotation_pivot": "top_left",
            "pivot_col": None,
            "pivot_row": None,
        },
        "export_components": ("uv",),
        "channel_names_train": ("u", "v"),
        "channel_names_rollout": ("U", "V"),
        "iD": 2,
    },
}

DEFAULT_EXPORT_KEYS: tuple[str, ...] = ("magicwin_reference", "smallfixwin")


def get_window_spec(export_key: str) -> dict[str, Any]:
    if export_key not in DYNAMIC_DT_WINDOWS:
        raise KeyError(
            f"Unknown export_key {export_key!r}; "
            f"choose from {list(DYNAMIC_DT_WINDOWS)}"
        )
    return DYNAMIC_DT_WINDOWS[export_key]


def primary_export_component(export_key: str) -> str:
    return get_window_spec(export_key)["export_components"][0]


def sequence_kwargs() -> dict[str, int]:
    return {
        "conditioning_steps": CONDITIONING_STEPS,
        "forecast_steps": FORECAST_STEPS,
    }
