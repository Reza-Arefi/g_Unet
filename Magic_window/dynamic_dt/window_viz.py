"""Window preview for dynamic_dt export notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

WINDOW_COLORS = {
    "magicwin_reference": "#00E5FF",
    "smallfixwin": "#FF6600",
}


def preview_export_window(
    export_key: str,
    *,
    time_idx: int = 0,
    save_path: Path | str | None = None,
) -> dict[str, Any]:
    """
    Plot full-domain + crop panels (bathymetry, u, v) for a catalog window.

    Reuses ``window_selection_setup.preview_window``.
    """
    import matplotlib.pyplot as plt

    from dynamic_dt.window_catalog import get_window_spec
    from window_selection_setup import ensure_setup, preview_window

    ensure_setup()
    entry = get_window_spec(export_key)
    window_cfg = {
        **entry["window"],
        "color": WINDOW_COLORS.get(export_key, "#FFEB3B"),
    }
    spec = preview_window(window_cfg, time_idx=time_idx)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.gcf().savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved window preview: {save_path}")

    return spec
