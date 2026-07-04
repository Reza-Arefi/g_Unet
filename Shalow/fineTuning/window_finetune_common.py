"""Shared helpers for single-catalog-window fine-tuning (windows 1..12)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from Magic_window.window_catalog import (
    WINDOW_SELECTION_LABELS,
    catalog_window_label,
    get_resolved_catalog,
)


def window_number_to_slot(window_number: int) -> int:
    """User-facing window index 1..12 → catalog array slot 0..11."""
    w = int(window_number)
    if w < 1 or w > 12:
        raise ValueError(f"FINETUNE_WINDOW must be 1..12, got {w}")
    return w - 1


def resolve_finetune_window(window_number: int) -> dict[str, Any]:
    """Return slot, export_key, and label for one catalog window."""
    slot = window_number_to_slot(window_number)
    catalog = get_resolved_catalog()
    entry = catalog[slot]
    return {
        "window_number": int(window_number),
        "window_slot": slot,
        "catalog_export_key": entry["export_key"],
        "catalog_label": catalog_window_label(slot),
        "catalog_entry": entry,
    }


def list_catalog_windows() -> list[dict[str, Any]]:
    """All 12 catalog windows for display in notebooks."""
    catalog = get_resolved_catalog()
    rows = []
    for slot, entry in enumerate(catalog):
        rows.append(
            {
                "window_number": slot + 1,
                "window_slot": slot,
                "export_key": entry["export_key"],
                "label": WINDOW_SELECTION_LABELS[slot],
            }
        )
    return rows
