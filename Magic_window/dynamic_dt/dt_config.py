"""Forecast step spacing (dt) options for dynamic_dt experiments."""

from __future__ import annotations

# User-facing step sizes in days → Zarr loader dt in hours
DT_DAYS_OPTIONS: dict[int, int] = {
    3: 3 * 24,    # 72
    5: 5 * 24,    # 120
    7: 7 * 24,    # 168
    10: 10 * 24,  # 240
    12: 12 * 24,  # 288
    15: 15 * 24,  # 360
    17: 17 * 24,  # 408
    20: 20 * 24,  # 480
}

DEFAULT_DT_DAYS = 15

# Mid-range export batch (01 export → 04 train → 05 rollout on cuda:1 & cuda:7)
MID_DT_DAYS: tuple[int, ...] = (23, 25, 27, 30)

# All dt values from original sweep + mid-range (for notebook 06 comparison)
ALL_DT_DAYS_UP_TO_30: tuple[int, ...] = tuple(
    sorted(set(DT_DAYS_OPTIONS) | set(MID_DT_DAYS))
)


def dt_hours(dt_days: int) -> int:
    if dt_days in DT_DAYS_OPTIONS:
        return DT_DAYS_OPTIONS[dt_days]
    if isinstance(dt_days, int) and dt_days > 0:
        return dt_days * 24
    raise ValueError(
        f"dt_days={dt_days} not supported; choose from {sorted(DT_DAYS_OPTIONS)} "
        "or any positive integer (hours = dt_days × 24)"
    )


def dt_suffix(dt_hours: int) -> str:
    return f"_dt{int(dt_hours)}"


def forecast_horizons_days(dt_days: int) -> tuple[int, int, int]:
    """Rollout eval horizons: +1, +2, +3 model steps."""
    return (dt_days, 2 * dt_days, 3 * dt_days)
