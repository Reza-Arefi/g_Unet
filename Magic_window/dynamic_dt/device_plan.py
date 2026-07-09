"""Map dynamic_dt forecast steps to GPU devices."""

from __future__ import annotations

from typing import Sequence

from dynamic_dt.dt_config import DT_DAYS_OPTIONS, MID_DT_DAYS

# Original 8 dt → cuda:0 … cuda:7 (legacy train/rollout batch)
STANDARD_DT_DAYS: tuple[int, ...] = tuple(sorted(DT_DAYS_OPTIONS))
ALL_DT_DAYS = STANDARD_DT_DAYS  # back-compat alias

# Mid-range export batch (notebooks 04/05 on cuda:1 and cuda:7)
MID_DT_DAYS_LIST: tuple[int, ...] = MID_DT_DAYS
GPU_POOL_1_7: tuple[int, ...] = (1, 7)

# Full sweep for comparison plots (notebook 06)
ALL_DT_DAYS_UP_TO_30: tuple[int, ...] = tuple(sorted(set(STANDARD_DT_DAYS) | set(MID_DT_DAYS)))

# Legacy 1:1 mapping dt[i] → gpu[i] for STANDARD_DT_DAYS only
_LEGACY_DT_TO_DEVICE: dict[int, int] = {dt: i for i, dt in enumerate(STANDARD_DT_DAYS)}

# Optional override set via configure_device_plan()
_active_dt_to_device: dict[int, int] | None = None


def configure_device_plan(
    dt_days_list: Sequence[int],
    gpu_pool: Sequence[int],
) -> dict[int, int]:
    """Round-robin map each dt onto gpu_pool (e.g. dt 23,27→1 and 25,30→7)."""
    global _active_dt_to_device
    pool = tuple(int(g) for g in gpu_pool)
    if not pool:
        raise ValueError("gpu_pool must contain at least one device id")
    days = sorted(int(d) for d in dt_days_list)
    _active_dt_to_device = {dt: pool[i % len(pool)] for i, dt in enumerate(days)}
    return dict(_active_dt_to_device)


def clear_device_plan() -> None:
    global _active_dt_to_device
    _active_dt_to_device = None


def active_device_plan() -> dict[int, int] | None:
    return None if _active_dt_to_device is None else dict(_active_dt_to_device)


def device_id_for_dt(dt_days: int) -> int:
    dt_days = int(dt_days)
    if _active_dt_to_device is not None:
        if dt_days not in _active_dt_to_device:
            raise ValueError(
                f"dt_days={dt_days} not in active plan {sorted(_active_dt_to_device)}; "
                "call configure_device_plan() first"
            )
        return _active_dt_to_device[dt_days]
    if dt_days not in _LEGACY_DT_TO_DEVICE:
        raise ValueError(
            f"dt_days={dt_days} has no device slot; "
            f"choose from {list(STANDARD_DT_DAYS)} or call configure_device_plan()"
        )
    return _LEGACY_DT_TO_DEVICE[dt_days]


def cuda_visible_devices(dt_days: int) -> str:
    return str(device_id_for_dt(dt_days))


def device_string(dt_days: int) -> str:
    """Device string inside a worker after CUDA_VISIBLE_DEVICES is set."""
    return "cuda:0"


def format_dt_device_plan(dt_days_list: Sequence[int] | None = None) -> str:
    if dt_days_list is None:
        if _active_dt_to_device is not None:
            days = sorted(_active_dt_to_device)
        else:
            days = list(STANDARD_DT_DAYS)
    else:
        days = sorted(int(d) for d in dt_days_list)
    lines = []
    for dt in days:
        dev = device_id_for_dt(dt)
        lines.append(f"  dt={dt:2d}d → cuda:{dev} (CUDA_VISIBLE_DEVICES={dev})")
    return "\n".join(lines)
