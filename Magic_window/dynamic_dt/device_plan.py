"""Map each dynamic_dt forecast step to a dedicated GPU (0–7)."""

from __future__ import annotations

from dynamic_dt.dt_config import DT_DAYS_OPTIONS

# Eight dt values → eight devices cuda:0 … cuda:7
ALL_DT_DAYS: tuple[int, ...] = tuple(sorted(DT_DAYS_OPTIONS))
NUM_DEVICES = len(ALL_DT_DAYS)

DT_TO_DEVICE: dict[int, int] = {dt: i for i, dt in enumerate(ALL_DT_DAYS)}
DEVICE_TO_DT: dict[int, int] = {i: dt for dt, i in DT_TO_DEVICE.items()}


def device_id_for_dt(dt_days: int) -> int:
    if dt_days not in DT_TO_DEVICE:
        raise ValueError(
            f"dt_days={dt_days} has no device slot; "
            f"choose from {list(ALL_DT_DAYS)}"
        )
    return DT_TO_DEVICE[dt_days]


def dt_days_for_device(device_id: int) -> int:
    if device_id not in DEVICE_TO_DT:
        raise ValueError(
            f"device_id={device_id} out of range; use 0–{NUM_DEVICES - 1}"
        )
    return DEVICE_TO_DT[device_id]


def cuda_visible_devices(dt_days: int) -> str:
    return str(device_id_for_dt(dt_days))


def device_string(dt_days: int) -> str:
    """Device string inside a worker after CUDA_VISIBLE_DEVICES is set."""
    return "cuda:0"


def format_dt_device_plan(dt_days_list: list[int] | tuple[int, ...] | None = None) -> str:
    days = dt_days_list or ALL_DT_DAYS
    lines = []
    for dt in days:
        dev = device_id_for_dt(dt)
        lines.append(f"  dt={dt:2d}d → cuda:{dev} (CUDA_VISIBLE_DEVICES={dev})")
    return "\n".join(lines)
