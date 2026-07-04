"""Select training channels from multi-field exports (u, v, ssh, bathymetry)."""

from __future__ import annotations

from typing import Iterable

# Canonical order written by ``01_export_random_window_uvsshbathy.ipynb``.
EXPORT_UVSSH_BATHY_NAMES = ("u", "v", "ssh", "bathymetry")

TRAINING_MODES: dict[str, dict[str, tuple[int, ...] | tuple[str, ...] | None]] = {
    "uv": {
        "channel_indices": (0, 1),
        "target_channel_indices": None,
        "channel_names": ("u", "v"),
    },
    "uvssh": {
        "channel_indices": (0, 1, 2),
        "target_channel_indices": None,
        "channel_names": ("u", "v", "ssh"),
    },
    "uvbathy": {
        "channel_indices": (0, 1, 3),
        "target_channel_indices": None,
        "channel_names": ("u", "v", "bathymetry"),
    },
    "uvsshbathy": {
        "channel_indices": (0, 1, 2, 3),
        "target_channel_indices": None,
        "channel_names": ("u", "v", "ssh", "bathymetry"),
    },
    # u,v,bathy in → predict u,v only (same output as uv baseline)
    "uvbathy_uv": {
        "channel_indices": (0, 1, 3),
        "target_channel_indices": (0, 1),
        "channel_names": ("u", "v", "bathymetry"),
    },
    # u,v,ssh in → predict u,v only (same output as uv baseline)
    "uvssh_uv": {
        "channel_indices": (0, 1, 2),
        "target_channel_indices": (0, 1),
        "channel_names": ("u", "v", "ssh"),
    },
    # u,v,ssh,bathy in → predict u,v only (same output as uv baseline)
    "uvsshbathy_uv": {
        "channel_indices": (0, 1, 2, 3),
        "target_channel_indices": (0, 1),
        "channel_names": ("u", "v", "ssh", "bathymetry"),
    },
}


def resolve_target_channel_indices(
    channel_indices: tuple[int, ...],
    target_channel_indices: Iterable[int] | None = None,
) -> tuple[int, ...]:
    """Export indices for prediction targets; default matches input channels."""
    if target_channel_indices is not None:
        tgt = tuple(int(i) for i in target_channel_indices)
        if not tgt:
            raise ValueError("target_channel_indices must be non-empty")
        return tgt
    return channel_indices


def resolve_channel_indices(
    iD: int | None = None,
    channel_indices: Iterable[int] | None = None,
) -> tuple[int, ...]:
    """Return export indices to use; default is prefix ``0 .. iD-1``."""
    if channel_indices is not None:
        ch = tuple(int(i) for i in channel_indices)
        if not ch:
            raise ValueError("channel_indices must be non-empty")
        return ch
    if iD is None or int(iD) <= 0:
        raise ValueError("iD must be positive when channel_indices is None")
    return tuple(range(int(iD)))


def mode_channel_indices(mode: str) -> tuple[int, ...]:
    if mode not in TRAINING_MODES:
        raise ValueError(
            f"Unknown training mode {mode!r}; choose from {sorted(TRAINING_MODES)}"
        )
    return TRAINING_MODES[mode]["channel_indices"]  # type: ignore[return-value]


def mode_channel_names(mode: str) -> tuple[str, ...]:
    if mode not in TRAINING_MODES:
        raise ValueError(
            f"Unknown training mode {mode!r}; choose from {sorted(TRAINING_MODES)}"
        )
    return TRAINING_MODES[mode]["channel_names"]  # type: ignore[return-value]


def mode_target_channel_indices(mode: str) -> tuple[int, ...] | None:
    if mode not in TRAINING_MODES:
        raise ValueError(
            f"Unknown training mode {mode!r}; choose from {sorted(TRAINING_MODES)}"
        )
    return TRAINING_MODES[mode].get("target_channel_indices")  # type: ignore[return-value]


def apply_training_mode(
    config,
    mode: str,
    *,
    set_channel_names: bool = True,
) -> None:
    """Set ``channel_indices``, ``target_channel_indices``, ``iD``, and ``channel_names``."""
    ch = mode_channel_indices(mode)
    tgt = mode_target_channel_indices(mode)
    names = mode_channel_names(mode)
    config.channel_indices = ch
    config.target_channel_indices = tgt
    ch_out = resolve_target_channel_indices(ch, tgt)
    config.iD = len(ch_out)
    if set_channel_names and hasattr(config, "channel_names"):
        config.channel_names = names[: config.iD] if tgt else names
