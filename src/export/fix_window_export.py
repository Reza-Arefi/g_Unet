"""Resumable fix-window export from Guiana Zarr to .npy files."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataloaders.guiana_margem_tensor import GuianaDataloader
from src.data.useful_transforms import RotatedFixedCrop


@dataclass
class ExportConfig:
    zarr_path: str
    dt: int = 96  # hours between frames (96 = 4-day steps); use 24 for daily
    conditioning_steps: int = 4
    forecast_steps: int = 56
    crop_height: int = 128
    crop_width: int = 128
    window_top: int = 330
    window_left: int = 70
    window_angle: float = -45.0
    seed: int = 42
    n_jobs: int = 30
    max_mask_frac: float = 0.7
    hour_of_day: int | list[int] | None = None
    start_time: str = "2016-01-01"
    end_time: str = "2019-12-31"
    output_prefix: str = "data/sample_16_19_fixwin"
    n_samples_to_gen: int = 5000
    chunk_size: int = 100
    # When set, export exactly these Zarr dataset indices (skips random subsample).
    fixed_dataset_indices: tuple[int, ...] | None = None
    # Old: ("u", "v", "ssh") all 3 | ("u", "v", "speed") predict speed only
    channel_names: tuple[str, ...] = ("u", "v")
    target_channel_indices: tuple[int, ...] = (0, 1)  # predict u and v

    @property
    def n_channels(self) -> int:
        return len(self.channel_names)

    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExportConfig:
        allowed = {f.name for f in fields(cls)}
        values = {k: v for k, v in data.items() if k in allowed}
        if "channel_names" in values and isinstance(values["channel_names"], list):
            values["channel_names"] = tuple(values["channel_names"])
        if "target_channel_indices" in values and isinstance(values["target_channel_indices"], list):
            values["target_channel_indices"] = tuple(values["target_channel_indices"])
        if "fixed_dataset_indices" in values and values["fixed_dataset_indices"] is not None:
            fdi = values["fixed_dataset_indices"]
            values["fixed_dataset_indices"] = tuple(int(x) for x in fdi)
        return cls(**values)


def candidate_indices_for_hour(dataset, hour_of_day):
    """Dataset indices whose sequence starts at an allowed hour (0-23)."""
    if hour_of_day is None:
        return np.arange(len(dataset), dtype=np.int64)

    if isinstance(hour_of_day, (int, np.integer)):
        allowed = {int(hour_of_day)}
    else:
        allowed = {int(h) for h in hour_of_day}

    invalid = allowed - set(range(24))
    if invalid:
        raise ValueError(f"Invalid hour(s) {sorted(invalid)}; expected integers 0-23")

    return np.array(
        [
            i
            for i, t0 in enumerate(dataset.starts)
            if pd.Timestamp(t0).hour in allowed
        ],
        dtype=np.int64,
    )


def stack_time_channels(flat: torch.Tensor, n_frames: int, n_channels: int = 3) -> np.ndarray:
    """Reshape flat u/v/ssh stack from GuianaDataloader; keep first `n_channels` fields."""
    t = flat.detach().cpu().numpy()
    zarr_channels = 3
    full = t.reshape(n_frames, zarr_channels, t.shape[-2], t.shape[-1])
    return full[:, :n_channels, :, :]


def uv_to_uvspeed(frames: np.ndarray) -> np.ndarray:
    """Replace ssh (channel 2) with speed |u,v| = sqrt(u^2 + v^2). Shape (..., 3, H, W)."""
    out = np.array(frames, copy=True)
    u = out[..., 0, :, :]
    v = out[..., 1, :, :]
    out[..., 2, :, :] = np.sqrt(u * u + v * v)
    return out


def _bathymetry_field_2d(bat: np.ndarray) -> np.ndarray:
    """Return cropped bathymetry as (H, W); loader stores (1, H, W) after crop."""
    field = np.squeeze(np.asarray(bat))
    if field.ndim != 2:
        raise ValueError(f"Expected bathymetry field (H, W), got shape {np.asarray(bat).shape}")
    return field


def _append_bathymetry_channel(seq: np.ndarray, bat: np.ndarray) -> np.ndarray:
    """Append time-invariant bathymetry as the last channel. seq: (T, C, H, W), bat: (H, W) or (1, H, W)."""
    bat2d = _bathymetry_field_2d(bat)
    bat_t = np.broadcast_to(
        bat2d[np.newaxis, np.newaxis, :, :],
        (seq.shape[0], 1, bat2d.shape[0], bat2d.shape[1]),
    )
    return np.concatenate([seq, bat_t.astype(seq.dtype, copy=False)], axis=1)


def transform_export_channels(seq: np.ndarray, channel_names: tuple[str, ...]) -> np.ndarray:
    names = tuple(channel_names)
    if names == ("u", "v", "ssh"):
        return seq
    if names == ("u", "v"):
        return seq[:, :2, :, :]
    if names == ("u", "v", "speed"):
        return uv_to_uvspeed(seq)
    if names == ("u",):
        return seq[:, 0:1, :, :]
    if names == ("v",):
        return seq[:, 1:2, :, :]
    if names == ("bathymetry",):
        raise ValueError("bathymetry-only export is handled in load_sequence")
    raise ValueError(f"Unsupported channel_names={names!r}")


def load_sequence(
    dataset,
    idx: int,
    *,
    channel_names: tuple[str, ...] = ("u", "v"),
):
    dataset.device = "cpu"
    x, y, mask = dataset[idx]
    names = tuple(channel_names)
    include_bathy = "bathymetry" in names

    if names == ("bathymetry",):
        bat2d = _bathymetry_field_2d(y["bat"].detach().cpu().numpy())
        total_steps = int(dataset.conditioning_steps + dataset.forecast_steps)
        seq = np.broadcast_to(
            bat2d[np.newaxis, np.newaxis, :, :],
            (total_steps, 1, bat2d.shape[0], bat2d.shape[1]),
        ).copy()
    else:
        dyn_names = tuple(c for c in names if c != "bathymetry")
        n_dyn = len(dyn_names)
        cond = stack_time_channels(y["y"], dataset.conditioning_steps, n_channels=n_dyn)
        fcst = stack_time_channels(x, dataset.forecast_steps, n_channels=n_dyn)
        seq = np.concatenate([cond, fcst], axis=0)
        seq = transform_export_channels(seq, dyn_names)
        if include_bathy:
            bat = y["bat"].detach().cpu().numpy()
            seq = _append_bathymetry_channel(seq, bat)

    times = dataset.sample_times(idx)
    return seq, mask.detach().cpu().numpy(), times, y


def _build_dataset(config: ExportConfig):
    fixed_crop = RotatedFixedCrop(
        size=(config.crop_height, config.crop_width),
        top=config.window_top,
        left=config.window_left,
        angle_deg=config.window_angle,
    )
    base_kwargs = dict(
        path=config.zarr_path,
        dt=config.dt,
        conditioning_steps=config.conditioning_steps,
        forecast_steps=config.forecast_steps,
        start_time=config.start_time,
        end_time=config.end_time,
    )
    # CPU only: parallel joblib workers each load samples; GPU would OOM with n_jobs>1.
    return GuianaDataloader(**base_kwargs, transform=fixed_crop, device="cpu")


def _process_sample(dataset, idx: int, config: ExportConfig):
    dataset.device = "cpu"
    seq, mask, times, y = load_sequence(dataset, int(idx), channel_names=config.channel_names)
    max_mask_frac = config.max_mask_frac
    if float(mask.mean()) >= max_mask_frac:
        return None

    t_labels = [str(t)[:16] for t in times]
    anchor_idx = config.conditioning_steps - 1
    dt = datetime.strptime(t_labels[anchor_idx], "%Y-%m-%d %H:%M")
    day_of_year = dt.timetuple().tm_yday
    hour = dt.hour
    latlon = y["latlon"].tolist()
    return seq, mask[0, :, :], day_of_year, hour, latlon


def _paths(config: ExportConfig):
    prefix = Path(config.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    chunks_dir = Path(f"{config.output_prefix}_chunks")
    state_path = Path(f"{config.output_prefix}_export_state.json")
    indices_path = Path(f"{config.output_prefix}_indices.npy")
    config_path = Path(f"{config.output_prefix}_export_config.json")
    return prefix, chunks_dir, state_path, indices_path, config_path


def _chunk_path(chunks_dir: Path, chunk_idx: int) -> Path:
    return chunks_dir / f"chunk_{chunk_idx:05d}.npz"


def _pack_chunk_results(
    results,
    total_steps: int,
    crop_h: int,
    crop_w: int,
    n_channels: int = 3,
):
    valid = [r for r in results if r is not None]
    n = len(valid)
    if n == 0:
        return None

    sims = np.zeros((n, total_steps, n_channels, crop_h, crop_w), dtype=np.float32)
    mask = np.zeros((n, crop_h, crop_w), dtype=np.float32)
    day = np.zeros((n,), dtype=np.int16)
    hour = np.zeros((n,), dtype=np.int8)
    latlon = np.zeros((n, 2), dtype=np.float32)

    for i, (seq, m, d, h, ll) in enumerate(valid):
        sims[i] = seq
        mask[i] = m
        day[i] = d
        hour[i] = h
        latlon[i] = ll

    return {"sims": sims, "mask": mask, "day": day, "hour": hour, "latlon": latlon, "n_valid": n}


def _init_state(config: ExportConfig, resume: bool) -> dict[str, Any]:
    _, chunks_dir, state_path, indices_path, config_path = _paths(config)

    config_path.write_text(json.dumps(asdict(config), indent=2, default=str))

    if not resume:
        if state_path.exists():
            state_path.unlink()
        if chunks_dir.exists():
            shutil.rmtree(chunks_dir)
        if indices_path.exists():
            indices_path.unlink()

    if resume and state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("config_hash") != config.config_hash():
            raise RuntimeError(
                f"Export config changed since last run. Delete {state_path} or use a new OUTPUT_PREFIX."
            )
        if state.get("status") == "finished":
            print(f"Export already finished ({state_path}). Final files at {config.output_prefix}_*.npy")
            return state
        print(f"Resuming export: {len(state['completed_chunks'])}/{state['n_chunks']} chunks done")
        return state

    dataset = _build_dataset(config)
    candidate_idx = candidate_indices_for_hour(dataset, config.hour_of_day)
    if len(candidate_idx) == 0:
        raise RuntimeError(
            f"No valid sequences for hour_of_day={config.hour_of_day!r} in "
            f"{config.start_time} .. {config.end_time}"
        )

    if config.fixed_dataset_indices is not None:
        indices = np.asarray(config.fixed_dataset_indices, dtype=np.int64)
        n_to_gen = len(indices)
        if n_to_gen == 0:
            raise RuntimeError("fixed_dataset_indices is empty")
    else:
        n_to_gen = min(config.n_samples_to_gen, len(candidate_idx))
        if n_to_gen >= len(candidate_idx):
            indices = np.asarray(candidate_idx, dtype=np.int64)
        else:
            rng = np.random.default_rng(config.seed)
            indices = rng.choice(candidate_idx, size=n_to_gen, replace=False)

    np.save(indices_path, indices)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    n_chunks = int(np.ceil(n_to_gen / config.chunk_size))
    state = {
        "status": "running",
        "config_hash": config.config_hash(),
        "output_prefix": config.output_prefix,
        "indices_path": str(indices_path),
        "n_to_gen": int(n_to_gen),
        "chunk_size": int(config.chunk_size),
        "n_chunks": n_chunks,
        "total_steps": int(dataset.total_steps),
        "completed_chunks": [],
        "n_valid": 0,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": None,
    }
    state_path.write_text(json.dumps(state, indent=2))
    print(
        f"Exporting {n_to_gen} samples in {n_chunks} chunks of {config.chunk_size} "
        f"(pool {len(candidate_idx)} hour-filtered, {len(dataset)} total)"
    )
    return state


def _save_state(state_path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    state_path.write_text(json.dumps(state, indent=2))


def run_export(config: ExportConfig, *, resume: bool = True) -> dict[str, Any]:
    """Process export in resumable chunks; merge to final .npy when complete."""
    _, chunks_dir, state_path, indices_path, _ = _paths(config)
    state = _init_state(config, resume=resume)

    if state.get("status") == "finished":
        return state

    dataset = _build_dataset(config)
    indices = np.load(indices_path)
    completed = set(state["completed_chunks"])

    for chunk_idx in range(state["n_chunks"]):
        if chunk_idx in completed:
            continue

        start = chunk_idx * state["chunk_size"]
        end = min(start + state["chunk_size"], state["n_to_gen"])
        batch_indices = indices[start:end]

        print(f"Chunk {chunk_idx + 1}/{state['n_chunks']} | indices {start}:{end} ...", flush=True)
        results = Parallel(n_jobs=config.n_jobs, verbose=1, prefer="threads")(
            delayed(_process_sample)(dataset, int(idx), config) for idx in batch_indices
        )

        packed = _pack_chunk_results(
            results,
            state["total_steps"],
            config.crop_height,
            config.crop_width,
            n_channels=config.n_channels,
        )
        chunk_file = _chunk_path(chunks_dir, chunk_idx)
        if packed is None:
            np.savez_compressed(chunk_file, n_valid=0)
            print(f"  chunk {chunk_idx}: 0/{len(batch_indices)} passed mask filter", flush=True)
        else:
            np.savez_compressed(chunk_file, **packed)
            state["n_valid"] += int(packed["n_valid"])
            print(
                f"  chunk {chunk_idx}: {packed['n_valid']}/{len(batch_indices)} valid "
                f"(total valid so far: {state['n_valid']})",
                flush=True,
            )

        state["completed_chunks"].append(chunk_idx)
        _save_state(state_path, state)

    merge_export_chunks(config, state=state)
    return state


def merge_export_chunks(config: ExportConfig, *, state: dict[str, Any] | None = None) -> None:
    """Merge chunk files into final .npy outputs."""
    _, chunks_dir, state_path, _, _ = _paths(config)

    if state is None:
        if not state_path.exists():
            raise FileNotFoundError(f"No export state at {state_path}")
        state = json.loads(state_path.read_text())

    chunk_files = sorted(chunks_dir.glob("chunk_*.npz"))
    if len(chunk_files) != state["n_chunks"]:
        raise RuntimeError(
            f"Expected {state['n_chunks']} chunk files, found {len(chunk_files)} in {chunks_dir}"
        )

    parts = []
    for path in chunk_files:
        data = np.load(path)
        n_valid = int(data["n_valid"]) if "n_valid" in data else 0
        if n_valid == 0:
            continue
        parts.append({k: data[k] for k in ("sims", "mask", "day", "hour", "latlon")})

    if not parts:
        raise RuntimeError("No valid samples exported (all failed mask filter).")

    data_sims = np.concatenate([p["sims"] for p in parts], axis=0)
    data_mask = np.concatenate([p["mask"] for p in parts], axis=0)
    data_day = np.concatenate([p["day"] for p in parts], axis=0)
    data_hour = np.concatenate([p["hour"] for p in parts], axis=0)
    data_latlon = np.concatenate([p["latlon"] for p in parts], axis=0)

    window_meta = np.array(
        [
            config.window_top,
            config.window_left,
            config.crop_height,
            config.crop_width,
            config.window_angle,
        ],
        dtype=np.float64,
    )

    np.save(f"{config.output_prefix}_sims.npy", data_sims)
    np.save(f"{config.output_prefix}_mask.npy", data_mask)
    np.save(f"{config.output_prefix}_day.npy", data_day)
    np.save(f"{config.output_prefix}_hour.npy", data_hour)
    np.save(f"{config.output_prefix}_latlon.npy", data_latlon)
    np.save(f"{config.output_prefix}_window.npy", window_meta)

    state["status"] = "finished"
    state["n_valid"] = int(data_sims.shape[0])
    state["finished_at"] = datetime.now().isoformat(timespec="seconds")
    _save_state(state_path, state)

    target_names = [config.channel_names[i] for i in config.target_channel_indices]

    print("Saved:")
    print(f"  {config.output_prefix}_sims.npy   {data_sims.shape}")
    print(f"  {config.output_prefix}_mask.npy   {data_mask.shape}")
    print(f"  {config.output_prefix}_window.npy {window_meta}")
    print(f"  channels: {list(config.channel_names)} | target (GT/pred): {target_names}")
    if data_sims.shape[0]:
        print(f"  exported hours: {sorted(set(int(h) for h in data_hour))}")
