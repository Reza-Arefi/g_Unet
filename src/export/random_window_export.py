"""Resumable random-window export from Guiana Zarr to .npy files."""

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
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataloaders.guiana_margem_tensor import GuianaDataloader
from src.export.fix_window_export import (
    candidate_indices_for_hour,
    stack_time_channels,
    transform_export_channels,
)
from src.train.fix_window_train import split_train_val_indices


@dataclass
class RandomWindowExportConfig:
    zarr_path: str
    dt: int = 360
    conditioning_steps: int = 3
    forecast_steps: int = 1
    crop_height: int = 128
    crop_width: int = 128
    seed: int = 42
    n_jobs: int = 30
    # Drop sample when invalid/land mask fraction >= this value (e.g. 0.05 → keep <5% land).
    max_mask_frac: float = 0.05
    max_crop_attempts: int = 200
    hour_of_day: int | list[int] | None = 12
    start_time: str = "2013-01-01"
    end_time: str = "2022-12-31"
    output_prefix: str = "Shalow/data/random_win_128/sample_13_22_train_uv_dt360"
    n_samples_to_gen: int = 3600
    crops_per_anchor: int = 1  # K random crops per temporal anchor (same zarr idx, different windows)
    chunk_size: int = 100
    fixed_dataset_indices: tuple[int, ...] | None = None
    channel_names: tuple[str, ...] = ("u", "v")
    target_channel_indices: tuple[int, ...] = (0, 1)
    export_key: str = "random_win_128"
    # Pre-export train crop windows per epoch (matches dynamic training RNG; read at train time).
    export_epoch_windows: bool = True
    n_export_epochs: int = 100
    random_crop_seed: int = 42
    val_split: float = 0.3
    val_split_mode: str = "random"  # "random" | "chronological"
    val_split_seed: int = 42
    train_crops_for_epoch_windows: tuple[int, ...] = (1, 2)  # k1 and k2 row filters

    @property
    def n_channels(self) -> int:
        return len(self.channel_names)

    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RandomWindowExportConfig:
        allowed = {f.name for f in fields(cls)}
        values = {k: v for k, v in data.items() if k in allowed}
        if "channel_names" in values and isinstance(values["channel_names"], list):
            values["channel_names"] = tuple(values["channel_names"])
        if "target_channel_indices" in values and isinstance(values["target_channel_indices"], list):
            values["target_channel_indices"] = tuple(values["target_channel_indices"])
        if "fixed_dataset_indices" in values and values["fixed_dataset_indices"] is not None:
            values["fixed_dataset_indices"] = tuple(int(x) for x in values["fixed_dataset_indices"])
        if "train_crops_for_epoch_windows" in values and isinstance(
            values["train_crops_for_epoch_windows"], list
        ):
            values["train_crops_for_epoch_windows"] = tuple(
                int(x) for x in values["train_crops_for_epoch_windows"]
            )
        return cls(**values)


def load_sequence_at_window(
    dataset,
    idx: int,
    *,
    top: int,
    left: int,
    crop_h: int,
    crop_w: int,
    channel_names: tuple[str, ...] = ("u", "v"),
):
    """Load one sequence on the full grid, then crop to ``(top, left, crop_h, crop_w)``."""
    x, y, mask = dataset[int(idx)]
    n_channels = len(channel_names)
    cond_steps = dataset.conditioning_steps
    fcst_steps = dataset.forecast_steps

    cond = stack_time_channels(y["y"], cond_steps, n_channels=n_channels)
    fcst = stack_time_channels(x, fcst_steps, n_channels=n_channels)
    seq = np.concatenate([cond, fcst], axis=0)
    seq = transform_export_channels(seq, channel_names)
    seq = seq[..., top : top + crop_h, left : left + crop_w]

    mask_np = mask.detach().cpu().numpy()
    if mask_np.ndim == 3:
        mask_np = mask_np[0]
    mask_crop = mask_np[top : top + crop_h, left : left + crop_w]

    latlon_grid = y["latlon_grid"].detach().cpu().numpy()
    cy = top + crop_h // 2
    cx = left + crop_w // 2
    latlon = [float(latlon_grid[0, cy, cx]), float(latlon_grid[1, cy, cx])]

    times = dataset.sample_times(int(idx))
    return seq, mask_crop, times, latlon


def _build_dataset(config: RandomWindowExportConfig):
    base_kwargs = dict(
        path=config.zarr_path,
        dt=config.dt,
        conditioning_steps=config.conditioning_steps,
        forecast_steps=config.forecast_steps,
        start_time=config.start_time,
        end_time=config.end_time,
    )
    return GuianaDataloader(**base_kwargs, transform=None, device="cpu")


def _grid_size(dataset) -> tuple[int, int]:
    info = dataset.info()
    return int(info["space"]["height"]), int(info["space"]["width"])


def plan_multi_crop_export(
    candidate_idx: np.ndarray,
    *,
    n_samples_to_gen: int,
    crops_per_anchor: int,
    seed: int,
    unique_anchors: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Plan export rows: ``n_anchors`` unique starts, each with ``K`` independent crops.

    Returns ``(unique_anchors, zarr_indices, crop_slots)`` where ``zarr_indices`` has length
    ``n_anchors * K`` (repeated anchors) and ``crop_slots`` is ``0 .. K-1`` per anchor.
    """
    K = max(1, int(crops_per_anchor))
    candidates = np.asarray(candidate_idx, dtype=np.int64)
    if unique_anchors is not None:
        anchors = np.asarray(unique_anchors, dtype=np.int64)
    else:
        n_anchors = min(len(candidates), int(n_samples_to_gen) // K)
        if n_anchors <= 0:
            raise ValueError("n_samples_to_gen too small for crops_per_anchor")
        rng = np.random.default_rng(seed)
        if n_anchors >= len(candidates):
            anchors = candidates
        else:
            anchors = rng.choice(candidates, size=n_anchors, replace=False)
    zarr_indices = np.repeat(anchors, K)
    crop_slots = np.tile(np.arange(K, dtype=np.int8), len(anchors))
    return anchors, zarr_indices, crop_slots


def find_valid_crop_window(
    dataset,
    ds_idx: int,
    pool_idx: int,
    epoch: int,
    *,
    crop_seed: int,
    crop_h: int,
    crop_w: int,
    max_mask_frac: float,
    max_crop_attempts: int,
    channel_names: tuple[str, ...] = ("u", "v"),
) -> np.ndarray:
    """Match ``RandomWindowZarrDataset`` train RNG: seed + epoch + zarr idx + pool idx."""
    h, w = _grid_size(dataset)
    if h < crop_h or w < crop_w:
        raise ValueError(f"Crop {crop_h}x{crop_w} does not fit grid {h}x{w}")
    rng = np.random.default_rng([int(crop_seed), int(epoch), int(ds_idx), int(pool_idx)])
    for _ in range(max_crop_attempts):
        top = int(rng.integers(0, h - crop_h + 1))
        left = int(rng.integers(0, w - crop_w + 1))
        _, mask, _, _ = load_sequence_at_window(
            dataset,
            int(ds_idx),
            top=top,
            left=left,
            crop_h=crop_h,
            crop_w=crop_w,
            channel_names=channel_names,
        )
        if float(mask.mean()) < float(max_mask_frac):
            return np.array([top, left, crop_h, crop_w], dtype=np.float64)
    raise RuntimeError(
        f"No valid crop (<{max_mask_frac:.0%} land) for zarr index {ds_idx} "
        f"epoch={epoch} pool_idx={pool_idx}"
    )


def _process_sample(
    dataset,
    idx: int,
    config: RandomWindowExportConfig,
    crop_slot: int = 0,
):
    dataset.device = "cpu"
    h, w = _grid_size(dataset)
    crop_h, crop_w = config.crop_height, config.crop_width
    if h < crop_h or w < crop_w:
        raise ValueError(f"Crop {crop_h}x{crop_w} does not fit grid {h}x{w}")

    rng = np.random.default_rng([config.seed, int(idx), int(crop_slot)])
    anchor_idx = config.conditioning_steps - 1

    for _ in range(config.max_crop_attempts):
        top = int(rng.integers(0, h - crop_h + 1))
        left = int(rng.integers(0, w - crop_w + 1))
        seq, mask, times, latlon = load_sequence_at_window(
            dataset,
            int(idx),
            top=top,
            left=left,
            crop_h=crop_h,
            crop_w=crop_w,
            channel_names=config.channel_names,
        )
        mask_frac = float(mask.mean())
        if mask_frac >= config.max_mask_frac:
            continue

        t_labels = [str(t)[:16] for t in times]
        dt = datetime.strptime(t_labels[anchor_idx], "%Y-%m-%d %H:%M")
        day_of_year = dt.timetuple().tm_yday
        hour = dt.hour
        window = np.array([top, left, crop_h, crop_w], dtype=np.float64)
        return seq, mask, day_of_year, hour, latlon, window

    return None


def _paths(config: RandomWindowExportConfig):
    prefix = Path(config.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    chunks_dir = Path(f"{config.output_prefix}_chunks")
    state_path = Path(f"{config.output_prefix}_export_state.json")
    indices_path = Path(f"{config.output_prefix}_indices.npy")
    crop_slots_path = Path(f"{config.output_prefix}_crop_slots.npy")
    config_path = Path(f"{config.output_prefix}_export_config.json")
    return prefix, chunks_dir, state_path, indices_path, crop_slots_path, config_path


def _clear_export_progress(config: RandomWindowExportConfig) -> None:
    """Remove chunk dir and resume state so export can start from scratch."""
    _, chunks_dir, state_path, indices_path, crop_slots_path, _ = _paths(config)
    if state_path.exists():
        state_path.unlink()
    if chunks_dir.exists():
        shutil.rmtree(chunks_dir)
    if indices_path.exists():
        indices_path.unlink()
    if crop_slots_path.exists():
        crop_slots_path.unlink()


def export_wants_fresh_start(config: RandomWindowExportConfig, *, resume: bool = True) -> bool:
    """True when export should not resume (explicit --no-resume or config hash mismatch)."""
    if not resume:
        return True
    _, _, state_path, _, _, _ = _paths(config)
    if not state_path.exists():
        return True
    state = json.loads(state_path.read_text())
    return state.get("config_hash") != config.config_hash()


def _chunk_path(chunks_dir: Path, chunk_idx: int) -> Path:
    return chunks_dir / f"chunk_{chunk_idx:05d}.npz"


def _pack_chunk_results(
    results,
    total_steps: int,
    crop_h: int,
    crop_w: int,
    n_channels: int = 2,
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
    windows = np.zeros((n, 4), dtype=np.float64)

    for i, (seq, m, d, h, ll, win) in enumerate(valid):
        sims[i] = seq
        mask[i] = m
        day[i] = d
        hour[i] = h
        latlon[i] = ll
        windows[i] = win

    return {
        "sims": sims,
        "mask": mask,
        "day": day,
        "hour": hour,
        "latlon": latlon,
        "windows": windows,
        "n_valid": n,
    }


def _init_state(config: RandomWindowExportConfig, resume: bool) -> dict[str, Any]:
    _, chunks_dir, state_path, indices_path, crop_slots_path, config_path = _paths(config)

    config_dict = asdict(config)
    config_dict["window_mode"] = "random"
    config_path.write_text(json.dumps(config_dict, indent=2, default=str))

    if not resume:
        _clear_export_progress(config)

    if resume and state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("config_hash") != config.config_hash():
            print(
                f"Export config changed — restarting from scratch "
                f"({Path(config.output_prefix).name})",
                flush=True,
            )
            _clear_export_progress(config)
        elif state.get("status") == "finished":
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

    K = max(1, int(config.crops_per_anchor))
    if config.fixed_dataset_indices is not None:
        unique_anchors = np.asarray(config.fixed_dataset_indices, dtype=np.int64)
        if len(unique_anchors) == 0:
            raise RuntimeError("fixed_dataset_indices is empty")
        _, indices, crop_slots = plan_multi_crop_export(
            candidate_idx,
            n_samples_to_gen=config.n_samples_to_gen,
            crops_per_anchor=K,
            seed=config.seed,
            unique_anchors=unique_anchors,
        )
        n_anchors = len(unique_anchors)
    else:
        unique_anchors, indices, crop_slots = plan_multi_crop_export(
            candidate_idx,
            n_samples_to_gen=config.n_samples_to_gen,
            crops_per_anchor=K,
            seed=config.seed,
        )
        n_anchors = len(unique_anchors)

    n_to_gen = len(indices)
    np.save(indices_path, indices)
    if K > 1:
        np.save(crop_slots_path, crop_slots)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    n_chunks = int(np.ceil(n_to_gen / config.chunk_size))
    state = {
        "status": "running",
        "config_hash": config.config_hash(),
        "output_prefix": config.output_prefix,
        "indices_path": str(indices_path),
        "n_to_gen": int(n_to_gen),
        "n_anchors": int(n_anchors),
        "crops_per_anchor": int(K),
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
        f"Exporting {n_to_gen} random-window rows ({n_anchors} anchors × {K} crops/anchor) "
        f"in {n_chunks} chunks of {config.chunk_size} "
        f"(pool {len(candidate_idx)} hour-filtered, {len(dataset)} total)",
        flush=True,
    )
    return state


def _save_state(state_path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    state_path.write_text(json.dumps(state, indent=2))


def run_export(config: RandomWindowExportConfig, *, resume: bool = True) -> dict[str, Any]:
    """Process random-window export in resumable chunks; merge to final .npy when complete."""
    _, chunks_dir, state_path, indices_path, crop_slots_path, _ = _paths(config)
    state = _init_state(config, resume=resume)

    if state.get("status") == "finished":
        if config.export_epoch_windows:
            _maybe_export_epoch_windows(config)
        return state

    dataset = _build_dataset(config)
    indices = np.load(indices_path)
    if crop_slots_path.exists():
        crop_slots = np.load(crop_slots_path)
    else:
        crop_slots = np.zeros(len(indices), dtype=np.int8)
    completed = set(state["completed_chunks"])

    for chunk_idx in range(state["n_chunks"]):
        if chunk_idx in completed:
            continue

        start = chunk_idx * state["chunk_size"]
        end = min(start + state["chunk_size"], state["n_to_gen"])
        batch_indices = indices[start:end]
        batch_slots = crop_slots[start:end]

        print(f"Chunk {chunk_idx + 1}/{state['n_chunks']} | indices {start}:{end} ...", flush=True)
        results = Parallel(n_jobs=config.n_jobs, verbose=1)(
            delayed(_process_sample)(dataset, int(idx), config, int(slot))
            for idx, slot in zip(batch_indices, batch_slots)
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


def merge_export_chunks(
    config: RandomWindowExportConfig, *, state: dict[str, Any] | None = None
) -> None:
    """Merge chunk files into final .npy outputs."""
    _, chunks_dir, state_path, _, _, _ = _paths(config)

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
        parts.append({k: data[k] for k in ("sims", "mask", "day", "hour", "latlon", "windows")})

    if not parts:
        raise RuntimeError("No valid samples exported (all failed mask filter).")

    data_sims = np.concatenate([p["sims"] for p in parts], axis=0)
    data_mask = np.concatenate([p["mask"] for p in parts], axis=0)
    data_day = np.concatenate([p["day"] for p in parts], axis=0)
    data_hour = np.concatenate([p["hour"] for p in parts], axis=0)
    data_latlon = np.concatenate([p["latlon"] for p in parts], axis=0)
    data_windows = np.concatenate([p["windows"] for p in parts], axis=0)

    # Sentinel for legacy loaders: random mode (top/left vary per sample; see _windows.npy).
    window_meta = np.array([-1.0, -1.0, config.crop_height, config.crop_width, 0.0], dtype=np.float64)

    np.save(f"{config.output_prefix}_sims.npy", data_sims)
    np.save(f"{config.output_prefix}_mask.npy", data_mask)
    np.save(f"{config.output_prefix}_day.npy", data_day)
    np.save(f"{config.output_prefix}_hour.npy", data_hour)
    np.save(f"{config.output_prefix}_latlon.npy", data_latlon)
    np.save(f"{config.output_prefix}_windows.npy", data_windows)
    np.save(f"{config.output_prefix}_window.npy", window_meta)

    state["status"] = "finished"
    state["n_valid"] = int(data_sims.shape[0])
    state["finished_at"] = datetime.now().isoformat(timespec="seconds")
    _save_state(state_path, state)

    target_names = [config.channel_names[i] for i in config.target_channel_indices]
    land_fracs = data_mask.mean(axis=(1, 2))

    print("Saved:")
    print(f"  {config.output_prefix}_sims.npy     {data_sims.shape}")
    print(f"  {config.output_prefix}_mask.npy     {data_mask.shape}")
    print(f"  {config.output_prefix}_windows.npy  {data_windows.shape}  [top, left, h, w] per sample")
    print(f"  {config.output_prefix}_window.npy   {window_meta}  (random-window sentinel)")
    print(f"  channels: {list(config.channel_names)} | target (GT/pred): {target_names}")
    if data_sims.shape[0]:
        print(f"  exported hours: {sorted(set(int(h) for h in data_hour))}")
        print(
            f"  land fraction: mean={land_fracs.mean():.2%} "
            f"min={land_fracs.min():.2%} max={land_fracs.max():.2%}"
        )

    if config.export_epoch_windows:
        export_train_epoch_windows(config)


def _pool_rows_for_train_crops(
    prefix: str,
    train_crops_per_anchor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load pool arrays filtered to crop_slot < K."""
    p = Path(prefix)
    indices = np.load(f"{p}_indices.npy")
    day = np.load(f"{p}_day.npy")
    hour = np.load(f"{p}_hour.npy")
    slots_path = Path(f"{p}_crop_slots.npy")
    if slots_path.exists():
        keep = np.load(slots_path) < int(train_crops_per_anchor)
        indices = indices[keep]
        day = day[keep]
        hour = hour[keep]
    return indices, day, hour, np.arange(len(indices), dtype=np.int64)


def _sort_pool_chronologically(
    indices: np.ndarray,
    day: np.ndarray,
    hour: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.lexsort((np.asarray(hour), np.asarray(day)))
    return indices[order], day[order], hour[order]


def _export_epoch_windows_for_k(
    dataset,
    config: RandomWindowExportConfig,
    *,
    train_crops_per_anchor: int,
) -> dict[str, Any]:
    """Pre-compute train crop windows for epochs 1..n_export_epochs (matches training RNG)."""
    indices, day, hour, _ = _pool_rows_for_train_crops(
        config.output_prefix, train_crops_per_anchor
    )
    if config.val_split_mode == "chronological":
        indices, day, hour = _sort_pool_chronologically(indices, day, hour)
    n_pool = len(indices)
    train_idx, val_idx = split_train_val_indices(
        n_pool,
        config.val_split,
        mode=config.val_split_mode,
        seed=config.val_split_seed,
    )
    tr_zidx = indices[train_idx]
    n_train = len(tr_zidx)
    n_epochs = int(config.n_export_epochs)
    epoch_windows = np.zeros((n_epochs, n_train, 4), dtype=np.float64)

    def _one(pool_idx: int, epoch: int) -> tuple[int, int, np.ndarray]:
        ds_idx = int(tr_zidx[pool_idx])
        win = find_valid_crop_window(
            dataset,
            ds_idx,
            pool_idx,
            epoch,
            crop_seed=config.random_crop_seed,
            crop_h=config.crop_height,
            crop_w=config.crop_width,
            max_mask_frac=config.max_mask_frac,
            max_crop_attempts=config.max_crop_attempts,
            channel_names=config.channel_names,
        )
        return epoch - 1, pool_idx, win

    tasks = [(p, e) for e in range(1, n_epochs + 1) for p in range(n_train)]
    print(
        f"  k={train_crops_per_anchor}: {n_train} train rows × {n_epochs} epochs "
        f"({len(tasks)} crops, {config.val_split_mode} split)",
        flush=True,
    )
    results = Parallel(n_jobs=config.n_jobs, verbose=1)(
        delayed(_one)(p, e) for p, e in tasks
    )
    for ep_i, pool_i, win in results:
        epoch_windows[ep_i, pool_i] = win

    tag = f"k{int(train_crops_per_anchor)}"
    out_path = Path(f"{config.output_prefix}_windows_per_epoch_{tag}.npy")
    np.save(out_path, epoch_windows)
    return {
        "path": str(out_path),
        "shape": list(epoch_windows.shape),
        "train_crops_per_anchor": int(train_crops_per_anchor),
        "n_train": int(n_train),
        "n_val": int(len(val_idx)),
        "n_pool": int(n_pool),
        "epochs": list(range(1, n_epochs + 1)),
    }


def _maybe_export_epoch_windows(config: RandomWindowExportConfig) -> None:
    """Run epoch-window export when enabled and outputs are missing."""
    if not config.export_epoch_windows:
        return
    k_values = tuple(int(k) for k in config.train_crops_for_epoch_windows)
    missing = any(
        not Path(f"{config.output_prefix}_windows_per_epoch_k{k}.npy").exists()
        for k in k_values
    )
    if missing:
        export_train_epoch_windows(config)


def export_train_epoch_windows(config: RandomWindowExportConfig) -> None:
    """Write ``_windows_per_epoch_k1.npy`` / ``_k2.npy`` for fast training-time lookup."""
    prefix = Path(config.output_prefix)
    meta_path = Path(f"{config.output_prefix}_epoch_windows_meta.json")
    if not Path(f"{prefix}_indices.npy").exists():
        print("Skipping epoch-window export: pool indices not found", flush=True)
        return

    print(
        f"Exporting per-epoch train windows ({config.n_export_epochs} epochs, "
        f"seed={config.random_crop_seed}, split={config.val_split_mode}) ...",
        flush=True,
    )
    dataset = _build_dataset(config)
    k_values = tuple(int(k) for k in config.train_crops_for_epoch_windows)
    entries = []
    for k in k_values:
        print(f"Epoch windows for train_crops_per_anchor={k}:", flush=True)
        entries.append(_export_epoch_windows_for_k(dataset, config, train_crops_per_anchor=k))

    meta = {
        "random_crop_seed": int(config.random_crop_seed),
        "n_export_epochs": int(config.n_export_epochs),
        "val_split": float(config.val_split),
        "val_split_mode": config.val_split_mode,
        "val_split_seed": int(config.val_split_seed),
        "max_mask_frac": float(config.max_mask_frac),
        "epoch_origin": "1-based (matches training set_epoch)",
        "rng": "[random_crop_seed, epoch, zarr_idx, train_pool_idx]",
        "variants": entries,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Saved epoch-window metadata: {meta_path}", flush=True)
