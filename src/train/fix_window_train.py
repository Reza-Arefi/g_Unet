"""Resumable fix-window Unet2D training."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gUnet import Unet2D
from src.train.losses import masked_loss, my_loss as myLoss


@dataclass
class TrainConfig:
    data_prefix: str = "data/sample_10_19_fixwin_dt96"
    val_data_prefix: str | None = None  # separate val export; skips in-array train/val split
    merge_val_export_prefix: str | None = None  # merge with data_prefix, sort by time, then split
    dt_hours: int = 96  # 96 = 4-day steps; use 24 for daily
    epochs: int = 100
    batch_size: int = 4
    lr: float = 2e-4
    weight_decay: float = 0.0  # L2 penalty (e.g. 1e-4); use with AdamW for decoupled decay
    optimizer: str = "adam"  # "adam" | "adamw"
    iD: int = 3
    device: str = "cuda:7"
    checkpoint_dir: str = "Models/fix_window"
    run_name: str = "fix_window_10_19_dt96"
    best_model_path: str = "Models/best_model_FixWindow1.pth"
    val_split: float = 0.3
    val_split_mode: str = "chronological"  # "chronological" | "random" (export split uses val_data_prefix)
    val_split_seed: int = 42  # used when val_split_mode == "random"
    loss_mask_buffer_px: int = 0  # dilate land in loss mask (coastal magic-window strip)
    filter_fully_ocean_samples: bool = False  # guiana-style: drop sequences with any land
    max_samples: int | None = None  # use first N sequences (export order); None = all
    dynamic_random_crop: bool = False  # train: new Zarr crop each epoch (random-window exports)
    random_crop_seed: int = 42
    max_mask_frac: float = 0.05  # reject crops with masked fraction >= this (dynamic mode)
    max_crop_attempts: int = 200
    train_crops_per_anchor: int | None = None  # e.g. 1 = first crop only; 2 = all K export crops
    preview_val_mask: bool = True  # save val_mask_preview.png before epoch 1
    unet_dim: int = 32
    unet_dim_mults: tuple[int, ...] = (1, 2, 4, 8)
    num_workers: int = 0  # DataLoader workers (Zarr dynamic crop: use many CPU cores)
    pin_memory: bool = False  # faster host→GPU copies when training on CUDA
    use_multi_gpu: bool = False  # nn.DataParallel on all visible CUDA devices

    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @property
    def dt_step_days(self) -> float:
        return self.dt_hours / 24.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        allowed = {f.name for f in fields(cls)}
        values = {k: v for k, v in data.items() if k in allowed}
        if "unet_dim_mults" in values and isinstance(values["unet_dim_mults"], list):
            values["unet_dim_mults"] = tuple(values["unet_dim_mults"])
        return cls(**values)


def split_train_val_indices(
    n: int,
    val_split: float,
    *,
    mode: str = "chronological",
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx) indexing into n exported sequences."""
    if n <= 0:
        raise ValueError("split_train_val_indices requires n > 0")
    n_val = max(1, int(n * val_split)) if n > 1 else 0
    if n_val >= n:
        n_val = max(1, n // 5)
    n_train = n - n_val

    if mode == "chronological":
        train_idx = np.arange(n_train, dtype=np.int64)
        val_idx = np.arange(n_train, n, dtype=np.int64)
    elif mode == "random":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        train_idx = np.sort(perm[:n_train])
        val_idx = np.sort(perm[n_train:])
    else:
        raise ValueError(
            f"Unknown val_split_mode {mode!r}; use 'chronological' or 'random'"
        )
    return train_idx, val_idx


def resolve_train_val_indices(
    config: TrainConfig,
    n_train: int,
    *,
    n_val: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx) for logging/preview; matches prepare_train_val_tensors."""
    if config.val_data_prefix:
        if n_val is None:
            raise ValueError("n_val is required when val_data_prefix is set")
        return (
            np.arange(n_train, dtype=np.int64),
            np.arange(n_val, dtype=np.int64),
        )
    return split_train_val_indices(
        n_train,
        config.val_split,
        mode=config.val_split_mode,
        seed=config.val_split_seed,
    )


def _index_tensors(
    data: torch.Tensor,
    mask: torch.Tensor,
    day: torch.Tensor,
    hour: torch.Tensor,
    indices: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    idx = torch.as_tensor(indices, dtype=torch.long)
    return data[idx], mask[idx], day[idx], hour[idx]


def _global_channel_stats(
    data: torch.Tensor,
    frame_indices: list[int],
    iD: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Guiana-style per-channel mean/std over all spatial pixels (mask not used in norm)."""
    frames = [data[:, t, :iD].float() for t in frame_indices]
    mean = sum(f.mean(dim=(0, 2, 3), keepdim=True) for f in frames) / float(len(frames))
    std = torch.cat(frames, dim=0).std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-8)
    return mean, std


class SimulationDataset(Dataset):
    """Time-series samples for autoregressive +1d increment training."""

    def __init__(
        self,
        data: torch.Tensor,
        mask: torch.Tensor,
        day: torch.Tensor,
        hour: torch.Tensor,
        mean=None,
        std=None,
        iD: int = 3,
    ):
        self.iD = iD
        n_frames = int(data.shape[1])

        if n_frames <= 4:
            # Compact export: 3 conditioning + 1 forecast (e.g. sample_13_22_*_dt360)
            cond_idx = list(range(n_frames - 1))
            anchor_idx = n_frames - 2
            forecast_idx = n_frames - 1
            self.x = torch.cat(
                [data[:, t, :iD, :, :] for t in cond_idx],
                dim=1,
            ).float()
            self.y = (
                data[:, forecast_idx, :iD, :, :] - data[:, anchor_idx, :iD, :, :]
            ).float()
            n_x_frames = len(cond_idx)
            n_y_steps = 1
        else:
            # Legacy long export: 4 frames in x (model uses last 3), 3 increment targets
            self.x = torch.cat(
                [
                    data[:, 0, :iD, :, :],
                    data[:, 1, :iD, :, :],
                    data[:, 2, :iD, :, :],
                    data[:, 3, :iD, :, :],
                ],
                dim=1,
            ).float()
            self.y = torch.cat(
                [
                    (data[:, 4, :iD, :, :] - data[:, 3, :iD, :, :]),
                    (data[:, 5, :iD, :, :] - data[:, 4, :iD, :, :]),
                    (data[:, 6, :iD, :, :] - data[:, 5, :iD, :, :]),
                ],
                dim=1,
            ).float()
            n_x_frames = 4
            n_y_steps = 3

        self.mask = mask

        if n_frames <= 4:
            stat_frames = list(range(n_frames - 1))
        else:
            stat_frames = [0, 1, 2, 3]

        if mean is None:
            self.mean, self.std = _global_channel_stats(data, stat_frames, iD)
        else:
            self.mean = mean
            self.std = std

        mean_x = self.mean.repeat(1, n_x_frames, 1, 1)
        std_x = self.std.repeat(1, n_x_frames, 1, 1)
        std_y = self.std.repeat(1, n_y_steps, 1, 1)

        self.x = (self.x - mean_x) / (std_x + 1e-8)
        self.y = self.y / (std_y + 1e-8)

        # Zero land (mask == 1); model sees zeros there, loss ignores them (Guiana pattern).
        ocean = (mask == 0).to(dtype=self.x.dtype).unsqueeze(1)
        self.x = self.x * ocean
        self.y = self.y * ocean

        self.day = day.float() / 365
        self.hour = hour.float() / 365

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.mask[idx], self.day[idx], self.hour[idx]


def create_spatial_weight(h=128, w=128, sigma=32):
    y = torch.arange(h).float()
    x = torch.arange(w).float()
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    cy = (h - 1) / 2
    cx = (w - 1) / 2
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    weight = torch.exp(-dist2 / (2 * sigma**2))
    return weight / weight.mean()


def row_keep_mask_for_train_crops(
    prefix: str,
    train_crops_per_anchor: int | None,
) -> np.ndarray | None:
    """Keep export rows with crop_slot < train_crops_per_anchor (uses ``_crop_slots.npy``)."""
    if train_crops_per_anchor is None:
        return None
    k = int(train_crops_per_anchor)
    if k <= 0:
        raise ValueError("train_crops_per_anchor must be positive when set")
    slots_path = Path(f"{prefix}_crop_slots.npy")
    if not slots_path.exists():
        return None
    slots = np.load(slots_path)
    return np.asarray(slots) < k


def load_training_arrays(config: TrainConfig, *, data_prefix: str | None = None):
    prefix = data_prefix or config.data_prefix
    sims_np = np.load(f"{prefix}_sims.npy", mmap_mode="r")
    mask_np = np.load(f"{prefix}_mask.npy")
    day_np = np.load(f"{prefix}_day.npy")
    hour_np = np.load(f"{prefix}_hour.npy")

    if config.filter_fully_ocean_samples:
        keep = np.all(mask_np == 0, axis=(1, 2))
        n_before = len(mask_np)
        sims_np = sims_np[keep]
        mask_np = mask_np[keep]
        day_np = day_np[keep]
        hour_np = hour_np[keep]
        print(
            f"Guiana-style ocean-only filter: kept {int(keep.sum())} / {n_before} sequences "
            f"(dropped {n_before - int(keep.sum())} with any land pixel)",
            flush=True,
        )
        if len(mask_np) == 0:
            raise ValueError(
                "filter_fully_ocean_samples removed all sequences — export has land on every crop. "
                "Use an ocean-only window or set filter_fully_ocean_samples=False for coastal training."
            )

    n_loaded = len(sims_np)
    crop_keep = row_keep_mask_for_train_crops(prefix, config.train_crops_per_anchor)
    if crop_keep is not None:
        n_before = n_loaded
        sims_np = sims_np[crop_keep]
        mask_np = mask_np[crop_keep]
        day_np = day_np[crop_keep]
        hour_np = hour_np[crop_keep]
        n_loaded = len(sims_np)
        print(
            f"Train crops per anchor: using {n_loaded} / {n_before} export rows "
            f"(crop slots < {int(config.train_crops_per_anchor)})",
            flush=True,
        )

    if config.max_samples is not None and int(config.max_samples) > 0:
        n_cap = min(int(config.max_samples), n_loaded)
        if n_cap < n_loaded:
            print(
                f"Sequence cap: using first {n_cap} / {n_loaded} "
                f"(max_samples={config.max_samples}, export order)",
                flush=True,
            )
        sims_np = sims_np[:n_cap]
        mask_np = mask_np[:n_cap]
        day_np = day_np[:n_cap]
        hour_np = hour_np[:n_cap]

    # Export already drops high-land samples (max_mask_frac). Coastal windows can
    # still have partial land; loss uses (mask == 0) per-pixel weights.
    crop_h, crop_w = sims_np.shape[-2], sims_np.shape[-1]
    if len(mask_np) > 1 and np.all(mask_np == mask_np[0:1]):
        print(
            f"Static land mask on all {len(mask_np)} sequences "
            f"({float(mask_np[0].mean()):.2%} land) — Guiana-style per-pixel masking",
            flush=True,
        )
    data = torch.from_numpy(np.asarray(sims_np))
    mask = torch.from_numpy(mask_np)
    day = torch.from_numpy(day_np)
    hour = torch.from_numpy(hour_np)
    return data, mask, day, hour, crop_h, crop_w


def load_merged_training_arrays(config: TrainConfig) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int
]:
    """Load train + val exports, sort by (day, hour), optionally cap with max_samples."""
    val_prefix = config.merge_val_export_prefix
    if not val_prefix:
        raise ValueError("merge_val_export_prefix must be set for merged loading")
    base_cfg = replace(config, max_samples=None)
    data_parts, mask_parts, day_parts, hour_parts = [], [], [], []
    crop_h = crop_w = None
    for label, prefix in (("train", config.data_prefix), ("val", val_prefix)):
        part_data, part_mask, part_day, part_hour, part_h, part_w = load_training_arrays(
            base_cfg, data_prefix=prefix
        )
        if crop_h is None:
            crop_h, crop_w = part_h, part_w
        elif (part_h, part_w) != (crop_h, crop_w):
            raise ValueError(
                f"Crop size mismatch between exports: {config.data_prefix} vs {val_prefix}"
            )
        data_parts.append(part_data)
        mask_parts.append(part_mask)
        day_parts.append(part_day)
        hour_parts.append(part_hour)
        print(f"  merged {label}: {len(part_data)} sequences from {prefix}", flush=True)

    data = torch.cat(data_parts, dim=0)
    mask = torch.cat(mask_parts, dim=0)
    day = torch.cat(day_parts, dim=0)
    hour = torch.cat(hour_parts, dim=0)
    order = np.lexsort((hour.numpy(), day.numpy()))
    data = data[torch.as_tensor(order, dtype=torch.long)]
    mask = mask[torch.as_tensor(order, dtype=torch.long)]
    day = day[torch.as_tensor(order, dtype=torch.long)]
    hour = hour[torch.as_tensor(order, dtype=torch.long)]

    n_loaded = len(data)
    if config.max_samples is not None and int(config.max_samples) > 0:
        n_cap = min(int(config.max_samples), n_loaded)
        if n_cap < n_loaded:
            print(
                f"Sequence cap: using first {n_cap} / {n_loaded} "
                f"(max_samples={config.max_samples}, chronological order)",
                flush=True,
            )
        data = data[:n_cap]
        mask = mask[:n_cap]
        day = day[:n_cap]
        hour = hour[:n_cap]

    print(
        f"Merged train+val exports: {len(data)} sequences "
        f"({config.data_prefix} + {val_prefix})",
        flush=True,
    )
    return data, mask, day, hour, int(crop_h), int(crop_w)


def prepare_train_val_tensors(
    data: torch.Tensor,
    mask: torch.Tensor,
    day: torch.Tensor,
    hour: torch.Tensor,
    config: TrainConfig,
    *,
    val_data: torch.Tensor | None = None,
    val_mask: torch.Tensor | None = None,
    val_day: torch.Tensor | None = None,
    val_hour: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    np.ndarray,
    np.ndarray,
]:
    """Split train/val in-memory or use pre-exported val arrays."""
    if config.val_data_prefix:
        if val_data is None or val_mask is None or val_day is None or val_hour is None:
            raise ValueError("val_data_prefix set but val tensors were not loaded")
        n_train = len(data)
        n_val = len(val_data)
        if n_train == 0 or n_val == 0:
            raise ValueError("train and val exports must each contain at least one sequence")
        train_idx = np.arange(n_train, dtype=np.int64)
        val_idx = np.arange(n_val, dtype=np.int64)
        print(
            f"Train/val from separate exports: {n_train} train ({config.data_prefix}), "
            f"{n_val} val ({config.val_data_prefix})",
            flush=True,
        )
        return (
            data,
            mask,
            day,
            hour,
            val_data,
            val_mask,
            val_day,
            val_hour,
            train_idx,
            val_idx,
        )

    n = len(data)
    if config.val_split_mode == "chronological" and not config.val_data_prefix:
        order = np.lexsort((hour.numpy(), day.numpy()))
        idx_t = torch.as_tensor(order, dtype=torch.long)
        data = data[idx_t]
        mask = mask[idx_t]
        day = day[idx_t]
        hour = hour[idx_t]
        print(
            f"Chronological ordering: sorted {n} sequences by (day, hour) before split",
            flush=True,
        )
    train_idx, val_idx = split_train_val_indices(
        n,
        config.val_split,
        mode=config.val_split_mode,
        seed=config.val_split_seed,
    )
    train_data, train_mask, train_day, train_hour = _index_tensors(
        data, mask, day, hour, train_idx
    )
    val_data, val_mask, val_day, val_hour = _index_tensors(
        data, mask, day, hour, val_idx
    )
    print(
        f"Train/val split ({config.val_split_mode}, seed={config.val_split_seed}): "
        f"{len(train_data)} train sequences, {len(val_data)} val sequences "
        f"({config.val_split:.0%} val, {n} total after mask filter)",
        flush=True,
    )
    return (
        train_data,
        train_mask,
        train_day,
        train_hour,
        val_data,
        val_mask,
        val_day,
        val_hour,
        train_idx,
        val_idx,
    )


def build_optimizer(
    params,
    *,
    lr: float,
    weight_decay: float = 0.0,
    optimizer: str = "adam",
) -> torch.optim.Optimizer:
    """Build Adam or AdamW. Prefer AdamW when ``weight_decay > 0`` (decoupled L2)."""
    name = optimizer.strip().lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer {optimizer!r}; use 'adam' or 'adamw'")


def build_model(crop_h: int, crop_w: int, config: TrainConfig) -> Unet2D:
    par = {
        "lb": 3,
        "nf": config.iD,
        "nx": crop_h,
        "ny": crop_w,
        "inp_shift": 0.0,
        "inp_scale": 1.0,
        "out_shift": 0.0,
        "out_scale": 1.0,
        "t_shift": 0.0,
        "t_scale": 1.0,
    }
    return Unet2D(
        dim=config.unet_dim,
        Par=par,
        dim_mults=config.unet_dim_mults,
        channels=par["lb"] * par["nf"],
    )


def _paths(config: TrainConfig):
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_dir = checkpoint_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    config_path = checkpoint_dir / f"train_config_{config.run_name}.json"
    state_path = checkpoint_dir / f"train_state_{config.run_name}.json"
    history_path = history_dir / f"training_history_{config.run_name}.pt"
    resume_path = checkpoint_dir / f"resume_{config.run_name}.pt"
    return config_path, state_path, history_path, resume_path


def _save_train_state(state_path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    state_path.write_text(json.dumps(state, indent=2))


def _atomic_torch_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _try_resume(
    config: TrainConfig,
    model: nn.Module,
    optimizer,
    scheduler,
    device: str,
    resume_path: Path,
    history_path: Path,
):
    start_epoch = 1
    hist = {
        "epoch": [],
        "train": [],
        "val": [],
        "val1": [],
        "val2": [],
        "val3": [],
        "lr": [],
        "best_val_loss": float("inf"),
        "best_epoch": 0,
    }
    best_val_loss = float("inf")
    best_epoch = 0

    if resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device)
        if ckpt.get("config_hash") != config.config_hash():
            raise RuntimeError(
                f"Training config changed since last run. Delete {resume_path} or use --no-resume."
            )
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        hist = ckpt["hist"]
        best_val_loss = ckpt["best_val_loss"]
        best_epoch = ckpt["best_epoch"]
        start_epoch = int(ckpt["epoch"]) + 1
        print(f"Resuming from epoch {start_epoch} (checkpoint {resume_path})", flush=True)
        return start_epoch, hist, best_val_loss, best_epoch

    if history_path.exists():
        prev = torch.load(history_path, map_location="cpu")
        if prev.get("epoch"):
            last_model = Path(config.checkpoint_dir) / f"last_model_{config.run_name}.pth"
            if last_model.exists():
                model.load_state_dict(torch.load(last_model, map_location=device))
                hist = prev
                best_val_loss = float(prev.get("best_val_loss", float("inf")))
                best_epoch = int(prev.get("best_epoch", 0))
                start_epoch = int(prev["epoch"][-1]) + 1
                print(
                    f"Resuming from epoch {start_epoch} using history + last model "
                    f"(optimizer reset)",
                    flush=True,
                )
                return start_epoch, hist, best_val_loss, best_epoch

    return start_epoch, hist, best_val_loss, best_epoch


def train(
    model: nn.Module,
    data: torch.Tensor,
    mask: torch.Tensor,
    day: torch.Tensor,
    hour: torch.Tensor,
    config: TrainConfig | None = None,
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    val_split: float | None = None,
    device: str | None = None,
    iD: int | None = None,
    checkpoint_dir: str | None = None,
    run_name: str | None = None,
    resume: bool = True,
) -> nn.Module:
    """Train with per-epoch checkpointing; safe to stop and resume."""
    if config is None:
        config = TrainConfig()
    if epochs is not None:
        config.epochs = epochs
    if batch_size is not None:
        config.batch_size = batch_size
    if lr is not None:
        config.lr = lr
    if val_split is not None:
        config.val_split = val_split
    if device is not None:
        config.device = device
    if iD is not None:
        config.iD = iD
    if checkpoint_dir is not None:
        config.checkpoint_dir = checkpoint_dir
    if run_name is not None:
        config.run_name = run_name

    device = config.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)
    model = model.to(device)

    n = len(data)
    (
        train_data,
        train_mask,
        train_day,
        train_hour,
        val_data,
        val_mask,
        val_day,
        val_hour,
        train_idx,
        val_idx,
    ) = prepare_train_val_tensors(data, mask, day, hour, config)

    train_size = len(train_data)
    val_size = len(val_data)

    train_set = SimulationDataset(train_data, train_mask, train_day, train_hour, iD=config.iD)
    val_set = SimulationDataset(
        val_data, val_mask, val_day, val_hour, mean=train_set.mean, std=train_set.std, iD=config.iD
    )

    config_path, state_path, history_path, resume_path = _paths(config)
    config_path.write_text(json.dumps(asdict(config), indent=2, default=str))

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)

    h, w = data.shape[-2], data.shape[-1]
    scale = max(h, w) / 128.0
    spatial_weight = create_spatial_weight(h=h, w=w, sigma=96 * scale)

    optimizer = build_optimizer(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        optimizer=config.optimizer,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    if resume:
        start_epoch, hist, best_val_loss, best_epoch = _try_resume(
            config, model, optimizer, scheduler, device, resume_path, history_path
        )
    else:
        start_epoch = 1
        hist = {
            "epoch": [],
            "train": [],
            "val": [],
            "val1": [],
            "val2": [],
            "val3": [],
            "lr": [],
            "best_val_loss": float("inf"),
            "best_epoch": 0,
        }
        best_val_loss = float("inf")
        best_epoch = 0
        if resume_path.exists():
            resume_path.unlink()

    if start_epoch == 1:
        norm_stats = {"mean": train_set.mean, "std": train_set.std}
        torch.save(norm_stats, Path(config.checkpoint_dir) / "norm_stats.pt")
        torch.save(norm_stats, "Models/norm_stats.pt")

    run_info = {
        "run_name": config.run_name,
        "data_prefix": config.data_prefix,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "epochs_planned": config.epochs,
        "batch_size": config.batch_size,
        "lr": config.lr,
        "iD": config.iD,
        "train_size": train_size,
        "val_size": val_size,
        "device": str(device),
        "checkpoint_dir": config.checkpoint_dir,
        "config_hash": config.config_hash(),
    }
    torch.save(run_info, Path(config.checkpoint_dir) / "run_info.pt")

    train_state = {
        "status": "running",
        "run_name": config.run_name,
        "config_hash": config.config_hash(),
        "epochs_planned": config.epochs,
        "last_completed_epoch": 0,
    }
    _save_train_state(state_path, train_state)

    print(f"Checkpoints: {config.checkpoint_dir}", flush=True)
    print(f"History:     {history_path}", flush=True)
    print(f"Resume:      {resume_path}", flush=True)

    if start_epoch > config.epochs:
        print(f"Training already complete ({start_epoch - 1}/{config.epochs} epochs).", flush=True)
        train_state["status"] = "finished"
        train_state["last_completed_epoch"] = config.epochs
        _save_train_state(state_path, train_state)
        return model

    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y, m, d, h_batch in train_loader:
            x, y, m, d, h_batch = (
                x.to(device),
                y.to(device),
                m.to(device),
                d.to(device),
                h_batch.to(device),
            )
            optimizer.zero_grad()
            pred = model(x[:, -3 * config.iD :], d + h_batch / 365.0)
            loss = myLoss(pred, y[:, : config.iD], m, spatial_weight)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= train_size

        model.eval()
        val_loss = val_loss1 = val_loss2 = val_loss3 = 0.0
        with torch.no_grad():
            for x, y, m, d, h_batch in val_loader:
                x, y, m, d, h_batch = (
                    x.to(device),
                    y.to(device),
                    m.to(device),
                    d.to(device),
                    h_batch.to(device),
                )
                pred1 = model(x[:, -3 * config.iD :], d + h_batch / 365.0)
                val_loss1 += myLoss(pred1, y[:, : config.iD], m, spatial_weight).item() * x.size(0)

        val_loss1 /= val_size
        val_loss2 /= val_size
        val_loss3 /= val_size
        scheduler.step(val_loss1)
        val_loss = val_loss1 + 0.5 * val_loss2 + 0.25 * val_loss3
        current_lr = optimizer.param_groups[0]["lr"]

        hist["epoch"].append(epoch)
        hist["train"].append(train_loss)
        hist["val"].append(val_loss)
        hist["val1"].append(val_loss1)
        hist["val2"].append(val_loss2)
        hist["val3"].append(val_loss3)
        hist["lr"].append(current_lr)
        hist["best_val_loss"] = best_val_loss
        hist["best_epoch"] = best_epoch
        _atomic_torch_save(hist, history_path)

        if val_loss < best_val_loss:
            print(
                f"epoch {epoch} | val loss {val_loss:.6f} | > "
                f"(improvement of {best_val_loss - val_loss:.6f})",
                flush=True,
            )
            best_val_loss = val_loss
            best_epoch = epoch
            hist["best_val_loss"] = best_val_loss
            hist["best_epoch"] = best_epoch
            best_model_path = Path(config.checkpoint_dir) / f"best_model_{config.run_name}.pth"
            torch.save(model.state_dict(), best_model_path)
            torch.save(model.state_dict(), config.best_model_path)
            _atomic_torch_save(hist, history_path)

        last_model_path = Path(config.checkpoint_dir) / f"last_model_{config.run_name}.pth"
        torch.save(model.state_dict(), last_model_path)

        _atomic_torch_save(
            {
                "epoch": epoch,
                "config_hash": config.config_hash(),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "hist": hist,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
            },
            resume_path,
        )

        train_state["last_completed_epoch"] = epoch
        _save_train_state(state_path, train_state)

        print(
            f"Epoch {epoch:04d} | Train Loss: {train_loss:.6f} | "
            f"Val Loss:{val_loss:.6f} > {val_loss1:.6f}, {val_loss2:.6f}, {val_loss3:.6f} | "
            f"LR: {current_lr:.2e}",
            flush=True,
        )

    run_info["finished_at"] = datetime.now().isoformat(timespec="seconds")
    run_info["epochs_completed"] = len(hist["epoch"])
    run_info["best_val_loss"] = best_val_loss
    run_info["best_epoch"] = best_epoch
    torch.save(run_info, Path(config.checkpoint_dir) / "run_info.pt")

    train_state["status"] = "finished"
    train_state["last_completed_epoch"] = config.epochs
    train_state["best_val_loss"] = best_val_loss
    train_state["best_epoch"] = best_epoch
    _save_train_state(state_path, train_state)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f} (epoch {best_epoch})", flush=True)
    print(f"Saved history to {history_path}", flush=True)
    return model


def run_training(config: TrainConfig, *, resume: bool = True) -> nn.Module:
    data, mask, day, hour, crop_h, crop_w = load_training_arrays(config)
    print(f"Dataset: {config.data_prefix}", flush=True)
    print(f"Samples: {len(data)} | crop {crop_h} x {crop_w}", flush=True)
    model = build_model(crop_h, crop_w, config)
    return train(model, data, mask, day, hour, config=config, resume=resume)
