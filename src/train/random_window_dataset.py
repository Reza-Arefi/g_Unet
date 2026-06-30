"""Training-time random-window crops from Guiana Zarr (new crop each epoch on train)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.dataloaders.guiana_margem_tensor import GuianaDataloader
from src.export.random_window_export import (
    RandomWindowExportConfig,
    _grid_size,
    find_valid_crop_window,
    load_sequence_at_window,
)
from src.train.fix_window_train import _global_channel_stats


def epoch_windows_path(prefix: str, train_crops_per_anchor: int) -> Path:
    return Path(f"{prefix}_windows_per_epoch_k{int(train_crops_per_anchor)}.npy")


def load_epoch_windows(prefix: str, train_crops_per_anchor: int) -> np.ndarray | None:
    """Load pre-exported train windows ``(n_epochs, n_train, 4)`` if present."""
    path = epoch_windows_path(prefix, train_crops_per_anchor)
    if not path.exists():
        return None
    return np.load(path)


def filter_pool_dict(pool: dict, keep: np.ndarray) -> dict:
    """Subset pool row arrays."""
    keep = np.asarray(keep, dtype=bool)
    out = dict(pool)
    for key in ("indices", "day", "hour", "windows"):
        if out.get(key) is not None:
            out[key] = np.asarray(out[key])[keep]
    return out


def load_random_window_pool(prefix: str) -> dict:
    """Load export indices, time metadata, optional windows, and crop size."""
    p = Path(prefix)
    indices = np.load(f"{p}_indices.npy")
    day = np.load(f"{p}_day.npy")
    hour = np.load(f"{p}_hour.npy")
    cfg_path = Path(f"{p}_export_config.json")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing export config: {cfg_path}")
    cfg_dict = json.loads(cfg_path.read_text())
    crop_h = int(cfg_dict.get("crop_height", 128))
    crop_w = int(cfg_dict.get("crop_width", 128))
    windows_path = Path(f"{p}_windows.npy")
    windows = np.load(windows_path) if windows_path.exists() else None
    return {
        "indices": indices,
        "day": day,
        "hour": hour,
        "windows": windows,
        "crop_h": crop_h,
        "crop_w": crop_w,
        "export_config": RandomWindowExportConfig.from_dict(cfg_dict),
    }


def sort_pool_chronologically(
    indices: np.ndarray,
    day: np.ndarray,
    hour: np.ndarray,
    windows: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Sort pool rows by (day, hour) for chronological train/val splits."""
    order = np.lexsort((np.asarray(hour), np.asarray(day)))
    indices = np.asarray(indices)[order]
    day = np.asarray(day)[order]
    hour = np.asarray(hour)[order]
    if windows is not None:
        windows = np.asarray(windows)[order]
    return indices, day, hour, windows


def norm_stats_from_exported_sims(
    data_prefix: str,
    train_idx: np.ndarray,
    *,
    iD: int = 2,
    n_frames: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel mean/std from exported sims on the train split (cond frames only)."""
    sims = np.load(f"{data_prefix}_sims.npy", mmap_mode="r")
    idx = np.asarray(train_idx, dtype=np.int64)
    data = torch.from_numpy(np.asarray(sims[idx]))
    stat_frames = list(range(n_frames - 1)) if n_frames <= 4 else [0, 1, 2, 3]
    return _global_channel_stats(data, stat_frames, iD)


def _build_zarr_dataset(cfg: RandomWindowExportConfig) -> GuianaDataloader:
    return GuianaDataloader(
        path=cfg.zarr_path,
        dt=cfg.dt,
        conditioning_steps=cfg.conditioning_steps,
        forecast_steps=cfg.forecast_steps,
        transform=None,
        device="cpu",
    )


def _compact_xy_from_seq(
    seq: np.ndarray,
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    iD: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build normalized (x, y) increment tensors from raw [T,C,H,W] sequence."""
    data = torch.from_numpy(seq[:, :iD].astype(np.float32, copy=False))
    n_frames = int(data.shape[0])
    cond_idx = list(range(n_frames - 1))
    anchor_idx = n_frames - 2
    forecast_idx = n_frames - 1
    x = torch.cat([data[t] for t in cond_idx], dim=0).float()
    y = (data[forecast_idx] - data[anchor_idx]).float()
    n_x_frames = len(cond_idx)
    mean_x = mean.repeat(1, n_x_frames, 1, 1)
    std_x = std.repeat(1, n_x_frames, 1, 1)
    std_y = std.repeat(1, 1, 1, 1)
    x = (x.unsqueeze(0) - mean_x) / (std_x + 1e-8)
    y = y.unsqueeze(0) / (std_y + 1e-8)
    return x.squeeze(0), y.squeeze(0)


class RandomWindowZarrDataset(Dataset):
    """
    Load sequences from Zarr with random 128×128 crops.

    Train: pre-exported ``epoch_windows[epoch-1]`` when provided; else new random crop
    each epoch (``set_epoch`` must be called each epoch).
    Val: fixed ``(top, left)`` per sample from export ``_windows.npy`` when provided.
    """

    def __init__(
        self,
        zarr_indices: np.ndarray,
        day: np.ndarray,
        hour: np.ndarray,
        export_config: RandomWindowExportConfig,
        *,
        mean: torch.Tensor,
        std: torch.Tensor,
        iD: int = 2,
        fixed_windows: np.ndarray | None = None,
        epoch_windows: np.ndarray | None = None,
        random_crop: bool = True,
        crop_seed: int = 42,
    ):
        self.zarr_indices = np.asarray(zarr_indices, dtype=np.int64)
        self.day = torch.from_numpy(np.asarray(day))
        self.hour = torch.from_numpy(np.asarray(hour))
        self.export_config = export_config
        self.mean = mean
        self.std = std
        self.iD = int(iD)
        self.fixed_windows = None if fixed_windows is None else np.asarray(fixed_windows)
        self.epoch_windows = None if epoch_windows is None else np.asarray(epoch_windows)
        self.random_crop = bool(random_crop)
        self.crop_seed = int(crop_seed)
        self.epoch = 0
        # Lazy S3/Zarr open per process — forked DataLoader workers must not reuse parent's boto3 client.
        self._zarr_ds: GuianaDataloader | None = None
        self._zarr_pid: int | None = None
        self._h: int | None = None
        self._w: int | None = None
        self._crop_h = int(export_config.crop_height)
        self._crop_w = int(export_config.crop_width)
        self._channel_names = tuple(export_config.channel_names)

    def _get_zarr_ds(self) -> GuianaDataloader:
        pid = os.getpid()
        if self._zarr_ds is None or self._zarr_pid != pid:
            self._zarr_ds = _build_zarr_dataset(self.export_config)
            self._h, self._w = _grid_size(self._zarr_ds)
            self._zarr_pid = pid
        return self._zarr_ds

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.zarr_indices)

    def _load_with_mask_filter(self, pool_idx: int, ds_idx: int) -> tuple[np.ndarray, np.ndarray]:
        zarr_ds = self._get_zarr_ds()
        h, w = self._h, self._w
        assert h is not None and w is not None

        if self.epoch_windows is not None:
            ep_i = max(0, min(int(self.epoch) - 1, self.epoch_windows.shape[0] - 1))
            top = int(self.epoch_windows[ep_i, pool_idx, 0])
            left = int(self.epoch_windows[ep_i, pool_idx, 1])
            seq, mask, _, _ = load_sequence_at_window(
                zarr_ds,
                int(ds_idx),
                top=top,
                left=left,
                crop_h=self._crop_h,
                crop_w=self._crop_w,
                channel_names=self._channel_names,
            )
            return seq, mask

        if not self.random_crop and self.fixed_windows is not None:
            top = int(self.fixed_windows[pool_idx, 0])
            left = int(self.fixed_windows[pool_idx, 1])
            seq, mask, _, _ = load_sequence_at_window(
                zarr_ds,
                int(ds_idx),
                top=top,
                left=left,
                crop_h=self._crop_h,
                crop_w=self._crop_w,
                channel_names=self._channel_names,
            )
            return seq, mask

        win = find_valid_crop_window(
            zarr_ds,
            int(ds_idx),
            int(pool_idx),
            int(self.epoch),
            crop_seed=self.crop_seed,
            crop_h=self._crop_h,
            crop_w=self._crop_w,
            max_mask_frac=float(self.export_config.max_mask_frac),
            max_crop_attempts=int(self.export_config.max_crop_attempts),
            channel_names=self._channel_names,
        )
        top = int(win[0])
        left = int(win[1])
        seq, mask, _, _ = load_sequence_at_window(
            zarr_ds,
            int(ds_idx),
            top=top,
            left=left,
            crop_h=self._crop_h,
            crop_w=self._crop_w,
            channel_names=self._channel_names,
        )
        return seq, mask

    def __getitem__(self, idx: int):
        ds_idx = int(self.zarr_indices[int(idx)])
        seq, mask_np = self._load_with_mask_filter(int(idx), ds_idx)
        x, y = _compact_xy_from_seq(seq, mean=self.mean, std=self.std, iD=self.iD)
        mask = torch.from_numpy(mask_np.astype(np.float32, copy=False))
        ocean = (mask == 0).to(dtype=x.dtype).unsqueeze(0)
        x = x * ocean
        y = y * ocean
        return (
            x,
            y,
            mask,
            self.day[int(idx)].float() / 365.0,
            self.hour[int(idx)].float() / 365.0,
        )
