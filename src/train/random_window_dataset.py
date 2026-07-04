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
from src.train.channel_select import resolve_channel_indices, resolve_target_channel_indices
from src.train.fix_window_train import _global_channel_stats


def _writable_float_tensor(arr: np.ndarray) -> torch.Tensor:
    """Copy mmap/read-only NumPy slices before ``torch.from_numpy``."""
    return torch.from_numpy(np.array(arr, dtype=np.float32, copy=True))


def window_pool_paths(prefix: str, train_crops_per_anchor: int) -> dict[str, Path]:
    tag = f"k{int(train_crops_per_anchor)}"
    return {
        "sims": Path(f"{prefix}_sims_pool_{tag}.npy"),
        "mask": Path(f"{prefix}_mask_pool_{tag}.npy"),
        "windows": Path(f"{prefix}_windows_pool_{tag}.npy"),
    }


def load_window_pool(prefix: str, train_crops_per_anchor: int) -> dict | None:
    """Load mmap'd window pools ``(n_pool, n_windows, ...)`` if exported."""
    paths = window_pool_paths(prefix, train_crops_per_anchor)
    if not paths["sims"].exists():
        return None
    sims = np.load(paths["sims"], mmap_mode="r")
    return {
        "sims": sims,
        "mask": np.load(paths["mask"], mmap_mode="r"),
        "windows": np.load(paths["windows"], mmap_mode="r") if paths["windows"].exists() else None,
        "n_windows": int(sims.shape[1]),
    }


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
    channel_indices: tuple[int, ...] | None = None,
    target_channel_indices: tuple[int, ...] | None = None,
    n_frames: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-channel mean/std on train split (cond frames): input then output channels."""
    ch_in = resolve_channel_indices(iD, channel_indices)
    ch_out = resolve_target_channel_indices(ch_in, target_channel_indices)
    sims = np.load(f"{data_prefix}_sims.npy", mmap_mode="r")
    idx = np.asarray(train_idx, dtype=np.int64)
    data = _writable_float_tensor(np.asarray(sims[idx]))
    stat_frames = list(range(n_frames - 1)) if n_frames <= 4 else [0, 1, 2, 3]
    mean_x, std_x = _global_channel_stats(data, stat_frames, channel_indices=ch_in)
    mean_y, std_y = _global_channel_stats(data, stat_frames, channel_indices=ch_out)
    return mean_x, std_x, mean_y, std_y


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
    mean_x: torch.Tensor,
    std_x: torch.Tensor,
    mean_y: torch.Tensor,
    std_y: torch.Tensor,
    channel_indices: tuple[int, ...],
    target_channel_indices: tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build normalized (x, y) increment tensors from raw [T,C,H,W] sequence."""
    ch_in = channel_indices
    ch_out = resolve_target_channel_indices(ch_in, target_channel_indices)
    data_in = _writable_float_tensor(seq[:, ch_in])
    data_out = _writable_float_tensor(seq[:, ch_out])
    n_frames = int(data_in.shape[0])
    cond_idx = list(range(n_frames - 1))
    anchor_idx = n_frames - 2
    forecast_idx = n_frames - 1
    x = torch.cat([data_in[t] for t in cond_idx], dim=0).float()
    y = (data_out[forecast_idx] - data_out[anchor_idx]).float()
    n_x_frames = len(cond_idx)
    mean_x_rep = mean_x.repeat(1, n_x_frames, 1, 1)
    std_x_rep = std_x.repeat(1, n_x_frames, 1, 1)
    std_y_rep = std_y.repeat(1, 1, 1, 1)
    x = (x.unsqueeze(0) - mean_x_rep) / (std_x_rep + 1e-8)
    y = y.unsqueeze(0) / (std_y_rep + 1e-8)
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
        mean_x: torch.Tensor,
        std_x: torch.Tensor,
        mean_y: torch.Tensor,
        std_y: torch.Tensor,
        iD: int = 2,
        channel_indices: tuple[int, ...] | None = None,
        target_channel_indices: tuple[int, ...] | None = None,
        fixed_windows: np.ndarray | None = None,
        epoch_windows: np.ndarray | None = None,
        random_crop: bool = True,
        crop_seed: int = 42,
    ):
        self.zarr_indices = np.asarray(zarr_indices, dtype=np.int64)
        self.day = torch.from_numpy(np.asarray(day))
        self.hour = torch.from_numpy(np.asarray(hour))
        self.export_config = export_config
        self.mean_x = mean_x
        self.std_x = std_x
        self.mean_y = mean_y
        self.std_y = std_y
        self.mean = mean_y
        self.std = std_y
        self.channel_indices = resolve_channel_indices(iD, channel_indices)
        self.target_channel_indices = resolve_target_channel_indices(
            self.channel_indices, target_channel_indices
        )
        self.iD = len(self.target_channel_indices)
        self.iD_in = len(self.channel_indices)
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
        x, y = _compact_xy_from_seq(
            seq,
            mean_x=self.mean_x,
            std_x=self.std_x,
            mean_y=self.mean_y,
            std_y=self.std_y,
            channel_indices=self.channel_indices,
            target_channel_indices=self.target_channel_indices,
        )
        mask = _writable_float_tensor(mask_np)
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


class WindowPoolCachedDataset(Dataset):
    """
    Train from pre-exported ``_sims_pool_k{n}.npy`` — no Zarr reads.

    Each epoch randomly picks ``slots_per_epoch`` distinct catalog windows per snapshot.
    Each picked window is a separate training row, so ``__len__`` =
    ``n_snapshots * slots_per_epoch`` (RNG: ``[crop_seed, epoch, snapshot_idx]``).
    """

    def __init__(
        self,
        sims_pool: np.ndarray,
        mask_pool: np.ndarray,
        pool_row_indices: np.ndarray,
        day: np.ndarray,
        hour: np.ndarray,
        *,
        mean_x: torch.Tensor,
        std_x: torch.Tensor,
        mean_y: torch.Tensor,
        std_y: torch.Tensor,
        iD: int = 2,
        channel_indices: tuple[int, ...] | None = None,
        target_channel_indices: tuple[int, ...] | None = None,
        crop_seed: int = 42,
        n_windows: int = 12,
        slots_per_epoch: int | None = None,
        fixed_window_slot: int | None = None,
    ):
        self.sims_pool = sims_pool
        self.mask_pool = mask_pool
        self.pool_row_indices = np.asarray(pool_row_indices, dtype=np.int64)
        self.day = torch.from_numpy(np.asarray(day))
        self.hour = torch.from_numpy(np.asarray(hour))
        self.mean_x = mean_x
        self.std_x = std_x
        self.mean_y = mean_y
        self.std_y = std_y
        self.mean = mean_y
        self.std = std_y
        self.channel_indices = resolve_channel_indices(iD, channel_indices)
        self.target_channel_indices = resolve_target_channel_indices(
            self.channel_indices, target_channel_indices
        )
        self.iD = len(self.target_channel_indices)
        self.iD_in = len(self.channel_indices)
        self.crop_seed = int(crop_seed)
        self.n_windows = int(n_windows)
        n_pick = int(slots_per_epoch) if slots_per_epoch is not None else 1
        if n_pick < 1 or n_pick > self.n_windows:
            raise ValueError(
                f"slots_per_epoch must be in 1..{self.n_windows}, got {n_pick}"
            )
        self.slots_per_epoch = n_pick
        if fixed_window_slot is not None:
            slot = int(fixed_window_slot)
            if slot < 0 or slot >= self.n_windows:
                raise ValueError(
                    f"fixed_window_slot must be in 0..{self.n_windows - 1}, got {slot}"
                )
            self.fixed_window_slot = slot
        else:
            self.fixed_window_slot = None
        self.epoch = 1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.pool_row_indices) * self.slots_per_epoch

    def _active_slots(self, snapshot_idx: int) -> np.ndarray:
        """``slots_per_epoch`` distinct window indices for one snapshot this epoch."""
        if self.fixed_window_slot is not None:
            return np.full(self.slots_per_epoch, self.fixed_window_slot, dtype=np.int64)
        rng = np.random.default_rng([self.crop_seed, self.epoch, int(snapshot_idx)])
        if self.slots_per_epoch >= self.n_windows:
            perm = rng.permutation(self.n_windows)
            return perm[: self.slots_per_epoch]
        return rng.choice(self.n_windows, size=self.slots_per_epoch, replace=False)

    def __getitem__(self, idx: int):
        snapshot_idx = int(idx) // self.slots_per_epoch
        slot_pick = int(idx) % self.slots_per_epoch
        row = int(self.pool_row_indices[snapshot_idx])
        slot = int(self._active_slots(snapshot_idx)[slot_pick])
        seq = np.asarray(self.sims_pool[row, slot], dtype=np.float32)
        mask_np = np.asarray(self.mask_pool[row, slot], dtype=np.float32)
        x, y = _compact_xy_from_seq(
            seq,
            mean_x=self.mean_x,
            std_x=self.std_x,
            mean_y=self.mean_y,
            std_y=self.std_y,
            channel_indices=self.channel_indices,
            target_channel_indices=self.target_channel_indices,
        )
        mask = _writable_float_tensor(mask_np)
        ocean = (mask == 0).to(dtype=x.dtype).unsqueeze(0)
        x = x * ocean
        y = y * ocean
        return (
            x,
            y,
            mask,
            self.day[snapshot_idx].float() / 365.0,
            self.hour[snapshot_idx].float() / 365.0,
        )


class WindowPoolEvalDataset(Dataset):
    """
    Deterministic eval: every snapshot × every catalog window (no random slot).

    ``__len__`` = ``n_snapshots * n_windows`` unless ``window_slot`` is set (viz / single-window).
    Returns ``(x, y, mask, day, hour, window_slot)``.
    """

    def __init__(
        self,
        sims_pool: np.ndarray,
        mask_pool: np.ndarray,
        pool_row_indices: np.ndarray,
        day: np.ndarray,
        hour: np.ndarray,
        *,
        mean_x: torch.Tensor,
        std_x: torch.Tensor,
        mean_y: torch.Tensor,
        std_y: torch.Tensor,
        iD: int = 2,
        channel_indices: tuple[int, ...] | None = None,
        target_channel_indices: tuple[int, ...] | None = None,
        window_slot: int | None = None,
    ):
        self.sims_pool = sims_pool
        self.mask_pool = mask_pool
        self.pool_row_indices = np.asarray(pool_row_indices, dtype=np.int64)
        self.day = torch.from_numpy(np.asarray(day))
        self.hour = torch.from_numpy(np.asarray(hour))
        self.mean_x = mean_x
        self.std_x = std_x
        self.mean_y = mean_y
        self.std_y = std_y
        self.mean = mean_y
        self.std = std_y
        self.channel_indices = resolve_channel_indices(iD, channel_indices)
        self.target_channel_indices = resolve_target_channel_indices(
            self.channel_indices, target_channel_indices
        )
        self.iD = len(self.target_channel_indices)
        self.iD_in = len(self.channel_indices)
        self.n_windows = int(sims_pool.shape[1])
        if window_slot is not None:
            slot = int(window_slot)
            if slot < 0 or slot >= self.n_windows:
                raise ValueError(f"window_slot must be in 0..{self.n_windows - 1}, got {slot}")
            self.window_slot = slot
        else:
            self.window_slot = None

    def __len__(self) -> int:
        n_snap = len(self.pool_row_indices)
        return n_snap if self.window_slot is not None else n_snap * self.n_windows

    def _unpack_index(self, idx: int) -> tuple[int, int]:
        if self.window_slot is not None:
            return int(idx), self.window_slot
        snap_i = int(idx) // self.n_windows
        slot = int(idx) % self.n_windows
        return snap_i, slot

    def __getitem__(self, idx: int):
        snap_i, slot = self._unpack_index(int(idx))
        row = int(self.pool_row_indices[snap_i])
        seq = np.asarray(self.sims_pool[row, slot], dtype=np.float32)
        mask_np = np.asarray(self.mask_pool[row, slot], dtype=np.float32)
        x, y = _compact_xy_from_seq(
            seq,
            mean_x=self.mean_x,
            std_x=self.std_x,
            mean_y=self.mean_y,
            std_y=self.std_y,
            channel_indices=self.channel_indices,
            target_channel_indices=self.target_channel_indices,
        )
        mask = _writable_float_tensor(mask_np)
        ocean = (mask == 0).to(dtype=x.dtype).unsqueeze(0)
        x = x * ocean
        y = y * ocean
        return (
            x,
            y,
            mask,
            self.day[snap_i].float() / 365.0,
            self.hour[snap_i].float() / 365.0,
        ) if self.window_slot is not None else (
            x,
            y,
            mask,
            self.day[snap_i].float() / 365.0,
            self.hour[snap_i].float() / 365.0,
            torch.tensor(slot, dtype=torch.int64),
        )


def evaluate_window_pool_metrics(
    model,
    dataloader,
    device: torch.device | str,
    *,
    iD_in: int,
    n_windows: int,
    loss_spatial_weight: torch.Tensor | None = None,
    loss_metric: str = "components",
    mean_y: torch.Tensor | None = None,
    std_y: torch.Tensor | None = None,
) -> dict:
    """Aggregate per-window and overall validation/test RMSE, L1, and speed skill."""
    from src.train.losses import (
        masked_rmse,
        masked_speed_rmse,
        reconstruction_loss,
        speed_skill_from_uv_fields,
    )

    model.eval()
    sum_rmse = np.zeros(n_windows, dtype=np.float64)
    sum_l1 = np.zeros(n_windows, dtype=np.float64)
    sum_skill = np.zeros(n_windows, dtype=np.float64)
    counts = np.zeros(n_windows, dtype=np.int64)
    total_rmse = 0.0
    total_l1 = 0.0
    total_skill = 0.0
    total_n = 0
    compute_skill = mean_y is not None and std_y is not None
    if compute_skill:
        mean_y = mean_y.to(device)
        std_y = std_y.to(device)

    with torch.no_grad():
        for batch in dataloader:
            x, y, m, d, h_batch, slots = batch
            x = x.to(device)
            y = y.to(device)
            m = m.to(device)
            d = d.to(device)
            h_batch = h_batch.to(device)
            pred = model(x[:, -3 * iD_in :], d + h_batch / 365.0)
            target = y[:, : pred.shape[1]]
            batch_size = int(pred.shape[0])
            for j in range(batch_size):
                pj = pred[j : j + 1]
                tj = target[j : j + 1]
                mj = m[j : j + 1]
                slot = int(slots[j].item())
                if loss_metric == "speed":
                    rmse_j = float(masked_speed_rmse(pj, tj, mj, loss_spatial_weight).item())
                else:
                    rmse_j = float(masked_rmse(pj, tj, mj, loss_spatial_weight).item())
                l1_j = float(
                    reconstruction_loss(
                        pj, tj, mj, metric=loss_metric, spatial_weight=loss_spatial_weight
                    ).item()
                )
                sum_rmse[slot] += rmse_j
                sum_l1[slot] += l1_j
                counts[slot] += 1
                total_rmse += rmse_j
                total_l1 += l1_j
                total_n += 1
                if compute_skill:
                    anchor = x[j : j + 1, -iD_in:]
                    anchor_abs = anchor * std_y + mean_y
                    target_abs = anchor_abs + tj * std_y
                    pred_abs = anchor_abs + pj * std_y
                    skill_j, _, _ = speed_skill_from_uv_fields(
                        anchor_abs,
                        target_abs,
                        pred_abs,
                        mj,
                        loss_spatial_weight,
                    )
                    skill_val = float(skill_j.item())
                    sum_skill[slot] += skill_val
                    total_skill += skill_val

    per_window_rmse = {
        f"window_{i:02d}": float(sum_rmse[i] / counts[i]) if counts[i] else float("nan")
        for i in range(n_windows)
    }
    per_window_l1 = {
        f"window_{i:02d}_l1": float(sum_l1[i] / counts[i]) if counts[i] else float("nan")
        for i in range(n_windows)
    }
    per_window_skill = {
        f"window_{i:02d}_skill": float(sum_skill[i] / counts[i]) if counts[i] else float("nan")
        for i in range(n_windows)
    }
    overall_rmse = float(total_rmse / total_n) if total_n else float("nan")
    overall_l1 = float(total_l1 / total_n) if total_n else float("nan")
    overall_skill = float(total_skill / total_n) if total_n and compute_skill else float("nan")
    out = {
        "overall": overall_rmse,
        "overall_l1": overall_l1,
        "overall_rmse": overall_rmse,
        "overall_skill": overall_skill,
        **per_window_rmse,
        **per_window_l1,
        **per_window_skill,
        "counts": counts.tolist(),
        "n_samples": int(total_n),
        "n_snapshots": int(counts.max()) if counts.any() else 0,
    }
    return out
