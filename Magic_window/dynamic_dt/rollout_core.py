"""Rollout evaluation for dynamic_dt exports (3-step autoregressive, speed metrics)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from gUnet import Unet2D
from src.data.dataloaders.guiana_margem_tensor import GuianaDataloader
from src.data.useful_transforms import FixedCrop, RotatedFixedCrop

from dynamic_dt.dt_config import forecast_horizons_days
from dynamic_dt.rollout_plotting import save_rollout_sample_plots
from dynamic_dt.sequence_config import CONDITIONING_STEPS


def load_test_data(
    data_prefix: str,
    stats_path: Path | str,
    *,
    iD: int = 2,
    conditioning_steps: int = CONDITIONING_STEPS,
    test_split: float = 1.0,
    seed: int = 42,
) -> dict[str, Any]:
    sims = torch.from_numpy(np.load(f"{data_prefix}_sims.npy"))
    mask = torch.from_numpy(np.load(f"{data_prefix}_mask.npy"))
    day = torch.from_numpy(np.load(f"{data_prefix}_day.npy"))
    hour = torch.from_numpy(np.load(f"{data_prefix}_hour.npy"))
    latlon = torch.from_numpy(np.load(f"{data_prefix}_latlon.npy"))
    export_indices = np.load(f"{data_prefix}_indices.npy")

    n = len(sims)
    test_idx = np.arange(n) if test_split >= 1.0 else np.random.default_rng(seed).permutation(n)[
        : int(n * test_split)
    ]

    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    mean = stats["mean"]
    std = stats["std"]
    mean_x = mean.repeat(1, conditioning_steps, 1, 1)
    std_x = std.repeat(1, conditioning_steps, 1, 1)

    return {
        "sims": sims[test_idx],
        "mask": mask[test_idx],
        "day": day[test_idx] / 365.0,
        "hour": hour[test_idx] / 365.0,
        "latlon": latlon[test_idx],
        "mean": mean_x,
        "std": std_x,
        "n_frames": int(sims.shape[1]),
        "export_indices": export_indices[test_idx],
        "iD": iD,
        "conditioning_steps": conditioning_steps,
    }


def _build_zarr_loader(export_config_path: Path, window_meta: np.ndarray, dt_hours: int):
    cfg = json.loads(export_config_path.read_text())
    cond_steps = int(cfg.get("conditioning_steps", CONDITIONING_STEPS))
    fcst_steps = int(cfg.get("forecast_steps", 1))
    top, left, h, w, angle = window_meta
    if float(angle) == 0.0:
        crop = FixedCrop(size=(int(h), int(w)), top=int(top), left=int(left))
    else:
        crop = RotatedFixedCrop(
            size=(int(h), int(w)), top=int(top), left=int(left), angle_deg=float(angle)
        )
    return GuianaDataloader(
        path=cfg["zarr_path"],
        dt=dt_hours,
        conditioning_steps=cond_steps,
        forecast_steps=fcst_steps,
        transform=crop,
    )


def _load_cropped_uv_at(zarr_ds, timestamp) -> torch.Tensor:
    """Load cropped u,v at one Zarr time. Returns [2, H, W]."""
    t = pd.Timestamp(timestamp)
    fields = zarr_ds.load_uvssh_frame(t).astype(np.float32, copy=False)  # [3, H, W] u,v,ssh
    fields = np.nan_to_num(fields, nan=0.0)
    h, w = fields.shape[-2:]
    mask_ch = (~np.isfinite(fields[:1])).astype(np.float32)
    bat = np.ones((1, h, w), dtype=np.float32)
    lat, lon = zarr_ds._latlon_grid(h, w)
    bundle = torch.cat(
        [
            torch.tensor(mask_ch, dtype=torch.float32).unsqueeze(0),
            torch.tensor(fields, dtype=torch.float32).unsqueeze(0),
            torch.tensor(bat, dtype=torch.float32).unsqueeze(0),
            torch.tensor(lat, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            torch.tensor(lon, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
        ],
        dim=1,
    )
    if zarr_ds.transform is not None:
        bundle = zarr_ds.transform(bundle)
        if not torch.is_tensor(bundle):
            bundle = torch.tensor(bundle, dtype=torch.float32)
    _, data, _, _, _ = torch.split(bundle, [1, 3, 1, 1, 1], dim=1)
    return data[0, :2].clone()


def load_gt_horizons_from_zarr(
    data_prefix: str,
    export_indices: np.ndarray,
    horizons_days: tuple[int, ...],
    crop_hw: tuple[int, int],
    *,
    anchor_frame_idx: int | None = None,
) -> dict[int, torch.Tensor]:
    cfg_path = Path(f"{data_prefix}_export_config.json")
    window_meta = np.load(f"{data_prefix}_window.npy")
    cfg = json.loads(cfg_path.read_text())
    dt_hours = int(cfg["dt"])
    cond_steps = int(cfg.get("conditioning_steps", CONDITIONING_STEPS))
    anchor_idx = cond_steps - 1 if anchor_frame_idx is None else anchor_frame_idx
    zarr_ds = _build_zarr_loader(cfg_path, window_meta, dt_hours)
    h, w = crop_hw
    out = {d: [] for d in horizons_days}
    for ds_idx in export_indices:
        times = zarr_ds.sample_times(int(ds_idx))
        anchor = pd.Timestamp(times[anchor_idx])
        for d in horizons_days:
            t = anchor + pd.Timedelta(days=d)
            if t not in zarr_ds.time_to_idx:
                out[d].append(torch.zeros(2, h, w))
                continue
            out[d].append(_load_cropped_uv_at(zarr_ds, t))
    return {d: torch.stack(out[d], dim=0) for d in horizons_days}


def _build_model_input(sims: torch.Tensor, frame_indices: Sequence[int], mean, std) -> torch.Tensor:
    parts = [sims[:, i] for i in frame_indices]
    x = torch.cat(parts, dim=1).float()
    return (x - mean) / (std + 1e-8)


def _ar_shift_input(x_batch: torch.Tensor, pred: torch.Tensor, n_cond: int, iD: int) -> torch.Tensor:
    chunks = [x_batch[:, i * iD : (i + 1) * iD] for i in range(1, n_cond)]
    chunks.append(pred)
    return torch.cat(chunks, dim=1)


def run_inference(
    model: torch.nn.Module,
    test_data: dict,
    *,
    dt_step_days: int,
    device: str = "cuda",
    batch_size: int = 16,
    iD: int = 2,
    conditioning_steps: int = CONDITIONING_STEPS,
):
    model = model.to(device).eval()
    sims = test_data["sims"][:, :, :iD, :, :]
    day, hour = test_data["day"], test_data["hour"]
    mean, std = test_data["mean"], test_data["std"]
    masks = test_data["mask"]
    y_mean = mean[:, :iD]
    y_std = std[:, :iD]
    n_cond = int(conditioning_steps)
    cond_idx = list(range(n_cond))

    x1 = _build_model_input(sims, cond_idx, mean, std)
    loader = DataLoader(torch.utils.data.TensorDataset(x1, day, hour, masks), batch_size=batch_size)

    preds1, preds2, preds3 = [], [], []
    with torch.no_grad():
        for x_batch, d_batch, h_batch, m_batch in loader:
            x_batch = x_batch.to(device)
            d_batch, h_batch, m_batch = d_batch.to(device), h_batch.to(device), m_batch.to(device)
            ocean = (m_batch == 0).to(x_batch.dtype).unsqueeze(1)
            x_batch = x_batch * ocean

            pred1 = model(x_batch, d_batch + h_batch / 365.0)
            pred1 = (pred1 + x_batch[:, -iD:]) * ocean

            x2 = _ar_shift_input(x_batch, pred1, n_cond, iD) * ocean
            pred2 = model(x2, d_batch + h_batch / 365.0 + dt_step_days / 365.0)
            pred2 = (pred2 + x2[:, -iD:]) * ocean

            x3 = _ar_shift_input(x2, pred2, n_cond, iD) * ocean
            pred3 = model(x3, d_batch + h_batch / 365.0 + 2 * dt_step_days / 365.0)
            pred3 = (pred3 + pred2) * ocean

            preds1.append(pred1.cpu())
            preds2.append(pred2.cpu())
            preds3.append(pred3.cpu())

    p1 = torch.cat(preds1) * (y_std + 1e-8) + y_mean
    p2 = torch.cat(preds2) * (y_std + 1e-8) + y_mean
    p3 = torch.cat(preds3) * (y_std + 1e-8) + y_mean

    anchor_idx = n_cond - 1
    forecast_idx = n_cond
    s_anchor = sims[:, anchor_idx]
    s_h1 = sims[:, forecast_idx]

    h2, h3 = 2 * dt_step_days, 3 * dt_step_days
    gt_extra = load_gt_horizons_from_zarr(
        test_data.get("data_prefix", ""),
        test_data["export_indices"],
        (h2, h3),
        (sims.shape[-2], sims.shape[-1]),
        anchor_frame_idx=anchor_idx,
    )
    s_h2, s_h3 = gt_extra[h2], gt_extra[h3]

    # Plot helpers: last 3 cond frames + anchor (t-2, t-1, t) and GT horizons
    s_m2 = sims[:, max(0, anchor_idx - 2)]
    s_m1 = sims[:, max(0, anchor_idx - 1)]
    s_t = s_anchor
    return s_m2, s_m1, s_t, s_h1, s_h2, s_h3, p1, p2, p3


def speed_rmse_metrics(
    anchor,
    gt_list: list,
    pred_list: list,
    mask: torch.Tensor,
    horizons: tuple[int, ...],
    *,
    patch_size: int | None = 4,
) -> list[dict[str, float]]:
    def spd(x):
        a = x.numpy() if torch.is_tensor(x) else np.asarray(x)
        return np.sqrt(a[:, 0] ** 2 + a[:, 1] ** 2)

    def rmse(a, b, m, ps):
        m = m.numpy() if torch.is_tensor(m) else np.asarray(m)
        for _ in range(a.ndim - 2):
            m = m[..., :, :] if m.ndim == a.ndim - 1 else m
        if ps is not None:
            cy, cx = a.shape[-2] // 2, a.shape[-1] // 2
            half = ps // 2
            a = a[..., cy - half : cy + half, cx - half : cx + half]
            b = b[..., cy - half : cy + half, cx - half : cx + half]
            m = m[..., cy - half : cy + half, cx - half : cx + half]
        valid = m == 0
        if not np.any(valid):
            return float("nan")
        return float(np.sqrt(np.mean((a[valid] - b[valid]) ** 2)))

    speed_now = spd(anchor)
    rows = []
    for d, gt, pred in zip(horizons, gt_list, pred_list):
        gt_s, pr_s = spd(gt), spd(pred)
        persist = rmse(speed_now, gt_s, mask, patch_size)
        model = rmse(pr_s, gt_s, mask, patch_size)
        rows.append({
            "horizon_days": d,
            "persistence_rmse": persist,
            "model_rmse": model,
            "skill_score": 1.0 - model / persist if persist > 0 else float("nan"),
        })
    return rows


def evaluate_rollout(
    *,
    data_prefix: str,
    model_path: Path | str,
    stats_path: Path | str,
    dt_days: int,
    train_cfg: dict[str, Any],
    output_dir: Path | str,
    device: str = "cuda",
    plot_indices: Sequence[int] | None = None,
    channel_names: Sequence[str] = ("U", "V"),
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    iD = int(train_cfg.get("iD", 2))
    conditioning_steps = int(train_cfg.get("conditioning_steps", CONDITIONING_STEPS))
    dt_hours = int(train_cfg.get("dt_hours", dt_days * 24))
    horizons = forecast_horizons_days(dt_days)
    crop_h = int(train_cfg.get("crop_height", 128))
    crop_w = int(train_cfg.get("crop_width", 128))

    test_data = load_test_data(
        data_prefix, stats_path, iD=iD, conditioning_steps=conditioning_steps
    )
    test_data["data_prefix"] = data_prefix

    Par = {
        "lb": conditioning_steps,
        "nf": iD,
        "nx": crop_h,
        "ny": crop_w,
        "inp_shift": 0.0,
        "inp_scale": 1.0,
        "out_shift": 0.0,
        "out_scale": 1.0,
        "t_shift": 0.0,
        "t_scale": 1.0,
    }
    model = Unet2D(
        dim=int(train_cfg.get("unet_dim", 32)),
        Par=Par,
        dim_mults=tuple(train_cfg.get("unet_dim_mults", (1, 2, 4, 8))),
        channels=Par["lb"] * Par["nf"],
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))

    s_m2, s_m1, s_t, s_h1, s_h2, s_h3, p1, p2, p3 = run_inference(
        model,
        test_data,
        dt_step_days=dt_days,
        device=device,
        iD=iD,
        conditioning_steps=conditioning_steps,
    )
    mask = test_data["mask"]
    center = speed_rmse_metrics(s_t, [s_h1, s_h2, s_h3], [p1, p2, p3], mask, horizons, patch_size=4)
    full = speed_rmse_metrics(s_t, [s_h1, s_h2, s_h3], [p1, p2, p3], mask, horizons, patch_size=None)

    sample_plots: list[str] = []
    if plot_indices:
        samples_dir = output_dir / "samples"
        # Avoid confusing leftovers when re-running rollouts with a different
        # `plot_indices` strategy.
        if samples_dir.exists():
            for child in samples_dir.iterdir():
                if child.is_file():
                    child.unlink()
        sample_plots = save_rollout_sample_plots(
            s_m2, s_m1, s_t, s_h1, s_h2, s_h3, p1, p2, p3,
            test_data,
            samples_dir=samples_dir,
            plot_indices=plot_indices,
            horizons=horizons,
            channel_names=channel_names,
        )

    results = {
        "data_prefix": data_prefix,
        "dt_days": dt_days,
        "dt_hours": dt_hours,
        "conditioning_steps": conditioning_steps,
        "horizons_days": list(horizons),
        "metric": "speed_rmse",
        "n_test": int(s_t.shape[0]),
        "metrics_center_4x4": center,
        "metrics_full_field": full,
        "sample_plots": sample_plots,
    }
    (output_dir / "rollout_results.json").write_text(json.dumps(results, indent=2) + "\n")
    return results
