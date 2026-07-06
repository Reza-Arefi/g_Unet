"""Rollout sample plots (same layout as Shalow/05_Sh_evaluate_magic_window_rollout)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch


def apply_mask(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = field.copy()
    out[mask == 1] = np.nan
    return out


def _symmetric_err_lim(fields: list[np.ndarray]) -> float:
    finite = np.concatenate([np.abs(f[np.isfinite(f)]) for f in fields if np.any(np.isfinite(f))])
    if finite.size == 0:
        return 1.0
    return max(float(np.nanmax(finite)), 1e-6)


def plot_sample(
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    p1,
    p2,
    p3,
    test_data: dict,
    *,
    sample_idx: int = 0,
    save_path: str | Path | None = None,
    horizons: tuple[int, int, int],
    channel_names: Sequence[str] = ("U", "V"),
) -> tuple[float, ...]:
    """Multi-panel rollout figure: cond frames, GT/pred per horizon, speed, ΔU/ΔV/ΔSpeed."""
    h1, h2, h3 = horizons
    forecast_titles = (f"t[n+{h1}d]", f"t[n+{h2}d]", f"t[n+{h3}d]")

    s1 = s1[sample_idx].numpy()
    s2 = s2[sample_idx].numpy()
    s3 = s3[sample_idx].numpy()
    s4 = s4[sample_idx].numpy()
    s5 = s5[sample_idx].numpy()
    s6 = s6[sample_idx].numpy()
    p1 = p1[sample_idx].numpy()
    p2 = p2[sample_idx].numpy()
    p3 = p3[sample_idx].numpy()
    mask = test_data["mask"][sample_idx].numpy()
    day = test_data["day"][sample_idx].item()
    hour = test_data["hour"][sample_idx].item()
    lat, lon = test_data["latlon"][sample_idx].tolist()
    n_channels = p1.shape[0]
    static_bathy = s3[2] if s3.shape[0] > 2 and n_channels > 2 else None
    ch_names = list(channel_names)[:n_channels]

    def _field_ch(field, ch):
        if ch < field.shape[0]:
            return field[ch]
        if ch == 2 and static_bathy is not None:
            return static_bathy
        raise IndexError(f"channel {ch} unavailable for field with shape {field.shape}")

    s1u, s1v = s1[0], s1[1]
    S1 = np.sqrt(s1u ** 2 + s1v ** 2)
    S2 = np.sqrt(s2[0] ** 2 + s2[1] ** 2)
    S3 = np.sqrt(s3[0] ** 2 + s3[1] ** 2)
    gt_speeds = [
        np.sqrt(s4[0] ** 2 + s4[1] ** 2),
        np.sqrt(s5[0] ** 2 + s5[1] ** 2),
        np.sqrt(s6[0] ** 2 + s6[1] ** 2),
    ]
    pred_speeds = [
        np.sqrt(p1[0] ** 2 + p1[1] ** 2),
        np.sqrt(p2[0] ** 2 + p2[1] ** 2),
        np.sqrt(p3[0] ** 2 + p3[1] ** 2),
    ]
    errs = [
        (
            float(np.sqrt(np.nanmean((gt_speeds[i] - S3) ** 2))),
            float(np.sqrt(np.nanmean((pred_speeds[i] - gt_speeds[i]) ** 2))),
        )
        for i in range(3)
    ]

    if save_path is not None:
        gt_fields = [s4, s5, s6]
        pred_fields = [p1, p2, p3]
        gt_speeds_masked = [apply_mask(gs, mask) for gs in gt_speeds]
        pred_speeds_masked = [apply_mask(ps, mask) for ps in pred_speeds]
        speed_horizon_pairs = [
            (g, p, np.nanmin([g, p]), np.nanmax([g, p]))
            for g, p in zip(gt_speeds_masked, pred_speeds_masked)
        ]
        err_u = [apply_mask(pf[0] - gf[0], mask) for gf, pf in zip(gt_fields, pred_fields)]
        err_v = [apply_mask(pf[1] - gf[1], mask) for gf, pf in zip(gt_fields, pred_fields)]
        err_spd = [apply_mask(ps - gs, mask) for ps, gs in zip(pred_speeds, gt_speeds)]
        err_lim_u = _symmetric_err_lim(err_u)
        err_lim_v = _symmetric_err_lim(err_v)
        err_lim_spd = _symmetric_err_lim(err_spd)

        n_uv_rows = n_channels * 2
        speed_row_gt = n_uv_rows
        speed_row_pred = n_uv_rows + 1
        err_u_row = n_uv_rows + 2
        err_v_row = n_uv_rows + 3
        err_spd_row = n_uv_rows + 4
        n_rows = n_uv_rows + 5

        fig = plt.figure(figsize=(6 * 3, 2.8 * n_rows))
        fig.suptitle(
            f"Sample {sample_idx} | Day {int(day * 365):3d}, Hour {int(hour * 365 * 24):2d}h | "
            f"Lat {lat:.2f}, Lon {lon:.2f} — adv-NO rollout +{h1}/{h2}/{h3}d",
            fontsize=13,
            fontweight="bold",
        )
        gs = gridspec.GridSpec(n_rows, 6, figure=fig, wspace=0.35, hspace=0.45)

        for ch in range(n_channels):
            ss = [
                apply_mask(_field_ch(s1, ch), mask),
                apply_mask(_field_ch(s2, ch), mask),
                apply_mask(_field_ch(s3, ch), mask),
            ]
            ch_scale = ss + [
                apply_mask(_field_ch(g, ch), mask) for g in gt_fields
            ] + [apply_mask(_field_ch(p, ch), mask) for p in pred_fields]
            ch_vmin, ch_vmax = np.nanmin(ch_scale), np.nanmax(ch_scale)
            for col, field, title in zip(range(3), ss, ["t[n-2]", "t[n-1]", "t[n]"]):
                ax = fig.add_subplot(gs[2 * ch, col])
                im = ax.imshow(field, vmin=ch_vmin, vmax=ch_vmax, cmap="viridis")
                ax.set_title(f"{ch_names[ch]} — {title}")
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046)

            horizon_pairs = []
            for gf, pf in zip(gt_fields, pred_fields):
                g = apply_mask(_field_ch(gf, ch), mask)
                p = apply_mask(_field_ch(pf, ch), mask)
                horizon_pairs.append((g, p, np.nanmin([g, p]), np.nanmax([g, p])))

            for j, ((g, p, vmin, vmax), title) in enumerate(zip(horizon_pairs, forecast_titles)):
                ax = fig.add_subplot(gs[2 * ch, 3 + j])
                im = ax.imshow(g, vmin=vmin, vmax=vmax, cmap="viridis")
                ax.set_title(f"GT {title}")
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046)

            for j, ((g, p, vmin, vmax), title, (pe, ue)) in enumerate(
                zip(horizon_pairs, forecast_titles, errs)
            ):
                ax = fig.add_subplot(gs[2 * ch + 1, 3 + j])
                im = ax.imshow(p, vmin=vmin, vmax=vmax, cmap="viridis")
                ax.set_title(f"Pred {title}\nP {pe:.3e} | M {ue:.3e}", fontsize=8)
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046)

        ss_spd = [apply_mask(S1, mask), apply_mask(S2, mask), apply_mask(S3, mask)]
        spd_scale = ss_spd + [g for g, _, _, _ in speed_horizon_pairs] + [
            p for _, p, _, _ in speed_horizon_pairs
        ]
        spd_vmin, spd_vmax = np.nanmin(spd_scale), np.nanmax(spd_scale)
        for col, field, title in zip(range(3), ss_spd, ["t[n-2]", "t[n-1]", "t[n]"]):
            ax = fig.add_subplot(gs[speed_row_gt, col])
            im = ax.imshow(field, vmin=spd_vmin, vmax=spd_vmax, cmap="viridis")
            ax.set_title(f"Speed — {title}")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046)
        for j, ((g, p, vmin, vmax), title, (pe, ue)) in enumerate(
            zip(speed_horizon_pairs, forecast_titles, errs)
        ):
            ax = fig.add_subplot(gs[speed_row_gt, 3 + j])
            im = ax.imshow(g, vmin=vmin, vmax=vmax, cmap="viridis")
            ax.set_title(f"GT speed {title}")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax = fig.add_subplot(gs[speed_row_pred, 3 + j])
            im = ax.imshow(p, vmin=vmin, vmax=vmax, cmap="viridis")
            ax.set_title(f"Pred speed {title}\nP {pe:.3e} | M {ue:.3e}", fontsize=8)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046)

        for j, (field, title) in enumerate(zip(err_u, forecast_titles)):
            ax = fig.add_subplot(gs[err_u_row, 3 + j])
            im = ax.imshow(field, cmap="RdBu_r", vmin=-err_lim_u, vmax=err_lim_u)
            ax.set_title(f"ΔU pred−GT {title}")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046)
        for j, (field, title) in enumerate(zip(err_v, forecast_titles)):
            ax = fig.add_subplot(gs[err_v_row, 3 + j])
            im = ax.imshow(field, cmap="RdBu_r", vmin=-err_lim_v, vmax=err_lim_v)
            ax.set_title(f"ΔV pred−GT {title}")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046)
        for j, (field, title) in enumerate(zip(err_spd, forecast_titles)):
            ax = fig.add_subplot(gs[err_spd_row, 3 + j])
            im = ax.imshow(field, cmap="RdBu_r", vmin=-err_lim_spd, vmax=err_lim_spd)
            ax.set_title(f"ΔSpeed pred−GT {title}")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046)

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved to {save_path}")

    return errs[0][0], errs[0][1], errs[1][0], errs[1][1], errs[2][0], errs[2][1]


def save_rollout_sample_plots(
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    p1,
    p2,
    p3,
    test_data: dict,
    *,
    samples_dir: Path | str,
    plot_indices: Sequence[int],
    horizons: tuple[int, int, int],
    channel_names: Sequence[str] = ("U", "V"),
) -> list[str]:
    samples_dir = Path(samples_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    n = int(s1.shape[0])
    for i in plot_indices:
        if i < 0 or i >= n:
            continue
        path = samples_dir / f"sample_{i}.png"
        plot_sample(
            s1, s2, s3, s4, s5, s6, p1, p2, p3,
            test_data,
            sample_idx=i,
            save_path=path,
            horizons=horizons,
            channel_names=channel_names,
        )
        saved.append(str(path))
    return saved
