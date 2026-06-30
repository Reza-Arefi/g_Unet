"""Per-epoch validation figures and training curves for fix-window U-Net training."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.train.losses import (
    build_loss_spatial_weight,
    gaussian_weight_at_radius,
    gaussian_weight_ring_summary,
    masked_loss,
    reconstruction_loss,
    speed_magnitude_uv,
)


DEFAULT_CHANNELS = ("u", "v", "ssh")

TRAIN_STYLE = {"color": "#1f77b4", "linewidth": 1.8, "label": "train"}
VAL_STYLE = {"color": "#ff7f0e", "linewidth": 1.8, "label": "val"}
BEST_STYLE = {"color": "#2ca02c", "marker": "*", "s": 120, "zorder": 5}


def _symmetric_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    v = float(np.nanpercentile(np.abs(finite), 98))
    v = max(v, 1e-6)
    return -v, v


def _apply_mask(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Set land/invalid pixels (mask == 1) to NaN for imshow."""
    out = field.copy()
    out[mask == 1] = np.nan
    return out


def _add_land_contour(ax, mask: np.ndarray, *, color: str = "#222222", linewidth: float = 0.9) -> None:
    """Outline land (mask == 1) on an imshow axis."""
    if mask.size == 0 or float(np.max(mask)) <= 0:
        return
    h, w = mask.shape[-2], mask.shape[-1]
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    ax.contour(xx, yy, mask.astype(np.float64), levels=[0.5], colors=[color], linewidths=linewidth)


def _mask_panel_cmap():
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    cmap = plt.cm.RdYlBu_r.copy()
    cmap.set_under("#8b0000")
    cmap.set_over("#1a5276")
    return cmap, mcolors.Normalize(vmin=0.0, vmax=1.0)


def _residual_error(pred: np.ndarray, target: np.ndarray, metric: str) -> np.ndarray:
    diff = target - pred
    if metric == "signed":
        return diff
    if metric in ("l1", "rmse"):
        return np.abs(diff)
    if metric == "mse":
        return diff**2
    beta = 1.0
    adiff = np.abs(diff)
    return np.where(adiff < beta, 0.5 * adiff**2 / beta, adiff - 0.5 * beta)


def _panel_figsize(h: int, w: int, n_rows: int, n_cols: int) -> tuple[float, float]:
    base = max(h, w) / 128.0
    return (3.2 * n_cols * base, 2.8 * n_rows * base)


def _sample_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    return float(masked_loss(pred.unsqueeze(0), target.unsqueeze(0), mask.unsqueeze(0)).item())


def _sample_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    metric: str = "components",
) -> float:
    return float(
        reconstruction_loss(
            pred.unsqueeze(0),
            target.unsqueeze(0),
            mask.unsqueeze(0),
            metric=metric,
        ).item()
    )


def _uv_with_speed(field: torch.Tensor) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Stack u, v, and speed |U,V| for visualization. Input [C,H,W], C>=2."""
    if field.shape[0] < 2:
        raise ValueError("speed viz requires at least u and v channels")
    spd = speed_magnitude_uv(field.unsqueeze(0)).squeeze(0)
    stacked = torch.stack([field[0], field[1], spd])
    return stacked, ("u", "v", "speed")


def normalized_uv_increment(
    state_next: torch.Tensor,
    state_prev: torch.Tensor,
    std: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Normalized Δu,Δv = (next − prev) / std with land zeroed.

    Matches ``SimulationDataset`` increment targets used in training / val viz.
    Accepts ``[C,H,W]`` or ``[B,C,H,W]`` physical u,v states.
    """
    squeeze = False
    if state_next.dim() == 3:
        state_next = state_next.unsqueeze(0)
        state_prev = state_prev.unsqueeze(0)
        squeeze = True

    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    elif mask.dim() == 3 and mask.shape[1] == 1:
        mask = mask.squeeze(1)

    std_ch = std
    if std.dim() == 4 and std.shape[1] > state_next.shape[1]:
        std_ch = std[:, : state_next.shape[1]]
    if std_ch.dim() == 3:
        std_ch = std_ch.unsqueeze(0)

    ocean = (mask == 0).to(dtype=state_next.dtype).unsqueeze(1)
    inc = (state_next - state_prev) / (std_ch + 1e-8)
    inc = inc * ocean
    return inc.squeeze(0) if squeeze else inc


def normalize_physical_uv(
    state: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Physical u,v -> normalized with land zeroed (matches ``SimulationDataset`` input scaling)."""
    squeeze = False
    if state.dim() == 3:
        state = state.unsqueeze(0)
        squeeze = True

    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    elif mask.dim() == 3 and mask.shape[1] == 1:
        mask = mask.squeeze(1)

    mean_ch = mean
    std_ch = std
    if mean.dim() == 4 and mean.shape[1] > state.shape[1]:
        mean_ch = mean[:, : state.shape[1]]
    if std.dim() == 4 and std.shape[1] > state.shape[1]:
        std_ch = std[:, : state.shape[1]]
    if mean_ch.dim() == 3:
        mean_ch = mean_ch.unsqueeze(0)
    if std_ch.dim() == 3:
        std_ch = std_ch.unsqueeze(0)

    ocean = (mask == 0).to(dtype=state.dtype).unsqueeze(1)
    norm = (state - mean_ch) / (std_ch + 1e-8)
    return (norm * ocean).squeeze(0) if squeeze else norm * ocean


def rollout_pred_increment(
    pred_state_norm: torch.Tensor,
    anchor_phys: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Normalized increment from a GT anchor state to a predicted normalized state."""
    anchor_norm = normalize_physical_uv(anchor_phys, mean, std, mask)
    return pred_state_norm - anchor_norm


def _mean_abs_error(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    valid = mask == 0
    if not np.any(valid):
        return float("nan")
    errs = [np.abs(target[c] - pred[c])[valid] for c in range(pred.shape[0])]
    return float(np.concatenate(errs).mean())


def _circle_pixel_mask(h: int, w: int, radius: float) -> np.ndarray:
    """Boolean mask: pixels with center distance ≤ ``radius`` (px)."""
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij",
    )
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    return dist2 <= radius ** 2


def _mean_abs_error_in_circle(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    radius: float,
) -> float:
    """Mean |target − pred| over valid ocean pixels inside ``radius`` from center."""
    if pred.ndim == 2:
        pred = pred[np.newaxis]
        target = target[np.newaxis]
    h, w = pred.shape[-2], pred.shape[-1]
    circle = _circle_pixel_mask(h, w, radius)
    valid = (mask == 0) & circle
    if not np.any(valid):
        return float("nan")
    errs = [np.abs(target[c] - pred[c])[valid] for c in range(pred.shape[0])]
    return float(np.concatenate(errs).mean())


def _mean_abs_error_channel_in_circle(
    pred_ch: np.ndarray,
    target_ch: np.ndarray,
    mask: np.ndarray,
    radius: float,
) -> float:
    """Per-channel L1 inside circle (ocean pixels only)."""
    h, w = pred_ch.shape[-2], pred_ch.shape[-1]
    circle = _circle_pixel_mask(h, w, radius)
    valid = (mask == 0) & circle
    if not np.any(valid):
        return float("nan")
    return float(np.abs(target_ch - pred_ch)[valid].mean())


def _mean_gaussian_weighted_abs_error(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    gauss: np.ndarray,
) -> float:
    """Mean |pred−target|⊙G over valid pixels (matches training L1 weighting)."""
    valid = mask == 0
    if not np.any(valid):
        return float("nan")
    g = gauss[valid]
    denom = float(g.sum()) or 1.0
    errs = []
    for c in range(pred.shape[0]):
        errs.append(np.abs(target[c] - pred[c])[valid] * g)
    return float(np.concatenate(errs).sum() / denom)


def _best_val_index(values: list[float]) -> int | None:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return None
    return int(np.nanargmin(arr))


def _mark_warmup(ax, warmup_epochs: int) -> None:
    if warmup_epochs > 0:
        ax.axvline(
            warmup_epochs + 0.5,
            color="#888888",
            linestyle=":",
            alpha=0.7,
            label="adv-NO start",
        )


def _mark_loss_phases(
    ax,
    perceptual_warmup_epochs: int,
    adversarial_warmup_epochs: int,
) -> None:
    if perceptual_warmup_epochs == adversarial_warmup_epochs and perceptual_warmup_epochs > 0:
        ax.axvline(
            perceptual_warmup_epochs + 0.5,
            color="#888888",
            linestyle=":",
            alpha=0.7,
            label="perc + adv start",
        )
        return
    if perceptual_warmup_epochs > 0:
        ax.axvline(
            perceptual_warmup_epochs + 0.5,
            color="#2ca02c",
            linestyle=":",
            alpha=0.7,
            label="perceptual start",
        )
    if adversarial_warmup_epochs > perceptual_warmup_epochs:
        ax.axvline(
            adversarial_warmup_epochs + 0.5,
            color="#888888",
            linestyle=":",
            alpha=0.7,
            label="adversarial start",
        )


def _plot_series_panel(
    ax,
    epochs: list[int],
    series: dict[str, list[float]],
    *,
    title: str,
    ylabel: str,
    mark_best_val_key: str | None = None,
    perceptual_warmup_epochs: int = 0,
    adversarial_warmup_epochs: int = 0,
    warmup_epochs: int = 0,
) -> None:
    for key, values in series.items():
        if not values:
            continue
        if key in ("train_perceptual", "train_adv", "train_d") and all(v == 0.0 for v in values):
            continue
        if key.startswith("train"):
            color = "#1f77b4"
            if key == "train_adv":
                color = "#9467bd"
            elif key == "train_d":
                color = "#8c564b"
            elif key == "train_total":
                color = "#1f77b4"
            ax.plot(epochs, values, color=color, linewidth=1.8, label=key.replace("_", " "))
        else:
            ax.plot(
                epochs,
                values,
                color=VAL_STYLE["color"],
                linewidth=VAL_STYLE["linewidth"],
                label=key.replace("_", " "),
            )

    if mark_best_val_key and mark_best_val_key in series:
        val_y = series[mark_best_val_key]
        best_i = _best_val_index(val_y)
        if best_i is not None:
            ax.scatter(
                [epochs[best_i]],
                [val_y[best_i]],
                **BEST_STYLE,
                label=f"best @ {epochs[best_i]}",
            )
            ax.axvline(epochs[best_i], color=BEST_STYLE["color"], alpha=0.25, linestyle="--")

    perc_warmup = perceptual_warmup_epochs if perceptual_warmup_epochs else warmup_epochs
    adv_warmup = adversarial_warmup_epochs if adversarial_warmup_epochs else warmup_epochs
    _mark_loss_phases(ax, perc_warmup, adv_warmup)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)


def save_fix_window_sample_figure(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    out_path: Path | str,
    *,
    channel_names: tuple[str, ...] = DEFAULT_CHANNELS,
    residual_metric: str = "l1",
    title: str = "",
    loss_text: str = "",
    dpi: int = 120,
) -> Path:
    """Save GT | prediction | error panels for one +1d increment sample."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred_np = pred.detach().cpu().numpy()
    tgt_np = target.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()

    n_ch = min(len(channel_names), pred_np.shape[0])
    ch_names = channel_names[:n_ch]
    _, h, w = pred_np.shape

    col_labels = ["GT", "Prediction", "|GT - pred|" if residual_metric == "l1" else "Error"]
    if n_ch >= 3 and ch_names[2] == "speed":
        mean_err = _mean_abs_error(pred_np[:2], tgt_np[:2], mask_np)
    else:
        mean_err = _mean_abs_error(pred_np, tgt_np, mask_np)
    if np.isfinite(mean_err):
        col_labels[2] = f"{col_labels[2]} (mean={mean_err:.4f})"
    ncols = len(col_labels)
    fig, axes = plt.subplots(
        n_ch,
        ncols,
        figsize=_panel_figsize(h, w, n_ch, ncols),
        squeeze=False,
    )

    cmap_field = plt.cm.RdBu_r.copy()
    cmap_field.set_bad(color="#e8e8e8")
    cmap_err = plt.cm.viridis.copy()
    cmap_err.set_bad(color="#e8e8e8")
    last_im = None

    for row in range(n_ch):
        tgt_g = _apply_mask(tgt_np[row], mask_np)
        pred_g = _apply_mask(pred_np[row], mask_np)
        err_g = _apply_mask(_residual_error(pred_np[row], tgt_np[row], residual_metric), mask_np)
        gt_lim = _symmetric_limits(np.concatenate([tgt_g.ravel(), pred_g.ravel()]))

        panels = [
            (tgt_g, cmap_field, "field", gt_lim),
            (pred_g, cmap_field, "field", gt_lim),
            (err_g, cmap_err, "err", None),
        ]
        for col, (data, cmap, kind, lim) in enumerate(panels):
            ax = axes[row, col]
            if kind == "field":
                vmin, vmax = lim
            else:
                finite = data[np.isfinite(data)]
                vmax = float(np.nanpercentile(finite, 98)) if finite.size else 1.0
                vmin, vmax = 0.0, max(vmax, 1e-6)
            last_im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            if col < 2 and float(np.max(mask_np)) > 0:
                _add_land_contour(ax, mask_np)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(col_labels[col], fontsize=9)
            if col == 0:
                ax.set_ylabel(ch_names[row], fontsize=10)

    full_title = title
    if loss_text:
        full_title = f"{title}\n{loss_text}" if title else loss_text
    if full_title:
        fig.suptitle(full_title, y=1.04, fontsize=10)
    fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    fig.subplots_adjust(wspace=0.15, hspace=0.12)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_validation_mask_preview(
    val_dataset,
    output_path: Path | str,
    *,
    sample_indices: list[int] | None = None,
    n_samples: int = 3,
    seed: int = 42,
    global_index_offset: int = 0,
    global_sample_indices: list[int] | None = None,
    channel_names: tuple[str, ...] = ("u", "v"),
    loss_mask_buffer_px: int = 0,
    show: bool = False,
    dpi: int = 130,
) -> Path:
    """Show land mask on fixed validation samples (same indices as epoch viz)."""
    import matplotlib.pyplot as plt

    if sample_indices is None:
        sample_indices = fixed_viz_indices(n_samples, len(val_dataset), seed)
    if not sample_indices:
        raise ValueError("validation dataset is empty")

    n_show = len(sample_indices)
    n_ch = min(2, len(channel_names))
    col_labels = [
        "land mask\n(1=land, 0=ocean)",
        "loss mask\n(ocean=1)",
        "target u\n(NaN on land)",
        "target v\n(NaN on land)",
    ]
    fig, axes = plt.subplots(
        n_show,
        len(col_labels),
        figsize=(3.2 * len(col_labels), 2.8 * n_show + 0.8),
        squeeze=False,
    )
    mask_cmap, mask_norm = _mask_panel_cmap()
    cmap_field = plt.cm.RdBu_r.copy()
    cmap_field.set_bad(color="#d5d5d5")

    for row, ds_idx in enumerate(sample_indices):
        _x, y, m, _d, _h = val_dataset[ds_idx]
        mask_np = m.detach().cpu().numpy().astype(np.float32)
        loss_mask_np = mask_np.copy()
        if loss_mask_buffer_px > 0:
            from src.train.losses import expand_invalid_mask_for_loss

            loss_mask_np = (
                expand_invalid_mask_for_loss(torch.from_numpy(mask_np), loss_mask_buffer_px)
                .numpy()
                .squeeze()
            )
        valid_for_loss = (loss_mask_np == 0).astype(np.float32)

        y_np = y.detach().cpu().numpy()
        if global_sample_indices is not None:
            global_idx = int(global_sample_indices[row])
        else:
            global_idx = global_index_offset + ds_idx
        land_frac = float(mask_np.mean())
        valid_frac = float(valid_for_loss.mean())

        panels = [
            (mask_np, mask_cmap, mask_norm, "mask"),
            (valid_for_loss, mask_cmap, mask_norm, "mask"),
            (_apply_mask(y_np[0], mask_np), cmap_field, None, "field"),
            (_apply_mask(y_np[1], mask_np) if n_ch > 1 else y_np[0], cmap_field, None, "field"),
        ]

        for col, (data, cmap, norm, kind) in enumerate(panels):
            ax = axes[row, col]
            if kind == "mask":
                im = ax.imshow(data, cmap=cmap, norm=norm, interpolation="nearest", origin="lower")
            else:
                finite = data[np.isfinite(data)]
                if finite.size:
                    lim = max(abs(float(finite.min())), abs(float(finite.max())), 1e-6)
                    im = ax.imshow(
                        data,
                        cmap=cmap,
                        vmin=-lim,
                        vmax=lim,
                        interpolation="nearest",
                        origin="lower",
                    )
                else:
                    im = ax.imshow(data, cmap=cmap, interpolation="nearest", origin="lower")
            if col >= 2 and land_frac > 0:
                _add_land_contour(ax, mask_np, color="#111111", linewidth=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(col_labels[col], fontsize=9)
            if col == 0:
                ax.set_ylabel(
                    f"val idx {ds_idx}\n(global {global_idx})\nland {land_frac:.1%}",
                    fontsize=8,
                )

        axes[row, 1].set_xlabel(f"valid {valid_frac:.1%}", fontsize=7)

    fig.suptitle(
        "Validation mask check — gray/NaN = excluded from loss | contour = land boundary",
        fontsize=11,
        y=1.01,
    )
    fig.subplots_adjust(wspace=0.22, hspace=0.28)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def _center_slices(h: int, w: int, patch: int) -> tuple[slice, slice]:
    """Center crop slices; patch must be even for symmetric crop around center."""
    cy, cx = h // 2, w // 2
    half = patch // 2
    if patch % 2 == 0:
        return slice(cy - half, cy + half), slice(cx - half, cx + half)
    return slice(cy - half, cy + half + 1), slice(cx - half, cx + half + 1)


def _crop_center_patch(
    field: np.ndarray,
    sy: slice,
    sx: slice,
) -> np.ndarray:
    if field.ndim == 2:
        return field[sy, sx]
    return field[..., sy, sx]


def _l1_error_2d(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Plain per-pixel L1 error |target − pred| (no mask, no Gaussian weight)."""
    return np.abs(target - pred)


def _add_circle_outline(
    ax,
    h: int,
    w: int,
    radius: float,
    *,
    zorder: int = 10,
) -> None:
    """Single high-contrast circle at ``radius`` px from field center."""
    from matplotlib.patches import Circle

    if radius <= 0:
        return
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    ax.add_patch(
        Circle(
            (cx, cy),
            radius,
            fill=False,
            edgecolor="black",
            linewidth=3.0,
            linestyle="-",
            alpha=0.85,
            zorder=zorder,
        )
    )
    ax.add_patch(
        Circle(
            (cx, cy),
            radius,
            fill=False,
            edgecolor="white",
            linewidth=1.8,
            linestyle="-",
            alpha=1.0,
            zorder=zorder + 1,
        )
    )


def _add_gaussian_distance_circles(
    ax,
    h: int,
    w: int,
    sigma: float,
    *,
    radii: tuple[float, ...] | None = None,
) -> None:
    """Overlay σ-distance circles (high-contrast on field and error panels)."""
    from matplotlib.patches import Circle

    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    if radii is None:
        radii = (sigma, 2.0 * sigma)
    for r in radii:
        if r <= 0:
            continue
        # Dark underlay + bright ring for visibility on any colormap
        ax.add_patch(
            Circle(
                (cx, cy),
                r,
                fill=False,
                edgecolor="black",
                linewidth=3.2,
                linestyle="-",
                alpha=0.85,
                zorder=10,
            )
        )
        ax.add_patch(
            Circle(
                (cx, cy),
                r,
                fill=False,
                edgecolor="white",
                linewidth=2.0,
                linestyle="-",
                alpha=1.0,
                zorder=11,
            )
        )
    ax.plot(
        cx,
        cy,
        "+",
        color="#00ffff",
        markersize=16,
        markeredgewidth=2.5,
        markeredgecolor="black",
        zorder=12,
    )


def _normalize_field_01(field: np.ndarray) -> np.ndarray:
    """Per-panel min–max normalize finite values to [0, 1]; NaN unchanged."""
    out = field.copy()
    finite = np.isfinite(out)
    if not np.any(finite):
        return out
    lo, hi = float(np.min(out[finite])), float(np.max(out[finite]))
    if hi <= lo:
        out[finite] = 0.0
        return out
    out[finite] = (out[finite] - lo) / (hi - lo)
    return out


def _normalize_field_with_bounds(field: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map ``field`` to [0, 1] using shared ``lo``/``hi``; NaN unchanged."""
    out = field.copy()
    finite = np.isfinite(out)
    if not np.any(finite):
        return out
    if hi <= lo:
        out[finite] = 0.0
    else:
        out[finite] = (out[finite] - lo) / (hi - lo)
    return out


def _field_finite_bounds(field: np.ndarray) -> tuple[float, float]:
    finite = np.isfinite(field)
    if not np.any(finite):
        return 0.0, 1.0
    vals = field[finite]
    return float(np.min(vals)), float(np.max(vals))


def _mask_error_inside_circle(field: np.ndarray, h: int, w: int, radius: float) -> np.ndarray:
    """Keep signed error inside ``radius`` px from center; outside → NaN."""
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij",
    )
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    out = field.copy()
    out[dist2 > radius ** 2] = np.nan
    return out


def _zoom_circle_error_crop(
    err_field: np.ndarray,
    h: int,
    w: int,
    radius: float,
) -> np.ndarray:
    """Mask to r≤radius, then center-crop to 2×radius (zoomed; no full-field frame)."""
    err_masked = _mask_error_inside_circle(err_field, h, w, radius)
    patch = int(round(2.0 * radius))
    patch = max(4, patch)
    if patch % 2 == 1:
        patch += 1
    sy, sx = _center_slices(h, w, patch)
    return _crop_center_patch(err_masked, sy, sx)


DEFAULT_GAUSS_CIRCLE_RADII: tuple[float, ...] = (16.0, 32.0, 64.0)
DEFAULT_INNER_CIRCLE_RADII: tuple[float, ...] = (32.0, 16.0)
GAUSS_CELL_INCHES: float = 2.75  # panel cell size (width = height) before fig_scale


def save_gaussian_circles_sample_figure(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    out_path: Path | str,
    *,
    gaussian_sigma: float = 32.0,
    channel_names: tuple[str, ...] = DEFAULT_CHANNELS,
    residual_metric: str = "l1",
    title: str = "",
    loss_text: str = "",
    dpi: int = 150,
    mask_fields: bool = True,
    fig_scale: float = 2.6,
    circle_radii: tuple[float, ...] | None = None,
    inner_circle_radii: tuple[float, ...] | None = DEFAULT_INNER_CIRCLE_RADII,
    inner_circle_radius: float | None = None,  # deprecated: use inner_circle_radii
) -> Path:
    """Full-field GT | pred | error | zoomed inner-circle error cols.

    ``residual_metric="l1"``: |GT−pred| in physical units (vmin=0).
    ``residual_metric="signed"``: signed GT−pred per-row min→0, max→1 (legacy).
    Zoom cols: circle-masked error, center-cropped to 2×radius; same scale as col 3 per row.
    Distance circles on cols 1–3 only (default r=16, 32, 64 px).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred_np = pred.detach().cpu().numpy()
    tgt_np = target.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()

    n_ch = min(len(channel_names), pred_np.shape[0])
    ch_names = channel_names[:n_ch]
    h, w = pred_np.shape[-2], pred_np.shape[-1]

    if n_ch >= 3 and ch_names[2] == "speed":
        mean_err = _mean_abs_error(pred_np[:2], tgt_np[:2], mask_np)
    else:
        mean_err = _mean_abs_error(pred_np, tgt_np, mask_np)

    radii = circle_radii if circle_radii is not None else DEFAULT_GAUSS_CIRCLE_RADII
    if inner_circle_radii is None:
        if inner_circle_radius is not None and inner_circle_radius > 0:
            zoom_radii = (float(inner_circle_radius), 16.0)
        else:
            zoom_radii = ()
    else:
        zoom_radii = tuple(float(r) for r in inner_circle_radii if r > 0)

    use_signed_display = residual_metric == "signed"
    err_label = "|GT - pred|" if residual_metric in ("l1", "l1_smooth") else "Error"

    row_data: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for row in range(n_ch):
        tgt_row = _apply_mask(tgt_np[row], mask_np) if mask_fields else tgt_np[row]
        pred_row = _apply_mask(pred_np[row], mask_np) if mask_fields else pred_np[row]
        err_raw = _residual_error(pred_np[row], tgt_np[row], residual_metric)
        err_field = _apply_mask(err_raw, mask_np) if mask_fields else err_raw
        row_data.append((tgt_row, pred_row, err_field))

    col_labels = ["GT", "Prediction", err_label]
    if np.isfinite(mean_err):
        col_labels[2] = f"{err_label} (MAE={mean_err:.4f})"
    zoom_col_start = 3
    for r in zoom_radii:
        row_l1s = [
            _mean_abs_error_channel_in_circle(pred_np[row], tgt_np[row], mask_np, r)
            for row in range(n_ch)
        ]
        label = f"r≤{r:g}px"
        l1_parts = [
            f"{ch_names[i]}={row_l1s[i]:.4f}"
            for i in range(n_ch)
            if np.isfinite(row_l1s[i])
        ]
        if l1_parts:
            label = f"{label}\nL1: " + "   ".join(l1_parts)
        col_labels.append(label)
    ncols = len(col_labels)
    cell_in = GAUSS_CELL_INCHES * fig_scale
    fig_w = cell_in * ncols
    fig_h = cell_in * n_ch + 0.85 * fig_scale
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        n_ch,
        ncols,
        width_ratios=[1.0] * ncols,
        wspace=0.14,
        hspace=0.05,
        left=0.02,
        right=0.995,
        top=0.86,
        bottom=0.02,
    )
    axes = np.empty((n_ch, ncols), dtype=object)
    for row in range(n_ch):
        for col in range(ncols):
            axes[row, col] = fig.add_subplot(gs[row, col])

    cmap_field = plt.cm.RdBu_r.copy()
    cmap_field.set_bad(color="#e8e8e8")
    if use_signed_display:
        cmap_err = plt.cm.RdBu_r.copy()
    else:
        cmap_err = plt.cm.inferno.copy()
    cmap_err.set_bad(color="#e8e8e8")

    for row in range(n_ch):
        tgt_row, pred_row, err_field = row_data[row]
        gt_lim = _symmetric_limits(np.concatenate([tgt_row.ravel(), pred_row.ravel()]))

        if use_signed_display:
            err_display = _normalize_field_01(tgt_row - pred_row)
            err_lim = (0.0, 1.0)
        else:
            err_display = err_field
            finite = err_field[np.isfinite(err_field)]
            vmax = float(np.nanpercentile(finite, 98)) if finite.size else 1.0
            err_lim = (0.0, max(vmax, 1e-6))

        panels: list[tuple[np.ndarray, object, tuple[float, float] | None, bool, bool]] = [
            (tgt_row, cmap_field, gt_lim, True, False),
            (pred_row, cmap_field, gt_lim, True, False),
            (err_display, cmap_err, err_lim, True, False),
        ]
        zoom_panel_meta: list[tuple[float, int, int]] = []
        for zr in zoom_radii:
            err_zoom = _zoom_circle_error_crop(err_field, h, w, zr)
            if use_signed_display:
                err_lo, err_hi = _field_finite_bounds(tgt_row - pred_row)
                err_zoom_display = _normalize_field_with_bounds(err_zoom, err_lo, err_hi)
                zoom_lim = (0.0, 1.0)
            else:
                err_zoom_display = err_zoom
                zoom_lim = err_lim
            patch = err_zoom_display.shape[-1]
            zoom_panel_meta.append((zr, patch, patch))
            panels.append((err_zoom_display, cmap_err, zoom_lim, False, True))

        zoom_col_start = 3
        for col, (data, cmap, lim, draw_circles, is_zoom) in enumerate(panels):
            ax = axes[row, col]
            vmin, vmax = lim
            im = ax.imshow(
                data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", origin="lower",
            )
            if draw_circles:
                _add_gaussian_distance_circles(ax, h, w, gaussian_sigma, radii=radii)
            if is_zoom:
                z_idx = col - zoom_col_start
                zr, crop_h, crop_w = zoom_panel_meta[z_idx]
                _add_circle_outline(ax, crop_h, crop_w, zr)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("auto")
            if row == 0:
                if col >= zoom_col_start:
                    ax.set_title(
                        col_labels[col],
                        fontsize=14,
                        pad=12,
                        fontweight="bold",
                        linespacing=1.35,
                    )
                else:
                    ax.set_title(col_labels[col], fontsize=9, pad=3)
            if col == 0:
                ax.set_ylabel(ch_names[row], fontsize=10)
            if col >= 1:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    radii_note = ", ".join(f"{r:g}" for r in radii)
    if use_signed_display:
        if radii:
            circle_note = (
                f"distance circles r={radii_note}px (cols 1–3) | "
                "full-field error: per-row signed (GT−pred) min→0, max→1"
            )
        else:
            circle_note = "full-field error: per-row signed (GT−pred) min→0, max→1"
        if zoom_radii:
            zoom_note = ", ".join(f"r≤{r:g}" for r in zoom_radii)
            circle_note += (
                f" | zoom cols ({zoom_note}): cropped signed error, "
                "0–1 from full-field min/max per row"
            )
    else:
        if radii:
            circle_note = (
                f"distance circles r={radii_note}px (cols 1–3) | "
                f"error panels: {err_label} (L1 units, vmin=0)"
            )
        else:
            circle_note = f"error panels: {err_label} (L1 units, vmin=0)"
        if zoom_radii:
            zoom_note = ", ".join(f"r≤{r:g}" for r in zoom_radii)
            circle_note += (
                f" | zoom cols ({zoom_note}): cropped |GT−pred|, "
                "same 0–vmax scale as col 3; titles = mean L1 in circle"
            )
    full_title = title
    if loss_text:
        full_title = f"{title}\n{loss_text}" if title else loss_text
    if full_title:
        full_title = f"{full_title}\n{circle_note}"
    else:
        full_title = circle_note
    fig.suptitle(full_title, y=0.98, fontsize=9)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return out_path


def save_gaussian_circles_validation_figures(
    model: torch.nn.Module,
    dataset,
    device: torch.device | str,
    out_dir: Path | str,
    *,
    iD: int = 3,
    epoch: int | None = None,
    n_samples: int = 4,
    seed: int = 42,
    sample_indices: list[int] | None = None,
    channel_names: tuple[str, ...] = DEFAULT_CHANNELS,
    loss_metric: str = "components",
    epoch_losses: dict[str, float] | None = None,
    gaussian_sigma: float = 32.0,
    circle_radii: tuple[float, ...] | None = None,
    center_crop_patches: tuple[int, ...] | int | None = None,  # deprecated, ignored
    inner_circle_radii: tuple[float, ...] | None = DEFAULT_INNER_CIRCLE_RADII,
    inner_circle_radius: float | None = None,
    residual_metric: str = "l1",
    dpi: int = 150,
    fig_scale: float = 2.0,
    layout: str = "flat_samples",
    pred_cache_dir: Path | str | None = None,
) -> list[Path]:
    """Save Gaussian-circle validation PNGs (u, v, speed | GT | pred | error | crop errors)."""
    out_dir = Path(out_dir)
    if layout == "flat_samples":
        if epoch is None:
            raise ValueError("flat_samples layout requires epoch")
        sample_dir = out_dir / "samples_gauss_circles"
        sample_dir.mkdir(parents=True, exist_ok=True)

        def _sample_path(sample_idx: int, ds_idx: int) -> Path:
            return sample_dir / f"sample_{sample_idx:02d}_idx{ds_idx}_epoch{int(epoch):03d}.png"

    elif layout == "epoch_dirs":
        sample_dir = out_dir / f"epoch_{int(epoch - 1):03d}_gauss_circles" if epoch is not None else out_dir
        sample_dir.mkdir(parents=True, exist_ok=True)

        def _sample_path(sample_idx: int, ds_idx: int) -> Path:
            return sample_dir / f"sample_{sample_idx:02d}_idx{ds_idx}.png"

    else:
        raise ValueError(f"Unknown layout {layout!r}; use 'epoch_dirs' or 'flat_samples'")

    if sample_indices is not None:
        indices = list(sample_indices)
    else:
        indices = fixed_viz_indices(n_samples, len(dataset), seed)
    if not indices:
        return []

    was_training = model.training
    model.eval()
    paths: list[Path] = []
    radii = circle_radii if circle_radii is not None else DEFAULT_GAUSS_CIRCLE_RADII

    with torch.no_grad():
        for sample_idx, ds_idx in enumerate(indices):
            x, y, m, d, h = dataset[ds_idx]
            x_b = x.unsqueeze(0).to(device)
            d_b = d.unsqueeze(0).to(device)
            h_b = h.unsqueeze(0).to(device)
            pred = model(x_b[:, -3 * iD :], d_b + h_b / 365.0)[0, :iD].cpu()
            target = y[:iD]
            mask = m

            if pred_cache_dir is not None and epoch is not None:
                save_pred_cache(
                    pred_cache_path(Path(pred_cache_dir), epoch, ds_idx),
                    pred,
                    target,
                    mask,
                    epoch=epoch,
                    ds_idx=ds_idx,
                )

            sample_loss = _sample_reconstruction_loss(pred, target, mask, metric=loss_metric)
            loss_label = "sample speed L1" if loss_metric == "speed" else "sample uv L1"

            if iD >= 2:
                pred_plot, plot_names = _uv_with_speed(pred)
                target_plot, _ = _uv_with_speed(target)
            else:
                pred_plot = pred
                target_plot = target
                plot_names = channel_names[: pred.shape[0]]

            title = f"epoch {epoch} | val idx {ds_idx}" if epoch is not None else f"val idx {ds_idx}"
            loss_parts = [f"{loss_label}={sample_loss:.4f}"]
            if epoch_losses:
                l1_suffix = "speed L1" if loss_metric == "speed" else "uv L1"
                for key in ("val_l1", "train_l1", "train"):
                    if key in epoch_losses:
                        if key in ("val_l1", "train_l1"):
                            part = f"{key.replace('_l1', '')} {l1_suffix}"
                        else:
                            part = key
                        loss_parts.append(f"{part}={epoch_losses[key]:.4f}")
            loss_text = " | ".join(loss_parts)

            path = save_gaussian_circles_sample_figure(
                pred_plot,
                target_plot,
                mask,
                _sample_path(sample_idx, ds_idx),
                gaussian_sigma=gaussian_sigma,
                channel_names=plot_names,
                residual_metric=residual_metric,
                title=title,
                loss_text=loss_text,
                dpi=dpi,
                fig_scale=fig_scale,
                circle_radii=radii,
                inner_circle_radii=inner_circle_radii,
                inner_circle_radius=inner_circle_radius,
            )
            paths.append(path)

    if was_training:
        model.train()
    return paths


CIRCLE_L1_CHANNEL_COLORS: dict[str, str] = {
    "u": "#1f77b4",
    "v": "#ff7f0e",
    "speed": "#2ca02c",
}


def _circle_l1_channels_for_plot(
    iD: int,
    channel_names: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    """Channel names tracked in circle-L1 curves (u/v/speed layout or single field)."""
    names = tuple(channel_names[:iD])
    if iD >= 2 and len(names) >= 2 and names[0] == "u" and names[1] == "v":
        return ("u", "v", "speed")
    return names


def circle_l1_hist_key(channel: str, radius: float) -> str:
    """History key for circle L1 curves (matches zoom-column titles)."""
    return f"circle_l1_{channel}_{radius:g}"


def ensure_circle_l1_hist(
    hist: dict[str, Any],
    inner_circle_radii: tuple[float, ...],
    *,
    channel_names: tuple[str, ...] = ("u", "v", "speed"),
) -> None:
    """Ensure per-radius circle L1 series exist and align with ``hist['epoch']``."""
    n_epochs = len(hist.get("epoch") or [])
    ch_names = _circle_l1_channels_for_plot(len(channel_names), channel_names)
    for radius in inner_circle_radii:
        if radius <= 0:
            continue
        for channel in ch_names:
            key = circle_l1_hist_key(channel, radius)
            if key not in hist:
                hist[key] = [float("nan")] * n_epochs
            elif len(hist[key]) < n_epochs:
                hist[key] = hist[key] + [float("nan")] * (n_epochs - len(hist[key]))


def compute_viz_sample_circle_l1(
    model: torch.nn.Module,
    dataset,
    device: torch.device | str,
    sample_indices: list[int],
    iD: int,
    inner_circle_radii: tuple[float, ...],
    *,
    channel_names: tuple[str, ...] = ("u", "v"),
) -> dict[float, dict[str, float]]:
    """Mean circle L1 per channel on fixed viz val samples (cols 4–5 metrics)."""
    radii = tuple(float(r) for r in inner_circle_radii if r > 0)
    if not sample_indices or not radii:
        return {}

    plot_names = _circle_l1_channels_for_plot(iD, channel_names)
    agg: dict[float, dict[str, list[float]]] = {
        r: {ch: [] for ch in plot_names} for r in radii
    }
    was_training = model.training
    model.eval()

    with torch.no_grad():
        for ds_idx in sample_indices:
            pred, target, mask = forward_fix_window_sample(
                model, dataset, ds_idx, device, iD=iD,
            )
            if iD >= 2:
                pred_plot, plot_ch = _uv_with_speed(pred)
                target_plot, _ = _uv_with_speed(target)
            else:
                pred_plot = pred
                target_plot = target
                plot_ch = tuple(channel_names[:iD])
            pred_np = pred_plot.numpy()
            tgt_np = target_plot.numpy()
            mask_np = mask.numpy()
            for radius in radii:
                for row, ch in enumerate(plot_ch):
                    agg[radius][ch].append(
                        _mean_abs_error_channel_in_circle(
                            pred_np[row], tgt_np[row], mask_np, radius,
                        )
                    )

    if was_training:
        model.train()

    out: dict[float, dict[str, float]] = {}
    for radius in radii:
        out[radius] = {
            ch: float(np.nanmean(vals)) if vals else float("nan")
            for ch, vals in agg[radius].items()
        }
    return out


def save_gaussian_circle_l1_curves(
    hist: dict[str, Any],
    out_dir: Path | str,
    inner_circle_radii: tuple[float, ...],
    *,
    perceptual_warmup_epochs: int = 5,
    warmup_epochs: int = 5,
    viz_sample_note: str = "viz sample",
    dpi: int = 140,
) -> Path | None:
    """
    Save combined PNG: circle L1 curves for zoom cols 4–5 (u, v, speed per radius).

    Output: ``out_dir/curves/circle_l1_overview.png``
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = hist.get("epoch") or []
    radii = tuple(float(r) for r in inner_circle_radii if r > 0)
    if not epochs or not radii:
        return None

    has_data = any(
        hist.get(circle_l1_hist_key(ch, r))
        for r in radii
        for ch in CIRCLE_L1_CHANNEL_COLORS
        if hist.get(circle_l1_hist_key(ch, r))
    )
    if not has_data:
        # single-channel runs (u or v only)
        has_data = any(
            k.startswith("circle_l1_") and hist.get(k)
            for k in hist
        )
    if not has_data:
        return None

    curves_dir = Path(out_dir) / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    out_path = curves_dir / "circle_l1_overview.png"

    ncols = len(radii)
    fig, axes = plt.subplots(1, ncols, figsize=(6.2 * ncols, 4.8), squeeze=False)
    if ncols == 1:
        axes_flat = [axes[0, 0]]
    else:
        axes_flat = [axes[0, i] for i in range(ncols)]

    for ax, radius in zip(axes_flat, radii):
        series: dict[str, list[float]] = {}
        for ch in CIRCLE_L1_CHANNEL_COLORS:
            key = circle_l1_hist_key(ch, radius)
            values = hist.get(key) or []
            if values:
                series[ch] = values
        for key, values in hist.items():
            if not key.startswith("circle_l1_") or not values:
                continue
            if not key.endswith(f"_{radius:g}") and not key.endswith(f"_{int(radius)}"):
                continue
            ch = key[len("circle_l1_") : key.rfind("_")]
            if ch and ch not in series:
                series[ch] = values

        mean_series: list[float] = []
        for i in range(len(epochs)):
            vals = [
                series[ch][i]
                for ch in series
                if i < len(series[ch]) and np.isfinite(series[ch][i])
            ]
            mean_series.append(float(np.mean(vals)) if vals else float("nan"))

        for ch, values in series.items():
            n = min(len(epochs), len(values))
            if n == 0:
                continue
            color = CIRCLE_L1_CHANNEL_COLORS.get(ch, "#9467bd")
            ax.plot(
                epochs[:n],
                values[:n],
                color=color,
                linewidth=1.8,
                label=ch,
            )

        best_i = _best_val_index(mean_series)
        if best_i is not None:
            ax.scatter(
                [epochs[best_i]],
                [mean_series[best_i]],
                color="#2ca02c",
                marker="*",
                s=120,
                zorder=5,
                label=f"best @ {epochs[best_i]}",
            )
            ax.axvline(epochs[best_i], color="#2ca02c", alpha=0.25, linestyle="--")

        _mark_loss_phases(ax, perceptual_warmup_epochs, warmup_epochs)
        ax.set_xlabel("epoch")
        ax.set_ylabel("circle L1")
        ax.set_title(
            f"r≤{radius:g}px\n"
            f"L1 in ocean circle ({viz_sample_note})",
            fontsize=11,
            fontweight="bold",
        )
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    radii_note = ", ".join(f"r≤{r:g}" for r in radii)
    fig.suptitle(
        f"Circle L1 vs epoch | zoom cols 4–5 ({radii_note}) | mean |GT−pred| in circle",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def backfill_circle_l1_in_hist(
    hist: dict[str, Any],
    model: torch.nn.Module,
    val_set,
    viz_indices: list[int],
    iD: int,
    inner_circle_radii: tuple[float, ...],
    checkpoint_dir: Path | str,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """
    Recompute circle-L1 series in ``hist`` from saved epoch checkpoints.

    For each recorded epoch, uses the latest checkpoint at or before that epoch
    (stepwise; checkpoints every N epochs are enough for trend plots).
    """
    ckpt_dir = Path(checkpoint_dir)
    ckpt_epochs: dict[int, Path] = {}
    for path in ckpt_dir.glob("epoch_*.pth"):
        try:
            ckpt_epochs[int(path.stem.split("_")[1])] = path
        except ValueError:
            continue
    if not ckpt_epochs:
        return hist

    ensure_circle_l1_hist(hist, inner_circle_radii)
    epochs = hist.get("epoch") or []
    for i, ep in enumerate(epochs):
        prior = [e for e in ckpt_epochs if e <= int(ep)]
        if not prior:
            continue
        ck_ep = max(prior)
        state = torch.load(ckpt_epochs[ck_ep], map_location=device, weights_only=True)
        model.load_state_dict(state)
        metrics = compute_viz_sample_circle_l1(
            model,
            val_set,
            device,
            viz_indices,
            iD,
            tuple(inner_circle_radii),
        )
        for radius, ch_metrics in metrics.items():
            for ch, val in ch_metrics.items():
                hist[circle_l1_hist_key(ch, radius)][i] = val
    return hist


def save_center_patch_sample_figure(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    out_path: Path | str,
    *,
    center_patch: int = 32,
    channel_names: tuple[str, ...] = DEFAULT_CHANNELS,
    residual_metric: str = "l1",
    title: str = "",
    loss_text: str = "",
    dpi: int = 150,
    mask_fields: bool = True,
    fig_scale: float = 4.0,
) -> Path:
    """GT | pred | |GT−pred| L1 on a center crop. Error panel is never masked or ⊙G."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred_np = pred.detach().cpu().numpy()
    tgt_np = target.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()

    n_ch = min(len(channel_names), pred_np.shape[0])
    ch_names = channel_names[:n_ch]
    _, h, w = pred_np.shape
    sy, sx = _center_slices(h, w, center_patch)

    pred_c = _crop_center_patch(pred_np, sy, sx)
    tgt_c = _crop_center_patch(tgt_np, sy, sx)
    mask_c = _crop_center_patch(mask_np, sy, sx)

    if n_ch >= 3 and ch_names[2] == "speed":
        mean_err = _mean_abs_error(pred_c[:2], tgt_c[:2], mask_c)
    else:
        mean_err = _mean_abs_error(pred_c, tgt_c, mask_c)

    col_labels = ["GT", "Prediction", "|GT - pred|"]
    if np.isfinite(mean_err):
        col_labels[2] = f"{col_labels[2]} (ocean mean={mean_err:.4f})"
    ncols = len(col_labels)
    ph, pw = pred_c.shape[-2], pred_c.shape[-1]
    base_w, base_h = _panel_figsize(ph, pw, n_ch, ncols)
    fig, axes = plt.subplots(
        n_ch,
        ncols,
        figsize=(base_w * fig_scale, base_h * fig_scale),
        squeeze=False,
    )

    cmap_field = plt.cm.RdBu_r.copy()
    cmap_field.set_bad(color="#e8e8e8")
    cmap_err = plt.cm.inferno.copy()
    last_im = None

    for row in range(n_ch):
        tgt_row = _apply_mask(tgt_c[row], mask_c) if mask_fields else tgt_c[row]
        pred_row = _apply_mask(pred_c[row], mask_c) if mask_fields else pred_c[row]
        err_l1 = _l1_error_2d(pred_c[row], tgt_c[row])
        gt_lim = _symmetric_limits(np.concatenate([tgt_row.ravel(), pred_row.ravel()]))

        panels = [
            (tgt_row, cmap_field, "field", gt_lim),
            (pred_row, cmap_field, "field", gt_lim),
            (err_l1, cmap_err, "err", None),
        ]
        for col, (data, cmap, kind, lim) in enumerate(panels):
            ax = axes[row, col]
            if kind == "field":
                vmin, vmax = lim
            else:
                vmax = float(np.nanpercentile(data, 98)) if data.size else 1.0
                vmin, vmax = 0.0, max(vmax, 1e-6)
            last_im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(col_labels[col], fontsize=9)
            if col == 0:
                ax.set_ylabel(ch_names[row], fontsize=10)

    full_title = title
    if loss_text:
        full_title = f"{title}\n{loss_text}" if title else loss_text
    if full_title:
        fig.suptitle(full_title, y=1.04, fontsize=10)
    fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    fig.subplots_adjust(wspace=0.15, hspace=0.12)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_gaussian_weighted_sample_figure(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    gauss: torch.Tensor | np.ndarray,
    out_path: Path | str,
    *,
    channel_names: tuple[str, ...] = DEFAULT_CHANNELS,
    residual_metric: str = "l1",
    title: str = "",
    loss_text: str = "",
    dpi: int = 120,
) -> Path:
    """GT | pred | (|error|⊙G) panels — error column uses Gaussian loss weight."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred_np = pred.detach().cpu().numpy()
    tgt_np = target.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()
    if isinstance(gauss, torch.Tensor):
        gauss_np = gauss.detach().cpu().numpy()
    else:
        gauss_np = np.asarray(gauss, dtype=np.float64)
    if gauss_np.ndim > 2:
        gauss_np = gauss_np.squeeze()

    n_ch = min(len(channel_names), pred_np.shape[0])
    ch_names = channel_names[:n_ch]
    _, h, w = pred_np.shape

    if n_ch >= 3 and ch_names[2] == "speed":
        mean_err = _mean_gaussian_weighted_abs_error(pred_np[:2], tgt_np[:2], mask_np, gauss_np)
    else:
        mean_err = _mean_gaussian_weighted_abs_error(pred_np, tgt_np, mask_np, gauss_np)

    col_labels = ["GT", "Prediction", "|GT - pred| ⊙ G"]
    if np.isfinite(mean_err):
        col_labels[2] = f"{col_labels[2]} (wmean={mean_err:.4f})"
    ncols = len(col_labels)
    fig, axes = plt.subplots(
        n_ch,
        ncols,
        figsize=_panel_figsize(h, w, n_ch, ncols),
        squeeze=False,
    )

    cmap_field = plt.cm.RdBu_r.copy()
    cmap_field.set_bad(color="#e8e8e8")
    cmap_err = plt.cm.inferno.copy()
    cmap_err.set_bad(color="#e8e8e8")
    last_im = None

    for row in range(n_ch):
        tgt_g = _apply_mask(tgt_np[row], mask_np)
        pred_g = _apply_mask(pred_np[row], mask_np)
        raw_err = _residual_error(pred_np[row], tgt_np[row], residual_metric)
        err_weighted = raw_err * gauss_np
        err_g = _apply_mask(err_weighted, mask_np)
        gt_lim = _symmetric_limits(np.concatenate([tgt_g.ravel(), pred_g.ravel()]))

        panels = [
            (tgt_g, cmap_field, "field", gt_lim),
            (pred_g, cmap_field, "field", gt_lim),
            (err_g, cmap_err, "err", None),
        ]
        for col, (data, cmap, kind, lim) in enumerate(panels):
            ax = axes[row, col]
            if kind == "field":
                vmin, vmax = lim
            else:
                finite = data[np.isfinite(data)]
                vmax = float(np.nanpercentile(finite, 98)) if finite.size else 1.0
                vmin, vmax = 0.0, max(vmax, 1e-6)
            last_im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(col_labels[col], fontsize=9)
            if col == 0:
                ax.set_ylabel(ch_names[row], fontsize=10)

    full_title = title
    if loss_text:
        full_title = f"{title}\n{loss_text}" if title else loss_text
    if full_title:
        fig.suptitle(full_title, y=1.04, fontsize=10)
    fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    fig.subplots_adjust(wspace=0.15, hspace=0.12)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def forward_fix_window_sample(
    model: torch.nn.Module,
    dataset,
    ds_idx: int,
    device: torch.device | str,
    *,
    iD: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run +1d increment forward pass for one validation sample."""
    model.eval()
    x, y, m, d, h = dataset[ds_idx]
    with torch.no_grad():
        x_b = x.unsqueeze(0).to(device)
        d_b = d.unsqueeze(0).to(device)
        h_b = h.unsqueeze(0).to(device)
        pred = model(x_b[:, -3 * iD :], d_b + h_b / 365.0)[0, :iD].cpu()
    return pred, y[:iD], m


def pred_cache_path(cache_dir: Path, epoch: int, ds_idx: int) -> Path:
    return cache_dir / f"epoch_{int(epoch):03d}_idx{int(ds_idx)}.pt"


def save_pred_cache(
    cache_path: Path,
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    epoch: int,
    ds_idx: int,
) -> Path:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"epoch": int(epoch), "ds_idx": int(ds_idx), "pred": pred, "target": target, "mask": mask},
        cache_path,
    )
    return cache_path


def load_pred_cache(cache_path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = torch.load(cache_path, map_location="cpu", weights_only=False)
    return data["pred"], data["target"], data["mask"]


def fixed_viz_indices(n_samples: int, dataset_size: int, seed: int = 42) -> list[int]:
    """Pick fixed validation indices once; same list every epoch for comparable viz."""
    n = min(n_samples, dataset_size)
    if n == 0:
        return []
    rng = random.Random(seed)
    return sorted(rng.sample(range(dataset_size), n))


def save_fix_window_validation_figures(
    model: torch.nn.Module,
    dataset,
    device: torch.device | str,
    out_dir: Path | str,
    *,
    iD: int = 3,
    epoch: int | None = None,
    n_samples: int = 4,
    seed: int = 42,
    sample_indices: list[int] | None = None,
    residual_metric: str = "l1",
    channel_names: tuple[str, ...] = DEFAULT_CHANNELS,
    loss_metric: str = "components",
    epoch_losses: dict[str, float] | None = None,
    dpi: int = 120,
    layout: str = "epoch_dirs",
    pred_cache_dir: Path | str | None = None,
) -> list[Path]:
    """
    Save GT / pred / L1-error PNGs for fixed validation samples.

    layout:
      - ``epoch_dirs``: ``viz/epoch_XXX/sample_00_idx57.png`` (legacy)
      - ``flat_samples``: ``viz/samples/sample_00_idx57_epoch001.png`` (faster, one folder)

    Uses the +1d normalized increment target that training optimizes.
    Pass ``sample_indices`` (or a fixed ``seed``) so the same val cases are plotted every epoch.
    """
    out_dir = Path(out_dir)
    if layout == "flat_samples":
        if epoch is None:
            raise ValueError("flat_samples layout requires epoch")
        sample_dir = out_dir / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)

        def _sample_path(sample_idx: int, ds_idx: int) -> Path:
            return sample_dir / f"sample_{sample_idx:02d}_idx{ds_idx}_epoch{int(epoch):03d}.png"

    elif layout == "epoch_dirs":
        sample_dir = out_dir / f"epoch_{int(epoch - 1):03d}" if epoch is not None else out_dir
        sample_dir.mkdir(parents=True, exist_ok=True)

        def _sample_path(sample_idx: int, ds_idx: int) -> Path:
            return sample_dir / f"sample_{sample_idx:02d}_idx{ds_idx}.png"

    else:
        raise ValueError(f"Unknown layout {layout!r}; use 'epoch_dirs' or 'flat_samples'")

    if sample_indices is not None:
        indices = list(sample_indices)
    else:
        indices = fixed_viz_indices(n_samples, len(dataset), seed)
    if not indices:
        return []

    was_training = model.training
    model.eval()
    paths: list[Path] = []

    with torch.no_grad():
        for sample_idx, ds_idx in enumerate(indices):
            x, y, m, d, h = dataset[ds_idx]
            x_b = x.unsqueeze(0).to(device)
            d_b = d.unsqueeze(0).to(device)
            h_b = h.unsqueeze(0).to(device)
            pred = model(x_b[:, -3 * iD :], d_b + h_b / 365.0)[0, :iD].cpu()
            target = y[:iD]
            mask = m

            if pred_cache_dir is not None and epoch is not None:
                save_pred_cache(
                    pred_cache_path(Path(pred_cache_dir), epoch, ds_idx),
                    pred,
                    target,
                    mask,
                    epoch=epoch,
                    ds_idx=ds_idx,
                )

            sample_loss = _sample_reconstruction_loss(pred, target, mask, metric=loss_metric)
            loss_label = "sample speed L1" if loss_metric == "speed" else "sample uv L1"

            # Plots: always show u, v, speed (with per-row 2D errors) when we have u,v
            if iD >= 2:
                pred_plot, plot_names = _uv_with_speed(pred)
                target_plot, _ = _uv_with_speed(target)
            else:
                pred_plot = pred
                target_plot = target
                plot_names = channel_names[: pred.shape[0]]

            title = f"epoch {epoch} | val idx {ds_idx}" if epoch is not None else f"val idx {ds_idx}"
            loss_parts = [f"{loss_label}={sample_loss:.4f}"]
            if epoch_losses:
                l1_suffix = "speed L1" if loss_metric == "speed" else "uv L1"
                for key in ("val_l1", "train_l1", "train", "train_perceptual", "train_adv", "train_d"):
                    if key in epoch_losses:
                        if key in ("val_l1", "train_l1"):
                            part = f"{key.replace('_l1', '')} {l1_suffix}"
                        else:
                            part = key
                        loss_parts.append(f"{part}={epoch_losses[key]:.4f}")
            loss_text = " | ".join(loss_parts)

            path = save_fix_window_sample_figure(
                pred_plot,
                target_plot,
                mask,
                _sample_path(sample_idx, ds_idx),
                channel_names=plot_names,
                residual_metric=residual_metric,
                title=title,
                loss_text=loss_text,
                dpi=dpi,
            )
            paths.append(path)

    if was_training:
        model.train()
    return paths


def display_adv_no_epoch_plots(
    viz_paths: list[Path],
    curve_paths: list[Path] | None = None,
    *,
    show_curves: bool = True,
    show_samples: bool = True,
) -> None:
    """Show latest epoch PNGs inline when running inside Jupyter."""
    try:
        from IPython.display import Image, display
    except ImportError:
        return

    curve_paths = curve_paths or []
    if show_curves:
        overview = next((p for p in curve_paths if p.name == "loss_overview.png"), None)
        if overview is not None and overview.exists():
            display(Image(filename=str(overview)))

    if show_samples:
        for path in viz_paths:
            if path.exists():
                display(Image(filename=str(path)))


def save_adv_no_training_curves(
    hist: dict[str, Any],
    out_dir: Path | str,
    *,
    perceptual_warmup_epochs: int = 5,
    warmup_epochs: int = 5,
    loss_metric: str = "components",
    val_reconstruction_metric: str = "l1",
    perceptual_label: str = "VGG-19",
    dpi: int = 140,
) -> list[Path]:
    """
    Save/update loss curve PNGs under ``out_dir/curves/``.

    Files: overview, l1, total, perceptual, adversarial, lr.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = hist.get("epoch") or []
    if not epochs:
        return []

    recon_label = "speed L1" if loss_metric == "speed" else "uv L1"
    val_recon_label = (
        ("speed RMSE" if loss_metric == "speed" else "uv RMSE")
        if val_reconstruction_metric == "rmse"
        else recon_label
    )
    val_series_key = (
        "val_rmse"
        if val_reconstruction_metric == "rmse" and hist.get("val_rmse")
        else "val_l1"
    )
    recon_short = val_recon_label if val_series_key == "val_rmse" else recon_label

    curves_dir = Path(out_dir) / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    def _save(fig, name: str) -> Path:
        path = curves_dir / name
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
        return path

    # --- L1 ---
    fig, ax = plt.subplots(figsize=(7, 4))
    _plot_series_panel(
        ax,
        epochs,
        {"train_l1": hist["train_l1"], val_series_key: hist[val_series_key]},
        title=f"Reconstruction (train {recon_label}, val {val_recon_label})",
        ylabel=recon_short,
        mark_best_val_key=val_series_key,
        perceptual_warmup_epochs=perceptual_warmup_epochs,
        adversarial_warmup_epochs=warmup_epochs,
    )
    _save(fig, "loss_l1.png")

    # --- Total generator loss ---
    fig, ax = plt.subplots(figsize=(7, 4))
    _plot_series_panel(
        ax,
        epochs,
        {"train_total": hist["train"]},
        title=f"Generator total loss ({recon_short} + perc + β·adv)",
        ylabel="total",
        perceptual_warmup_epochs=perceptual_warmup_epochs,
        adversarial_warmup_epochs=warmup_epochs,
    )
    _save(fig, "loss_total.png")

    # --- Perceptual ---
    fig, ax = plt.subplots(figsize=(7, 4))
    _plot_series_panel(
        ax,
        epochs,
        {"train_perceptual": hist["train_perceptual"], "val_perceptual": hist["val_perceptual"]},
        title=f"Perceptual ({perceptual_label})",
        ylabel="perceptual",
        perceptual_warmup_epochs=perceptual_warmup_epochs,
        adversarial_warmup_epochs=warmup_epochs,
    )
    _save(fig, "loss_perceptual.png")

    # --- Adversarial ---
    fig, ax = plt.subplots(figsize=(7, 4))
    _plot_series_panel(
        ax,
        epochs,
        {"train_adv": hist["train_adv"], "train_d": hist["train_d"]},
        title="Adversarial (generator adv + discriminator D)",
        ylabel="GAN loss",
        perceptual_warmup_epochs=perceptual_warmup_epochs,
        adversarial_warmup_epochs=warmup_epochs,
    )
    _save(fig, "loss_adversarial.png")

    # --- Learning rate ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, hist["lr"], color="#d62728", linewidth=1.8)
    _mark_loss_phases(ax, perceptual_warmup_epochs, warmup_epochs)
    ax.set_xlabel("epoch")
    ax.set_ylabel("LR")
    ax.set_title("Generator learning rate")
    ax.grid(True, alpha=0.3)
    if perceptual_warmup_epochs > 0 or warmup_epochs > 0:
        ax.legend(loc="upper right", fontsize=8)
    _save(fig, "loss_lr.png")

    # --- Overview 2x2 ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    _plot_series_panel(
        axes[0, 0],
        epochs,
        {"train_l1": hist["train_l1"], val_series_key: hist[val_series_key]},
        title=recon_short,
        ylabel=recon_short,
        mark_best_val_key=val_series_key,
        perceptual_warmup_epochs=perceptual_warmup_epochs,
        adversarial_warmup_epochs=warmup_epochs,
    )
    _plot_series_panel(
        axes[0, 1],
        epochs,
        {"train_total": hist["train"]},
        title="Total",
        ylabel="total",
        perceptual_warmup_epochs=perceptual_warmup_epochs,
        adversarial_warmup_epochs=warmup_epochs,
    )
    _plot_series_panel(
        axes[1, 0],
        epochs,
        {"train_perceptual": hist["train_perceptual"], "val_perceptual": hist["val_perceptual"]},
        title="Perceptual",
        ylabel="perceptual",
        perceptual_warmup_epochs=perceptual_warmup_epochs,
        adversarial_warmup_epochs=warmup_epochs,
    )
    _plot_series_panel(
        axes[1, 1],
        epochs,
        {"train_adv": hist["train_adv"], "train_d": hist["train_d"]},
        title="Adversarial",
        ylabel="GAN",
        perceptual_warmup_epochs=perceptual_warmup_epochs,
        adversarial_warmup_epochs=warmup_epochs,
    )
    best = hist.get("best_epoch", 0)
    best_val = hist.get("best_val_loss", float("nan"))
    fig.suptitle(
        f"adv-NO training | best val {val_recon_label} {best_val:.4f} @ epoch {best}",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    _save(fig, "loss_overview.png")

    return saved


def save_loss_mask_preview(
    data_prefix: str,
    output_path: Path,
    *,
    sample_index: int = 0,
    loss_spatial_mode: str = "full",
    loss_gaussian_sigma: float = 32.0,
    loss_mask_buffer_px: int = 0,
    channel_names: tuple[str, ...] = ("u", "v"),
    distance_circle_radii: tuple[float, ...] | None = None,
    show: bool = False,
) -> Path:
    """Show how Gaussian center weighting enters the loss (state ⊙ G, center > edges)."""
    import matplotlib.pyplot as plt

    circle_radii = distance_circle_radii if distance_circle_radii is not None else DEFAULT_GAUSS_CIRCLE_RADII

    sims_np = np.asarray(np.load(f"{data_prefix}_sims.npy", mmap_mode="r"))
    if sims_np.size == 0:
        raise ValueError(f"No samples in {data_prefix}")

    idx = int(np.clip(sample_index, 0, len(sims_np) - 1))
    h, w = sims_np.shape[-2], sims_np.shape[-1]
    coastal_mask = None
    coastal_frac = 0.0
    mask_path = Path(f"{data_prefix}_mask.npy")
    if mask_path.exists():
        coastal_mask = np.load(mask_path, mmap_mode="r")[idx].astype(np.float32)
        coastal_frac = float(coastal_mask.mean())
        if loss_mask_buffer_px > 0:
            from src.train.losses import expand_invalid_mask_for_loss

            coastal_mask = (
                expand_invalid_mask_for_loss(torch.from_numpy(coastal_mask), loss_mask_buffer_px)
                .numpy()
                .squeeze()
            )
            coastal_frac = float(coastal_mask.mean())
    n_ch = min(2, sims_np.shape[2], len(channel_names))
    names = channel_names[:n_ch]
    n_t = int(sims_np.shape[1])
    if n_t <= 4:
        t_in, t_out = n_t - 2, n_t - 1
    else:
        t_in, t_out = 3, 4

    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2

    weight_t = build_loss_spatial_weight(loss_spatial_mode, h, w, loss_gaussian_sigma)
    ring_stats: dict[str, float] = {}
    g_at_radii: dict[float, float] = {}
    if weight_t is not None:
        gauss_display = np.exp(-dist2 / (2.0 * loss_gaussian_sigma**2))
        gauss = weight_t.numpy()
        ring_stats = gaussian_weight_ring_summary(weight_t, ring_radii=circle_radii)
        g_at_radii = {r: gaussian_weight_at_radius(weight_t, r) for r in circle_radii}
        radii_note = ", ".join(f"{r:g}" for r in circle_radii)
        gauss_title = (
            f"G(r) = exp(−r²/2σ²) mean-norm\n"
            f"σ={loss_gaussian_sigma}px | circles r={radii_note}px"
        )
    else:
        gauss = np.ones((h, w), dtype=np.float32)
        gauss_display = gauss.copy()
        gauss_title = "G = 1  (uniform)"

    if coastal_mask is not None:
        valid = 1.0 - coastal_mask
        gauss = gauss * valid
        gauss_display = gauss_display * valid
        gauss_title += f"\n× valid ocean (land {coastal_frac:.1%} masked out)"
        if loss_mask_buffer_px > 0:
            gauss_title += f" + {loss_mask_buffer_px}px coastal buffer"

    gauss_vis = gauss_display / (float(gauss_display.max()) or 1.0)
    edge_g = float(gauss[h // 2, 0])
    center_g = float(gauss[h // 2, w // 2])

    # 2 rows (u, v) × 3 cols: full state | G | state ⊙ G
    fig = plt.figure(figsize=(13.5, 4.6 * n_ch + 1.2))
    gs = fig.add_gridspec(n_ch, 4, width_ratios=[1.0, 1.0, 1.0, 1.15], wspace=0.28, hspace=0.35)

    for row, ch in enumerate(range(n_ch)):
        field = (sims_np[idx, t_out, ch] - sims_np[idx, t_in, ch]).astype(np.float64)
        weighted = field * gauss
        vmin, vmax = _symmetric_limits(field)
        wlim = max(abs(weighted.min()), abs(weighted.max()), 1e-6)

        ax0 = fig.add_subplot(gs[row, 0])
        ax0.imshow(field, cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower")
        if loss_spatial_mode == "gaussian_center":
            _add_gaussian_distance_circles(ax0, h, w, loss_gaussian_sigma, radii=circle_radii)
        ax0.set_title(f"{names[ch]}  (full state)")
        ax0.set_ylabel(names[ch], fontsize=11, fontweight="bold")
        ax0.set_xticks([])
        ax0.set_yticks([])
        fig.colorbar(ax0.images[0], ax=ax0, fraction=0.046, pad=0.04)

        ax1 = fig.add_subplot(gs[row, 1])
        if row == 0:
            im1 = ax1.imshow(gauss_vis, cmap="inferno", vmin=0, vmax=1, origin="lower")
            if loss_spatial_mode == "gaussian_center":
                _add_gaussian_distance_circles(ax1, h, w, loss_gaussian_sigma, radii=circle_radii)
            ax1.set_title(gauss_title)
            fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="G (rel.)")
        else:
            ax1.axis("off")
            ax1.text(
                0.5, 0.5, "same G for\nall channels",
                ha="center", va="center", fontsize=10, transform=ax1.transAxes,
            )
        ax1.set_xticks([])
        ax1.set_yticks([])

        ax2 = fig.add_subplot(gs[row, 2])
        im2 = ax2.imshow(weighted, cmap="RdBu_r", vmin=-wlim, vmax=wlim, origin="lower")
        if loss_spatial_mode == "gaussian_center":
            _add_gaussian_distance_circles(ax2, h, w, loss_gaussian_sigma, radii=circle_radii)
        loss_title = f"{names[ch]} ⊙ G"
        if coastal_mask is not None:
            loss_title += "\n(land masked out)"
        elif loss_spatial_mode == "gaussian_center":
            loss_title += "\n(edges suppressed)"
        ax2.set_title(loss_title)
        ax2.set_xticks([])
        ax2.set_yticks([])
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        if row == 0:
            ax3 = fig.add_subplot(gs[row, 3])
            mid = h // 2
            x_off = np.arange(w) - cx
            ax3.plot(x_off, gauss[mid, :], color="#e6550d", linewidth=2.0, label="G (training)")
            ax3.plot(
                x_off, gauss_display[mid, :], color="#31a354", linewidth=1.2,
                linestyle="--", alpha=0.85, label="G (raw, same shape)",
            )
            if loss_spatial_mode == "gaussian_center":
                for r in circle_radii:
                    ax3.axvline(r, color="#888888", linewidth=0.8, linestyle=":", alpha=0.7)
                    ax3.axvline(-r, color="#888888", linewidth=0.8, linestyle=":", alpha=0.7)
            ax3.legend(loc="upper right", fontsize=7)
            ax3.set_xlabel("distance from center (px)")
            ax3.set_ylabel("G (training weight)")
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, float(gauss.max()) * 1.08)
            if loss_spatial_mode == "gaussian_center":
                ax3.set_title("Center prioritized\nover distant points")
                g_lines = [f"G(center)={center_g:.2f}", f"G(edge)={edge_g:.2f}"]
                for r in circle_radii:
                    g_lines.append(f"G(r={r:g})={g_at_radii[r]:.2f}")
                ring_lines = [f"{k}: {v:.2f}" for k, v in ring_stats.items()]
                ax3.annotate(
                    "\n".join(g_lines)
                    + f"\n≈{center_g / max(edge_g, 1e-6):.1f}× center vs edge\n"
                    + "Ring mean G:\n" + "\n".join(ring_lines),
                    xy=(0.03, 0.97), xycoords="axes fraction", va="top", fontsize=7,
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9),
                )
            else:
                ax3.set_title("Uniform G\n(flat profile)")
        elif row == 1:
            ax3 = fig.add_subplot(gs[row, 3])
            ax3.axis("off")
            if coastal_mask is not None:
                formula = (
                    "How loss uses weights (training):\n\n"
                    "• valid = (mask == 0) ocean pixels\n"
                    "• L1: mean( |pred − target| ⊙ valid [⊙ G] )\n\n"
                    f"• Land/coastal invalid: {coastal_frac:.1%}\n"
                    "  of crop — excluded from loss.\n\n"
                    "Inputs/targets zeroed on land\n"
                    "after global mean/std norm\n"
                    "(Guiana-style masked L1)."
                )
            else:
                formula = (
                    "How loss uses G (training):\n\n"
                    "• L1:  mean( |pred − target| ⊙ G )\n"
                    "       (center pixels count more)\n\n"
                    "• Inner rings (r≤16) get higher G\n"
                    "  than outer rings (r>64).\n\n"
                    "Model still outputs the full\n"
                    "128×128 state; only the loss\n"
                    "is center-focused."
                )
            ax3.text(
                0.0, 1.0, formula, va="top", ha="left", fontsize=9.5,
                family="monospace", transform=ax3.transAxes,
                bbox=dict(boxstyle="round", facecolor="#f0f4ff", alpha=0.95),
            )

    mode_note = (
        f"{loss_spatial_mode}"
        + (f", σ={loss_gaussian_sigma}px" if loss_spatial_mode == "gaussian_center" else "")
    )
    radii_suptitle = ", ".join(f"{r:g}" for r in circle_radii)
    fig.suptitle(
        f"Loss focus: multiply state by Gaussian G | sample {idx} | {mode_note}\n"
        f"Distance circles r={radii_suptitle}px | "
        r"Training $G(r)=\exp(-r^2/2\sigma^2)$ mean-normalized; inner rings weighted higher",
        fontsize=11,
        y=1.01,
    )
    fig.subplots_adjust(top=0.88, hspace=0.35, wspace=0.28)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path
