"""Save GT vs prediction maps on the graph grid (32×32 nodes)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

CHANNEL_NAMES = ("u", "v", "ssh")


def _slice_lead_channels(feat: np.ndarray, lead_idx: int = 0, n_ch: int = 3) -> np.ndarray:
    """Take u,v,ssh for one forecast lead from stacked (N, forecast_steps*3)."""
    feat = np.asarray(feat)
    if feat.shape[1] == n_ch:
        return feat
    start = lead_idx * n_ch
    return feat[:, start : start + n_ch]


def nodes_to_grid(feat: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    """(N, C) or (N,) -> (C, grid_h, grid_w)."""
    feat = np.asarray(feat)
    if feat.ndim == 1:
        feat = feat[:, None]
    n, c = feat.shape
    if n != grid_h * grid_w:
        raise ValueError(f"Expected {grid_h * grid_w} nodes, got {n}")
    return feat.T.reshape(c, grid_h, grid_w)


def masked_grid(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(C, H, W) with invalid nodes as NaN."""
    out = data.copy()
    valid = mask >= 0.5
    for c in range(out.shape[0]):
        out[c] = np.where(valid, out[c], np.nan)
    return out


def save_pred_gt_figure(
    pred: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    node_mask: np.ndarray | torch.Tensor,
    out_path: Path | str,
    *,
    title: str = "",
    grid_h: int | None = None,
    grid_w: int | None = None,
    show_error: bool = True,
    lead_idx: int = 0,
    dpi: int = 120,
) -> Path:
    """
    Save a figure with GT, prediction, and optional |error| per channel.

    Rows = u, v, ssh. Columns = ground truth | prediction [| abs error].
    """
    pred = _slice_lead_channels(torch.as_tensor(pred).detach().cpu().numpy(), lead_idx)
    target = _slice_lead_channels(torch.as_tensor(target).detach().cpu().numpy(), lead_idx)
    node_mask = torch.as_tensor(node_mask).detach().cpu().numpy().reshape(-1)

    n = node_mask.shape[0]
    if grid_h is None or grid_w is None:
        grid_h = grid_w = int(n**0.5)
    mask_g = node_mask.reshape(grid_h, grid_w)

    pred_g = masked_grid(nodes_to_grid(pred, grid_h, grid_w), mask_g)
    tgt_g = masked_grid(nodes_to_grid(target, grid_h, grid_w), mask_g)
    lead_suffix = f" | lead +{lead_idx}" if lead_idx else ""

    ncols = 3 if show_error else 2
    col_labels = ["Ground truth", "Prediction", "|Error|"][:ncols]
    fig, axes = plt.subplots(3, ncols, figsize=(3.2 * ncols, 7.5), squeeze=False)

    vmin = np.nanpercentile(np.concatenate([tgt_g.ravel(), pred_g.ravel()]), 2)
    vmax = np.nanpercentile(np.concatenate([tgt_g.ravel(), pred_g.ravel()]), 98)
    err_g = np.abs(pred_g - tgt_g) if show_error else None

    last_im = None
    for row, ch_name in enumerate(CHANNEL_NAMES):
        panels = [tgt_g[row], pred_g[row]]
        if show_error:
            panels.append(err_g[row])
        for col, data in enumerate(panels):
            ax = axes[row, col]
            last_im = ax.imshow(data, cmap="RdBu_r", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(col_labels[col], fontsize=10)
            if col == 0:
                ax.set_ylabel(ch_name, fontsize=10)

    if title:
        fig.suptitle(title + lead_suffix, y=1.01, fontsize=11)
    fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    fig.subplots_adjust(wspace=0.15, hspace=0.12)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


@torch.no_grad()
def save_model_sample_figure(
    model,
    sample: dict,
    device: torch.device,
    out_path: Path | str,
    *,
    title: str = "",
    show_error: bool = True,
) -> Path:
    a = sample.get("A")
    if a is not None:
        a = a.to(device)
    feas = sample["feas"].to(device)
    target = sample["target"]
    mask = sample["node_mask"]
    gh = sample.get("grid_h")
    gw = sample.get("grid_w")
    pred = model.predict_nodes(a, feas, gh, gw).cpu()
    n_out = pred.shape[1]
    n_leads = max(1, n_out // 3)

    meta = sample.get("meta") or {}
    if not title:
        title = f"{sample.get('stem', '')} | target {meta.get('target_time', '')}"

    paths = []
    for lead in range(n_leads):
        stem = Path(out_path).stem
        suffix = f"_lead{lead}" if n_leads > 1 else ""
        p = Path(out_path).with_name(f"{stem}{suffix}{Path(out_path).suffix}")
        paths.append(
            save_pred_gt_figure(
                pred, target, mask, p, title=title, show_error=show_error, lead_idx=lead,
                grid_h=gh, grid_w=gw,
            )
        )
    return paths[-1]


def _time_labels_from_meta(meta: dict, n_cond: int, n_fcst: int) -> list[str]:
    """Row labels for conditioning + forecast times."""
    times = meta.get("times") or []
    labels = []
    if len(times) >= n_cond + n_fcst:
        for i in range(n_cond):
            labels.append(str(times[i]))
        for i in range(n_cond, n_cond + n_fcst):
            labels.append(str(times[i]))
        return labels
  # fallback relative labels
    out = []
    if n_cond >= 3:
        out.extend(["t−2", "t−1"])
    elif n_cond == 2:
        out.extend(["t−1", "t−2"][:n_cond])
    else:
        out.extend([f"cond {k}" for k in range(n_cond)])
    for k in range(n_fcst):
        out.append("t" if k == 0 else f"t+{k}")
    return out


@torch.no_grad()
def save_timeline_figure(
    model,
    sample: dict,
    device: torch.device,
    out_path: Path | str,
    *,
    title: str = "",
    show_error: bool = True,
    dpi: int = 120,
) -> Path:
    """
    Timeline figure: t−2, t−1 (inputs), then t, t+1, t+2 (GT vs prediction).

    Layout: one row per time, six columns — GT/Pred for u, v, ssh.
    Input rows show observed conditioning only (Pred panel = same field, labeled input).
    """
    a = sample.get("A")
    if a is not None:
        a = a.to(device)
    feas = sample["feas"].to(device)
    target = sample["target"]
    mask = sample["node_mask"]
    gh = int(sample.get("grid_h") or int(mask.numel() ** 0.5))
    gw = int(sample.get("grid_w") or gh)

    pred = model.predict_nodes(a, feas, gh, gw).cpu().numpy()
    feas = feas.cpu().numpy()
    target = target.cpu().numpy()
    mask_g = mask.numpy().reshape(gh, gw)

    n_cond = feas.shape[1] // 3
    n_fcst = target.shape[1] // 3
    cond = feas.T.reshape(n_cond, 3, gh, gw)
    tgt = target.T.reshape(n_fcst, 3, gh, gw)
    prd = pred.T.reshape(n_fcst, 3, gh, gw)

    meta = sample.get("meta") or {}
    time_strs = _time_labels_from_meta(meta, n_cond, n_fcst)

    # Rows: last two conditioning frames + all forecast leads (t−2 … t+2 when 3+3)
    rows: list[tuple[str, str, np.ndarray, np.ndarray | None]] = []
    cond_start = max(0, n_cond - 2)
    for i in range(cond_start, n_cond):
        rel = i - n_cond
        lbl = f"t{rel}" if rel < 0 else "t"
        if i < len(time_strs):
            lbl = time_strs[i]
        rows.append((lbl, "input", cond[i], None))
    for k in range(n_fcst):
        lbl = time_strs[n_cond + k] if n_cond + k < len(time_strs) else ("t" if k == 0 else f"t+{k}")
        rows.append((lbl, "forecast", tgt[k], prd[k]))

    n_rows = len(rows)
    ncols = 9
    fig, axes = plt.subplots(n_rows, ncols, figsize=(14, 2.0 * n_rows + 1), squeeze=False)
    col_titles = [
        "GT u", "Pred u", "|err| u",
        "GT v", "Pred v", "|err| v",
        "GT ssh", "Pred ssh", "|err| ssh",
    ]

    all_gt = np.concatenate([r[2] for r in rows], axis=0)
    all_pr = np.concatenate([r[3] for r in rows if r[3] is not None], axis=0) if n_fcst else all_gt
    vmin = np.nanpercentile(np.concatenate([all_gt.ravel(), all_pr.ravel()]), 2)
    vmax = np.nanpercentile(np.concatenate([all_gt.ravel(), all_pr.ravel()]), 98)

    last_im = None
    for row_idx, (t_lbl, kind, gt_cube, pred_cube) in enumerate(rows):
        gt_m = masked_grid(gt_cube, mask_g)
        pr_m = masked_grid(pred_cube, mask_g) if pred_cube is not None else gt_m
        err_m = np.abs(pr_m - gt_m) if pred_cube is not None else None

        for ch in range(3):
            base = ch * 3
            panels = [gt_m[ch], pr_m[ch], err_m[ch] if err_m is not None else None]
            for j, data in enumerate(panels):
                ax = axes[row_idx, base + j]
                if data is None:
                    ax.axis("off")
                    continue
                cmap = "RdBu_r" if j < 2 else "viridis"
                v0, v1 = (vmin, vmax) if j < 2 else (0, np.nanpercentile(data, 98))
                last_im = ax.imshow(data, cmap=cmap, vmin=v0, vmax=v1)
                ax.set_xticks([])
                ax.set_yticks([])
                if row_idx == 0:
                    title = col_titles[base + j]
                    if kind == "input" and j == 1:
                        title = "obs (input)"
                    ax.set_title(title, fontsize=8)
            if kind == "input":
                for j in (1, 2):
                    axes[row_idx, base + j].axis("off")

        axes[row_idx, 0].set_ylabel(t_lbl, fontsize=8, rotation=0, labelpad=48, ha="right")

    if not title:
        title = f"timeline {sample.get('stem', '')}"
    fig.suptitle(title, y=1.01, fontsize=11)
    fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    fig.subplots_adjust(wspace=0.12, hspace=0.18)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_timeline_validation_figures(
    model,
    dataset,
    device: torch.device,
    out_dir: Path | str,
    *,
    n_samples: int = 4,
    seed: int = 42,
    show_error: bool = True,
) -> list[Path]:
    """Save timeline PNGs for random dataset indices (same selection as ``save_validation_figures``)."""
    import random

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(n_samples, len(dataset))
    if n == 0:
        return []

    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), n)

    was_training = model.training
    model.eval()
    paths = []
    for idx in indices:
        sample = dataset[idx]
        stem = sample["stem"]
        path = out_dir / f"timeline_{stem}.png"
        save_timeline_figure(
            model, sample, device, path,
            title=f"timeline | {stem}",
            show_error=show_error,
        )
        paths.append(path)
    if was_training:
        model.train()
    return paths


def save_validation_figures(
    model,
    dataset,
    device: torch.device,
    out_dir: Path | str,
    *,
    epoch: int | None = None,
    n_samples: int = 4,
    seed: int = 42,
    show_error: bool = True,
) -> list[Path]:
    """Pick ``n_samples`` from ``dataset`` and save GT vs pred PNGs."""
    import random

    out_dir = Path(out_dir)
    if epoch is not None:
        out_dir = out_dir / f"epoch_{epoch:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    n = min(n_samples, len(dataset))
    if n == 0:
        return []

    rng = random.Random(seed if epoch is None else seed + int(epoch))
    indices = rng.sample(range(len(dataset)), n)

    was_training = model.training
    model.eval()
    paths = []
    for idx in indices:
        sample = dataset[idx]
        stem = sample["stem"]
        path = out_dir / f"{stem}.png"
        save_model_sample_figure(
            model, sample, device, path,
            title=f"epoch {epoch} | {stem}" if epoch is not None else stem,
            show_error=show_error,
        )
        paths.append(path)
    if was_training:
        model.train()
    return paths
