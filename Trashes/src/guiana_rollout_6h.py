"""
6-hour Graph U-Net rollout: Δt=6h, 5-frame conditioning [t−24…t], 1-step decode → t+6.

Training: autoregressive rollout over multiple horizons with weighted loss.
Evaluation: rollout limit h_max from correlation, RMSE vs climatology, energy, spectral slope.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.graph_builder import sequence_to_graph
from src.guiana_gnet import GuianaGNet, masked_mse

CHANNEL_NAMES = ("u", "v", "ssh")
HIGH_RES_NODES = 64 * 64


def default_rollout_lambdas(n_steps: int, growth: float = 1.15) -> list[float]:
    """λ_i increases with horizon step (1-indexed distance)."""
    w = [growth ** (i - 1) for i in range(1, n_steps + 1)]
    s = sum(w)
    return [x / s for x in w]


def patch_to_graph_tensors(
    y_cond: np.ndarray | torch.Tensor,
    x_tgt: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor | None,
    *,
    grid_size: int = 128,
) -> dict[str, torch.Tensor | int | None]:
    """Build graph dict from conditioning (C,H,W) and target (3,H,W)."""
    graph = sequence_to_graph(y_cond, x_tgt, mask, grid_size=grid_size)
    out: dict[str, Any] = {
        "feas": graph["feas"].float(),
        "target": graph["target"].float(),
        "node_mask": graph["node_mask"].float(),
        "grid_h": int(graph["grid_h"]),
        "grid_w": int(graph["grid_w"]),
        "n_nodes": int(graph["n_nodes"]),
    }
    if "A_sparse" in graph:
        out["A_sparse"] = graph["A_sparse"].float()
    if "A" in graph:
        out["A"] = graph["A"].float()
    return out


def graph_adjacency(graph: dict, device: torch.device) -> torch.Tensor | None:
    n = int(graph["n_nodes"])
    if n > HIGH_RES_NODES:
        return None
    g = graph.get("A_sparse") or graph.get("A")
    return g.to(device) if g is not None else None


def block_pool_channels_torch(feat: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    """Mean-pool (C, H, W) -> (C, grid_h, grid_w); differentiable."""
    c, h, w = feat.shape
    bh, bw = h // grid_h, w // grid_w
    return feat.reshape(c, grid_h, bh, grid_w, bw).mean(dim=(2, 4))


def frames_to_feas(frames: list[torch.Tensor], grid_h: int, grid_w: int) -> torch.Tensor:
    """Last 5 frames (3,H,W) each -> node features (N, 15)."""
    y = torch.cat(frames[-5:], dim=0)
    y_n = block_pool_channels_torch(y, grid_h, grid_w)
    n = grid_h * grid_w
    return y_n.reshape(y_n.shape[0], n).permute(1, 0).contiguous()


def nodes_to_patch_tensor(pred: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    """(N, 3) -> (3, H, W), keeps autograd graph."""
    return pred.T.reshape(3, grid_h, grid_w)


def nodes_to_patch(pred: torch.Tensor, grid_h: int, grid_w: int) -> np.ndarray:
    """(N, 3) -> (3, H, W) numpy (inference)."""
    return nodes_to_patch_tensor(pred, grid_h, grid_w).detach().cpu().numpy().astype(np.float32)


def split_cond_patch(y_cond: np.ndarray | torch.Tensor, n_cond: int = 5) -> list[np.ndarray]:
    """(n_cond*3, H, W) -> list of (3, H, W)."""
    y = np.asarray(y_cond, dtype=np.float32)
    frames = []
    for i in range(n_cond):
        frames.append(y[i * 3 : (i + 1) * 3])
    return frames


def split_forecast_patch(x_fcst: np.ndarray | torch.Tensor, n_steps: int) -> list[np.ndarray]:
    """(n_steps*3, H, W) -> list of (3, H, W)."""
    x = np.asarray(x_fcst, dtype=np.float32)
    return [x[i * 3 : (i + 1) * 3] for i in range(n_steps)]


def predict_graph_step_train(
    model: GuianaGNet,
    graph: dict,
    history_frames: list[torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable one-step predict; history may include prior predictions."""
    gh, gw = int(graph["grid_h"]), int(graph["grid_w"])
    feas = frames_to_feas(history_frames, gh, gw).to(device)
    g_adj = graph_adjacency(graph, device)
    pred = model.predict_nodes(g_adj, feas, gh, gw)
    return pred, nodes_to_patch_tensor(pred, gh, gw)


@torch.no_grad()
def predict_graph_step(
    model: GuianaGNet,
    graph: dict,
    cond_frames: list[np.ndarray],
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray]:
    """
    cond_frames: list of (3,H,W), length = conditioning_steps (5).
    Returns (pred_nodes N×3, pred_patch 3×H×W).
    """
    gh, gw = int(graph["grid_h"]), int(graph["grid_w"])
    frames = [
        torch.as_tensor(f, device=device, dtype=torch.float32) for f in cond_frames[-5:]
    ]
    pred, pred_patch = predict_graph_step_train(model, graph, frames, device)
    return pred, pred_patch.detach().cpu().numpy().astype(np.float32)


def rollout_loss_autoregressive(
    model: GuianaGNet,
    graph: dict,
    cond_frames: list[np.ndarray],
    target_frames: list[np.ndarray],
    node_mask: torch.Tensor,
    device: torch.device,
    lambdas: list[float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Autoregressive rollout: feed predictions back into the 5-frame window.

    target_frames: GT patches at t+6, t+12, … (len = n_rollout steps).
    """
    n_steps = len(target_frames)
    if lambdas is None:
        lambdas = default_rollout_lambdas(n_steps)
    if len(lambdas) != n_steps:
        raise ValueError(f"len(lambdas)={len(lambdas)} != n_steps={n_steps}")

    gh, gw = int(graph["grid_h"]), int(graph["grid_w"])
    # Real conditioning frames (constants); later entries are predicted patches with grad.
    history = [
        torch.as_tensor(f, device=device, dtype=torch.float32) for f in cond_frames
    ]
    total_loss: torch.Tensor | None = None
    metrics: dict[str, float] = {}

    for step in range(n_steps):
        pred, pred_patch = predict_graph_step_train(model, graph, history, device)
        tgt_graph = patch_to_graph_tensors(
            cond_frames[-1], target_frames[step], None, grid_size=gh
        )
        target = tgt_graph["target"].to(device)
        loss_step = masked_mse(pred, target, node_mask)
        term = float(lambdas[step]) * loss_step
        total_loss = term if total_loss is None else total_loss + term
        metrics[f"loss_step_{step + 1}"] = loss_step.item()
        metrics[f"rmse_step_{step + 1}"] = loss_step.sqrt().item()
        history.append(pred_patch)

    if total_loss is None:
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    metrics["loss_total"] = total_loss.item()
    return total_loss, metrics


@torch.no_grad()
def rollout_predict(
    model: GuianaGNet,
    graph: dict,
    cond_frames: list[np.ndarray],
    n_steps: int,
    device: torch.device,
) -> list[np.ndarray]:
    """Inference rollout; returns list of predicted (3,H,W) patches."""
    history = [f.copy() for f in cond_frames]
    preds = []
    for _ in range(n_steps):
        _, pred_patch = predict_graph_step(model, graph, history[-5:], device)
        preds.append(pred_patch)
        history.append(pred_patch)
    return preds


def _valid_node_values(patch: np.ndarray, valid_2d: np.ndarray) -> np.ndarray:
    out = []
    for c in range(3):
        out.append(patch[c][valid_2d].ravel())
    return np.concatenate(out)


def spatial_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom < 1e-12:
        return float("nan")
    return float((a * b).sum() / denom)


def patch_rmse(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    d = _valid_node_values(pred - gt, valid)
    return float(np.sqrt(np.mean(d ** 2))) if d.size else float("nan")


def kinetic_energy(patch: np.ndarray, valid: np.ndarray) -> float:
    u, v = patch[0][valid], patch[1][valid]
    return float(np.mean(u * u + v * v))


def spectral_slope_2d(field: np.ndarray, valid: np.ndarray) -> float:
    """Radial mean power spectrum slope (log-log)."""
    f = np.where(valid, field, np.nan)
    f = np.nan_to_num(f, nan=0.0)
    fft = np.fft.fft2(f)
    power = np.abs(fft) ** 2
    h, w = f.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(int)
    r_max = min(cy, cx)
    bins = np.bincount(r.ravel(), power.ravel(), minlength=r_max + 1)[: r_max + 1]
    counts = np.bincount(r.ravel(), minlength=r_max + 1)[: r_max + 1]
    counts = np.maximum(counts, 1)
    radial = bins / counts
    rs = np.arange(1, len(radial))
    if len(rs) < 4:
        return float("nan")
    p = radial[1:]
    p = np.maximum(p, 1e-12)
    slope, _ = np.polyfit(np.log(rs), np.log(p), 1)
    return float(slope)


def evaluate_rollout_step(
    pred: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray,
    *,
    clim_rmse: float,
    corr_thresh: float = 0.8,
    energy_err_thresh: float = 0.2,
    slope_ratio_tol: float = 0.35,
) -> dict[str, Any]:
    """Per-step metrics and failure flags for rollout limit detection."""
    rmse = patch_rmse(pred, gt, valid)
    corr = float(
        np.mean([spatial_corr(pred[c][valid], gt[c][valid]) for c in range(3)])
    )
    e_pred = kinetic_energy(pred, valid)
    e_gt = kinetic_energy(gt, valid)
    energy_err = abs(e_pred - e_gt) / max(e_gt, 1e-12)
    slope_p = np.mean([spectral_slope_2d(pred[c], valid) for c in range(3)])
    slope_g = np.mean([spectral_slope_2d(gt[c], valid) for c in range(3)])
    slope_ratio = abs(slope_p - slope_g) / max(abs(slope_g), 1e-6)

    fail_corr = corr < corr_thresh if np.isfinite(corr) else True
    fail_rmse = rmse > clim_rmse if np.isfinite(rmse) and np.isfinite(clim_rmse) else False
    fail_energy = energy_err > energy_err_thresh
    fail_slope = slope_ratio > slope_ratio_tol
    failed = fail_corr or fail_rmse or fail_energy or fail_slope

    return {
        "rmse": rmse,
        "corr": corr,
        "energy_err": energy_err,
        "slope_ratio": slope_ratio,
        "fail_corr": fail_corr,
        "fail_rmse": fail_rmse,
        "fail_energy": fail_energy,
        "fail_slope": fail_slope,
        "failed": failed,
    }


def find_h_max(step_metrics: list[dict[str, Any]]) -> int | None:
    """First 1-based horizon step where failure triggers, else None."""
    for i, m in enumerate(step_metrics, start=1):
        if m.get("failed"):
            return i
    return None


def save_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
