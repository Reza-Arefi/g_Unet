"""Build grid graphs from Guiana ocean patches for Graph U-Net training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def bottom_right_window(h_full: int, w_full: int, crop_size: int) -> dict:
    if crop_size > h_full or crop_size > w_full:
        raise ValueError(f"crop_size {crop_size} does not fit {h_full}x{w_full}")
    return {
        "crop_h": int(crop_size),
        "crop_w": int(crop_size),
        "top": int(h_full - crop_size),
        "left": int(w_full - crop_size),
    }


def block_pool_channels(feat: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    """Mean-pool ``(C, H, W)`` to ``(C, grid_h, grid_w)`` (H, W divisible by grid)."""
    feat = np.asarray(feat, dtype=np.float32)
    c, h, w = feat.shape
    if h % grid_h or w % grid_w:
        raise ValueError(f"Cannot pool {h}x{w} into {grid_h}x{grid_w} integer blocks")
    bh, bw = h // grid_h, w // grid_w
    out = feat.reshape(c, grid_h, bh, grid_w, bw).mean(axis=(2, 4))
    return out


def grid_adjacency_sparse(grid_h: int, grid_w: int) -> torch.Tensor:
    """Sparse 4-connected grid adjacency (row-normalized). Use for large grids (e.g. 128²)."""
    n = grid_h * grid_w
    rows, cols = [], []
    for r in range(grid_h):
        for c in range(grid_w):
            i = r * grid_w + c
            for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < grid_h and 0 <= cc < grid_w:
                    rows.append(i)
                    cols.append(rr * grid_w + cc)
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.ones(len(rows), dtype=torch.float32)
    a = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()
    deg = torch.sparse.sum(a, dim=1).to_dense().clamp(min=1.0)
    inv_deg = deg.pow(-1)
    r, c = a.indices()
    scaled = a.values() * inv_deg[r]
    return torch.sparse_coo_tensor(a.indices(), scaled, a.shape).coalesce()


def grid_adjacency(grid_h: int, grid_w: int) -> torch.Tensor:
    """Dense 4-connected grid adjacency with self-loops (fine for grid ≤ 64)."""
    n = grid_h * grid_w
    a = torch.zeros(n, n, dtype=torch.float32)
    for r in range(grid_h):
        for c in range(grid_w):
            i = r * grid_w + c
            a[i, i] = 1.0
            if r > 0:
                a[i, i - grid_w] = 1.0
            if r + 1 < grid_h:
                a[i, i + grid_w] = 1.0
            if c > 0:
                a[i, i - 1] = 1.0
            if c + 1 < grid_w:
                a[i, i + 1] = 1.0
    deg = a.sum(dim=1, keepdim=True).clamp(min=1.0)
    return a / deg


def sequence_to_graph(
    y_cond: np.ndarray | torch.Tensor,
    x_tgt: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor | None = None,
    grid_size: int = 32,
) -> dict:
    """
  Convert one sequence sample to a coarse grid graph.

  Args:
      y_cond: (conditioning_steps*3, H, W) — past frames stacked (e.g. 9 for 3 steps)
      x_tgt: (forecast_steps*3, H, W) — future frames (e.g. 3 for 1 step, 9 for 3 steps)
      mask: optional (1, H, W), 1 = invalid
      grid_size: nodes per side after block pooling (32 → 32² nodes; 128 → one node/pixel)

  Returns:
      dict with keys ``A`` or ``A_sparse``, ``feas``, ``target``, ``node_mask``, ``grid_h``, ``grid_w``
  """
    y = np.asarray(y_cond, dtype=np.float32)
    x = np.asarray(x_tgt, dtype=np.float32)
    h, w = y.shape[-2], y.shape[-1]
    if h % grid_size or w % grid_size:
        raise ValueError(f"Patch {h}x{w} must be divisible by grid_size={grid_size}")

    y_n = block_pool_channels(y, grid_size, grid_size)
    x_n = block_pool_channels(x, grid_size, grid_size)
    n = grid_size * grid_size
    feas = torch.from_numpy(y_n.reshape(y_n.shape[0], n).T)
    target = torch.from_numpy(x_n.reshape(x_n.shape[0], n).T)
    use_sparse = grid_size >= 64
    if use_sparse:
        a = grid_adjacency_sparse(grid_size, grid_size)
    else:
        a = grid_adjacency(grid_size, grid_size)

    if mask is not None:
        m = np.asarray(mask, dtype=np.float32)
        if m.ndim == 3:
            m = m[0]
        m_n = block_pool_channels(m[None], grid_size, grid_size)[0]
        node_mask = torch.from_numpy((m_n.reshape(n) < 0.5).astype(np.float32))
    else:
        node_mask = torch.ones(n, dtype=torch.float32)

    out = {
        "feas": feas,
        "target": target,
        "node_mask": node_mask,
        "grid_h": grid_size,
        "grid_w": grid_size,
        "n_nodes": n,
        "adj_format": "sparse" if use_sparse else "dense",
    }
    if use_sparse:
        out["A_sparse"] = a
    else:
        out["A"] = a
    return out


def save_graph_sample(path: Path, graph: dict, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"graph": graph, "meta": meta}, path)


def load_graph_sample(path: Path) -> tuple[dict, dict]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj["graph"], obj["meta"]


def write_manifest(out_dir: Path, payload: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
