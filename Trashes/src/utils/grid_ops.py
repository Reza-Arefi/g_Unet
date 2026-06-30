"""Grid (H×W) message passing without dense N×N adjacency — for 128×128 graphs."""

from __future__ import annotations

import torch
import torch.nn as nn


def grid_neighbor_aggregate(h: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    """
    Degree-normalized 4-neighbor + self aggregation on a regular grid.

    Args:
        h: (N, F) row-major nodes
    Returns:
        (N, F)
    """
    x = h.view(grid_h, grid_w, -1).permute(2, 0, 1).unsqueeze(0)  # 1, F, H, W
    acc = x.clone()
    if grid_w > 1:
        acc = acc + torch.nn.functional.pad(x[..., :-1], (1, 0, 0, 0))
        acc = acc + torch.nn.functional.pad(x[..., 1:], (0, 1, 0, 0))
    if grid_h > 1:
        acc = acc + torch.nn.functional.pad(x[:, :, :-1, :], (0, 0, 1, 0))
        acc = acc + torch.nn.functional.pad(x[:, :, 1:, :], (0, 0, 0, 1))
    deg = torch.ones(1, 1, grid_h, grid_w, device=h.device, dtype=h.dtype)
    if grid_w > 1:
        deg = deg + torch.nn.functional.pad(
            torch.ones(1, 1, grid_h, grid_w - 1, device=h.device, dtype=h.dtype), (1, 0, 0, 0)
        )
        deg = deg + torch.nn.functional.pad(
            torch.ones(1, 1, grid_h, grid_w - 1, device=h.device, dtype=h.dtype), (0, 1, 0, 0)
        )
    if grid_h > 1:
        deg = deg + torch.nn.functional.pad(
            torch.ones(1, 1, grid_h - 1, grid_w, device=h.device, dtype=h.dtype), (0, 0, 1, 0)
        )
        deg = deg + torch.nn.functional.pad(
            torch.ones(1, 1, grid_h - 1, grid_w, device=h.device, dtype=h.dtype), (0, 0, 0, 1)
        )
    out = (acc / deg.clamp(min=1.0)).squeeze(0).permute(1, 2, 0).reshape(-1, h.shape[-1])
    return out


class GridGCN(nn.Module):
    """GCN layer on a regular H×W grid (no dense adjacency matrix)."""

    def __init__(self, in_dim: int, out_dim: int, act: nn.Module, drop_p: float):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.act = act
        self.drop = nn.Dropout(p=drop_p) if drop_p > 0.0 else nn.Identity()

    def forward(self, h: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        h = self.drop(h)
        h = grid_neighbor_aggregate(h, grid_h, grid_w)
        h = self.proj(h)
        return self.act(h)
