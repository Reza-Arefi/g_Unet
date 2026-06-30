"""Zarr-backed 6h rollout samples for training / evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.dataloaders.guiana_margem_tensor import GuianaSequenceDataset
from src.data.graph_builder import grid_adjacency_sparse
from src.guiana_rollout_6h import patch_to_graph_tensors, split_cond_patch, split_forecast_patch

HIGH_RES_NODES = 64 * 64


class GuianaRollout6hDataset(Dataset):
    """
    One sample = real conditioning [t−24, t−18, t−12, t−6, t] + GT futures at +6h, +12h, …

    Uses ``GuianaSequenceDataset`` with ``forecast_steps = rollout_steps`` (stacked targets).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        crop_size: int = 128,
        dt_hours: int = 6,
        conditioning_steps: int = 5,
        rollout_steps: int = 12,
        start_time=None,
        end_time=None,
        time_range=None,
        graph_grid: int = 128,
    ):
        self.rollout_steps = int(rollout_steps)
        self.conditioning_steps = int(conditioning_steps)
        self.graph_grid = int(graph_grid)
        self._seq = GuianaSequenceDataset(
            path=path,
            crop_size=crop_size,
            dt=dt_hours,
            conditioning_steps=conditioning_steps,
            forecast_steps=self.rollout_steps,
            start_time=start_time,
            end_time=end_time,
            time_range=time_range,
        )
        self._template = self._build_graph_template()

    def _build_graph_template(self) -> dict:
        gh = gw = self.graph_grid
        n = gh * gw
        if n > HIGH_RES_NODES:
            a = grid_adjacency_sparse(gh, gw)
            return {"grid_h": gh, "grid_w": gw, "n_nodes": n, "A_sparse": a.float()}
        from src.data.graph_builder import grid_adjacency

        return {
            "grid_h": gh,
            "grid_w": gw,
            "n_nodes": n,
            "A": grid_adjacency(gh, gw),
        }

    @property
    def window(self) -> dict:
        return self._seq.window()

    def __len__(self) -> int:
        return len(self._seq)

    def sample_times(self, idx: int) -> list:
        return self._seq.sample_times(idx)

    def __getitem__(self, idx: int) -> dict:
        x, meta, mask = self._seq[int(idx)]
        y = meta["y"]
        if isinstance(y, torch.Tensor):
            y_np = y.cpu().numpy()
        else:
            y_np = np.asarray(y)
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy()
        else:
            x_np = np.asarray(x)
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = np.asarray(mask)

        cond_frames = split_cond_patch(y_np, self.conditioning_steps)
        target_frames = split_forecast_patch(x_np, self.rollout_steps)
        times = self.sample_times(idx)

        g = patch_to_graph_tensors(
            y_np,
            target_frames[0],
            mask_np,
            grid_size=self.graph_grid,
        )
        node_mask = g["node_mask"]

        return {
            "idx": int(idx),
            "stem": f"{idx:06d}",
            "cond_frames": cond_frames,
            "target_frames": target_frames,
            "node_mask": node_mask,
            "grid_h": g["grid_h"],
            "grid_w": g["grid_w"],
            "graph": {**self._template, "node_mask": node_mask},
            "times": [str(t) for t in times],
            "t_anchor": str(times[self.conditioning_steps - 1]),
        }


def collate_rollout_6h(batch: list[dict]) -> dict:
    return {
        "idx": [b["idx"] for b in batch],
        "stem": [b["stem"] for b in batch],
        "cond_frames": [b["cond_frames"] for b in batch],
        "target_frames": [b["target_frames"] for b in batch],
        "node_mask": torch.stack([b["node_mask"] for b in batch]),
        "grid_h": batch[0]["grid_h"],
        "grid_w": batch[0]["grid_w"],
        "graph": batch[0]["graph"],
        "times": [b["times"] for b in batch],
        "t_anchor": [b["t_anchor"] for b in batch],
    }
