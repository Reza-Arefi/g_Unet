"""PyTorch dataset over exported Guiana graph ``.pt`` files."""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.data.graph_builder import load_graph_sample

HIGH_RES_NODES = 64 * 64


def list_graph_stems(graphs_dir: Path) -> list[str]:
    graphs_dir = Path(graphs_dir)
    return sorted(p.stem for p in graphs_dir.glob("*.pt"))


def filter_stems_compatible(
    graphs_dir: Path,
    stems: list[str],
    *,
    in_ch: int,
    out_ch: int,
    n_nodes: int | None = None,
) -> tuple[list[str], list[str]]:
    """
    Keep only graphs whose ``feas`` / ``target`` channel counts (and optional ``n_nodes``) match.

    Use when a folder mixes old (e.g. ``target`` dim 3) and new (dim 9) exports.
    """
    graphs_dir = Path(graphs_dir)
    kept, skipped = [], []
    for stem in stems:
        graph, _ = load_graph_sample(graphs_dir / f"{stem}.pt")
        ok = (
            graph["feas"].shape[1] == in_ch
            and graph["target"].shape[1] == out_ch
        )
        if n_nodes is not None:
            ok = ok and int(graph["n_nodes"]) == n_nodes
        if ok:
            kept.append(stem)
        else:
            skipped.append(stem)
    return kept, skipped


def train_val_split(
    stems: list[str],
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    stems = list(stems)
    rng = random.Random(seed)
    rng.shuffle(stems)
    n_val = max(1, int(round(len(stems) * val_frac)))
    return stems[n_val:], stems[:n_val]


def _adjacency_from_graph(graph: dict) -> torch.Tensor | None:
    if "A" in graph:
        return graph["A"].float()
    if "A_sparse" in graph:
        return graph["A_sparse"].float()
    return None


class GuianaGraphDataset(Dataset):
    """One sample = one saved graph (``feas``, ``target``, ``node_mask``, grid meta)."""

    def __init__(self, graphs_dir: Path, stems: list[str] | None = None):
        self.graphs_dir = Path(graphs_dir)
        if stems is None:
            stems = list_graph_stems(self.graphs_dir)
        if not stems:
            raise FileNotFoundError(f"No .pt graphs in {self.graphs_dir}")
        self.stems = stems

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> dict:
        stem = self.stems[idx]
        graph, meta = load_graph_sample(self.graphs_dir / f"{stem}.pt")
        gh = int(graph["grid_h"])
        gw = int(graph["grid_w"])
        n_nodes = int(graph["n_nodes"])
        return {
            "stem": stem,
            "A": _adjacency_from_graph(graph),
            "feas": graph["feas"].float(),
            "target": graph["target"].float(),
            "node_mask": graph["node_mask"].float(),
            "grid_h": gh,
            "grid_w": gw,
            "n_nodes": n_nodes,
            "high_res": n_nodes > HIGH_RES_NODES,
            "meta": meta,
        }


def collate_graph_batch(batch: list[dict]) -> dict:
    high_res = batch[0]["high_res"]
    out = {
        "stem": [b["stem"] for b in batch],
        "feas": torch.stack([b["feas"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
        "node_mask": torch.stack([b["node_mask"] for b in batch]),
        "grid_h": batch[0]["grid_h"],
        "grid_w": batch[0]["grid_w"],
        "high_res": high_res,
        "meta": [b["meta"] for b in batch],
    }
    if not high_res:
        out["A"] = torch.stack([b["A"] for b in batch])
    else:
        out["A"] = None
    return out


def load_manifest(graphs_root: Path) -> dict:
    path = Path(graphs_root) / "manifest.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text())
