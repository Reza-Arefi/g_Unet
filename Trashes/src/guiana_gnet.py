"""Graph and grid CNN models for per-node ocean regression (u, v, ssh)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.grid_ops import GridGCN
from src.utils.ops import GCN, GraphUnet, Initializer, norm_g

# Above this node count we use grid convs (no dense N×N, no GraphUnet pool).
HIGH_RES_NODES = 64 * 64


def nodes_to_grid(h: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    """(N, F) row-major nodes -> (1, F, H, W)."""
    return h.view(grid_h, grid_w, -1).permute(2, 0, 1).unsqueeze(0)


def grid_to_nodes(x: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    """(1, F, H, W) -> (N, F) row-major nodes."""
    return x.squeeze(0).permute(1, 2, 0).reshape(grid_h * grid_w, -1)


class _CNNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, drop: float):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ELU(inplace=True),
        ]
        if drop > 0.0:
            layers.append(nn.Dropout2d(p=drop))
        layers.extend([nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ELU(inplace=True)])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, drop: float):
        super().__init__()
        self.conv = _CNNBlock(in_ch + skip_ch, out_ch, drop)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class GuianaGridCNN(nn.Module):
    """
    128×128 grid regressor via U-Net CNN (no graph adjacency).

    Expects one node per pixel: ``feas`` (N, in_dim) with N = grid_h × grid_w.
    Adjacency ``g`` is ignored.
    """

    def __init__(
        self,
        in_dim: int = 9,
        out_dim: int = 3,
        l_dim: int = 48,
        h_dim: int = 128,
        drop_n: float = 0.3,
        drop_c: float = 0.2,
        **_,
    ):
        super().__init__()
        del h_dim  # kept for API parity with GuianaGNet
        c1, c2, c3, c4 = l_dim, l_dim * 2, l_dim * 4, l_dim * 8
        self.enc1 = _CNNBlock(in_dim, c1, drop_n)
        self.enc2 = _CNNBlock(c1, c2, drop_n)
        self.enc3 = _CNNBlock(c2, c3, drop_n)
        self.bottleneck = _CNNBlock(c3, c4, drop_n)
        self.dec3 = _UpBlock(c4, c3, c3, drop_n)
        self.dec2 = _UpBlock(c3, c2, c2, drop_n)
        self.dec1 = _UpBlock(c2, c1, c1, drop_n)
        self.out_drop = nn.Dropout2d(p=drop_c) if drop_c > 0.0 else nn.Identity()
        self.head = nn.Conv2d(c1, out_dim, kernel_size=1)
        self.pool = nn.MaxPool2d(2)
        Initializer.weights_init(self)

    def encode_grid(
        self,
        h: torch.Tensor,
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        x = nodes_to_grid(h, grid_h, grid_w)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        z = self.bottleneck(self.pool(e3))
        z = self.dec3(z, e3)
        z = self.dec2(z, e2)
        z = self.dec1(z, e1)
        return self.head(self.out_drop(z))

    def predict_nodes(
        self,
        g: torch.Tensor | None,
        h: torch.Tensor,
        grid_h: int | None = None,
        grid_w: int | None = None,
    ) -> torch.Tensor:
        del g
        n = h.shape[0]
        gh = grid_h or int(n**0.5)
        gw = grid_w or gh
        if gh * gw != n:
            raise ValueError(f"Node count {n} does not match grid {gh}x{gw}")
        return grid_to_nodes(self.encode_grid(h, gh, gw), gh, gw)

    def forward(
        self,
        g: torch.Tensor | None,
        h: torch.Tensor,
        target: torch.Tensor,
        node_mask: torch.Tensor,
        grid_h: int | None = None,
        grid_w: int | None = None,
    ) -> tuple[torch.Tensor, dict]:
        pred = self.predict_nodes(g, h, grid_h, grid_w)
        loss = masked_mse(pred, target, node_mask)
        metrics = regression_metrics(pred, target, node_mask)
        return loss, metrics


MODEL_CHOICES = ("gnet", "grid_cnn")


def build_guiana_model(name: str, **kwargs) -> nn.Module:
    """Factory for graph / grid CNN regressors sharing the same training API."""
    if name == "gnet":
        return GuianaGNet(**kwargs)
    if name == "grid_cnn":
        return GuianaGridCNN(**kwargs)
    raise ValueError(f"Unknown model {name!r}; choose from {MODEL_CHOICES}")


class GuianaGNet(nn.Module):
    """
    Node-level regressor: ``feas`` (N, 9) -> ``target`` (N, 3).

    - **≤ 64×64 nodes:** dense adjacency + Graph U-Net (original path)
    - **128×128 nodes:** grid GCN stack (one node per pixel, no dense 16k² matrix)
    """

    def __init__(
        self,
        in_dim: int = 9,
        out_dim: int = 3,
        l_dim: int = 48,
        h_dim: int = 128,
        ks: list[float] | None = None,
        act_n: str = "ELU",
        drop_n: float = 0.3,
        drop_c: float = 0.2,
        high_res_layers: int = 4,
    ):
        super().__init__()
        if ks is None:
            ks = [0.9, 0.8, 0.7]
        self.n_act = getattr(nn, act_n)()
        self.s_gcn = GCN(in_dim, l_dim, self.n_act, drop_n)
        self.g_unet = GraphUnet(
            ks, l_dim, l_dim, l_dim, self.n_act, drop_n,
        )
        self.grid_gcns = nn.ModuleList(
            [GridGCN(in_dim, l_dim, self.n_act, drop_n)]
            + [GridGCN(l_dim, l_dim, self.n_act, drop_n) for _ in range(high_res_layers - 1)]
        )
        self.out_l_1 = nn.Linear(l_dim, h_dim)
        self.out_l_2 = nn.Linear(h_dim, out_dim)
        self.out_drop = nn.Dropout(p=drop_c)
        Initializer.weights_init(self)

    def _is_high_res(self, n_nodes: int) -> bool:
        return n_nodes > HIGH_RES_NODES

    def encode(
        self,
        g: torch.Tensor | None,
        h: torch.Tensor,
        grid_h: int | None = None,
        grid_w: int | None = None,
    ) -> torch.Tensor:
        n = h.shape[0]
        gh = grid_h or int(n**0.5)
        gw = grid_w or gh

        if self._is_high_res(n):
            z = h
            for i, layer in enumerate(self.grid_gcns):
                z = layer(h if i == 0 else z, gh, gw)
            return z

        if g is None:
            raise ValueError("Dense adjacency required for low-resolution graphs")
        g = norm_g(g)
        h = self.s_gcn(g, h)
        hs = self.g_unet(g, h)
        return hs[-1]

    def predict_nodes(
        self,
        g: torch.Tensor | None,
        h: torch.Tensor,
        grid_h: int | None = None,
        grid_w: int | None = None,
    ) -> torch.Tensor:
        z = self.encode(g, h, grid_h, grid_w)
        z = self.out_drop(z)
        z = self.n_act(self.out_l_1(z))
        z = self.out_drop(z)
        return self.out_l_2(z)

    def forward(
        self,
        g: torch.Tensor | None,
        h: torch.Tensor,
        target: torch.Tensor,
        node_mask: torch.Tensor,
        grid_h: int | None = None,
        grid_w: int | None = None,
    ) -> tuple[torch.Tensor, dict]:
        pred = self.predict_nodes(g, h, grid_h, grid_w)
        loss = masked_mse(pred, target, node_mask)
        metrics = regression_metrics(pred, target, node_mask)
        return loss, metrics


def masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    node_mask: torch.Tensor,
) -> torch.Tensor:
    m = node_mask.unsqueeze(-1)
    diff = (pred - target) * m
    denom = m.sum().clamp(min=1.0) * pred.shape[-1]
    return (diff.pow(2).sum() / denom)


def regression_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    node_mask: torch.Tensor,
) -> dict:
    m = node_mask.unsqueeze(-1)
    diff = (pred - target).abs() * m
    denom = m.sum().clamp(min=1.0)
    mae = diff.sum() / (denom * pred.shape[-1])
    rmse = masked_mse(pred, target, node_mask).sqrt()
    return {"mae": mae.detach(), "rmse": rmse.detach()}
