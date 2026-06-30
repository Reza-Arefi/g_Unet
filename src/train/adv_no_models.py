"""adv-NO discriminator and perceptual loss backends (VGG / LPIPS)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torchvision.models import VGG19_Weights, vgg19


def field_to_rgb(x: torch.Tensor) -> torch.Tensor:
    """Map [B, C, H, W] fields to 3 channels for ImageNet-style perceptual nets."""
    if x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    if x.shape[1] == 2:
        return torch.cat([x, x[:, :1]], dim=1)
    return x[:, :3]


class VGGFeatureExtractor2D(nn.Module):
    """First L VGG-19 feature maps for 2D perceptual loss (Oommen et al. 2026)."""

    def __init__(
        self,
        layers: tuple[int, ...] = (0, 5, 10, 19, 28),
        pretrained: bool = True,
    ):
        super().__init__()
        weights = VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = vgg19(weights=weights)
        self.layers = layers
        self.net = nn.Sequential(*[vgg.features[i] for i in range(max(layers) + 1)])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = field_to_rgb(x)

        features: list[torch.Tensor] = []
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i in self.layers:
                features.append(x)
        return features


class VGGPerceptualLoss2D(nn.Module):
    """Weighted multi-layer VGG-19 L1 perceptual distance."""

    def __init__(
        self,
        perceptual_weights: tuple[float, ...] = (0.1, 0.1, 1.0, 1.0, 1.0),
        layers: tuple[int, ...] = (0, 5, 10, 19, 28),
        pretrained: bool = True,
    ):
        super().__init__()
        self.extractor = VGGFeatureExtractor2D(layers=layers, pretrained=pretrained)
        self.perceptual_weights = perceptual_weights
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_feats = self.extractor(pred)
        with torch.no_grad():
            target_feats = self.extractor(target)
        return sum(
            self.l1(pf, tf) * w
            for pf, tf, w in zip(pred_feats, target_feats, self.perceptual_weights)
        )


class LPIPSPerceptualLoss2D(nn.Module):
    """Learned perceptual patch similarity (LPIPS) for 2D fields."""

    def __init__(self, net: str = "alex"):
        super().__init__()
        import lpips

        self.lpips = lpips.LPIPS(net=net, verbose=False)
        for param in self.lpips.parameters():
            param.requires_grad = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_rgb = field_to_rgb(pred)
        target_rgb = field_to_rgb(target)
        return self.lpips(pred_rgb, target_rgb).mean()


def build_perceptual_loss(
    backend: str,
    *,
    perceptual_weights: tuple[float, ...] = (0.1, 0.1, 1.0, 1.0, 1.0),
    lpips_net: str = "alex",
) -> nn.Module:
    if backend == "vgg":
        return VGGPerceptualLoss2D(perceptual_weights=perceptual_weights)
    if backend == "lpips":
        return LPIPSPerceptualLoss2D(net=lpips_net)
    raise ValueError(f"Unknown perceptual_backend {backend!r}; use 'vgg' or 'lpips'")


def perceptual_backend_label(backend: str, lpips_net: str = "alex") -> str:
    if backend == "lpips":
        return f"LPIPS ({lpips_net})"
    return "VGG-19"


class UNetDiscriminatorSN2D(nn.Module):
    """2D U-Net discriminator with spectral normalization (Real-ESRGAN style)."""

    def __init__(self, in_channels: int = 3, base_feat: int = 64, skip_connection: bool = True):
        super().__init__()
        norm = spectral_norm
        self.skip_connection = skip_connection

        self.conv0 = nn.Conv2d(in_channels, base_feat, kernel_size=3, stride=1, padding=1)
        self.conv1 = norm(nn.Conv2d(base_feat, base_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(base_feat * 2, base_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(base_feat * 4, base_feat * 8, 4, 2, 1, bias=False))

        self.conv4 = norm(nn.Conv2d(base_feat * 8, base_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(base_feat * 4, base_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(base_feat * 2, base_feat, 3, 1, 1, bias=False))
        self.conv7 = norm(nn.Conv2d(base_feat, base_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(base_feat, base_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(base_feat, 1, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        x3 = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            x4 = x4 + x2

        x4 = F.interpolate(x4, scale_factor=2, mode="bilinear", align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            x5 = x5 + x1

        x5 = F.interpolate(x5, scale_factor=2, mode="bilinear", align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            x6 = x6 + x0

        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        return self.conv9(out)
