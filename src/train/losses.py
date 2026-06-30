"""Shared fix-window loss functions (L1 baseline and adv-NO composite)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.train.adv_no_models import VGGPerceptualLoss2D


def create_gaussian_center_weight(h: int, w: int, sigma: float) -> torch.Tensor:
    """Gaussian weight centered on the field; mean-normalized to ~1."""
    y = torch.arange(h, dtype=torch.float32)
    x = torch.arange(w, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    cy = (h - 1) / 2
    cx = (w - 1) / 2
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    weight = torch.exp(-dist2 / (2 * sigma**2))
    return weight / weight.mean()


def build_loss_spatial_weight(
    loss_spatial_mode: str,
    h: int,
    w: int,
    sigma: float,
) -> torch.Tensor | None:
    """Spatial weight G used in training (None = uniform / full field)."""
    if loss_spatial_mode == "gaussian_center":
        return create_gaussian_center_weight(h, w, sigma)
    if loss_spatial_mode == "full":
        return None
    raise ValueError(
        f"Unknown loss_spatial_mode {loss_spatial_mode!r}; use 'full' or 'gaussian_center'"
    )


def gaussian_weight_summary(weight: torch.Tensor) -> dict[str, float]:
    """Center vs edge stats for logging (matches preview interpretation)."""
    h, w = weight.shape[-2], weight.shape[-1]
    cy, cx = h // 2, w // 2
    center = float(weight[cy, cx])
    edge = float(weight[cy, 0])
    return {
        "center": center,
        "edge": edge,
        "center_over_edge": center / max(edge, 1e-8),
    }


def gaussian_weight_at_radius(weight: torch.Tensor, radius: float) -> float:
    """Training-normalized G on the horizontal midline at ``radius`` px from center."""
    h, w = weight.shape[-2], weight.shape[-1]
    cy, cx = h // 2, w // 2
    col = int(round(cx + radius))
    col = max(0, min(w - 1, col))
    return float(weight[cy, col])


def gaussian_weight_ring_summary(
    weight: torch.Tensor,
    ring_radii: tuple[float, ...] = (16.0, 32.0, 64.0),
) -> dict[str, float]:
    """Mean training weight G inside each radial band (center prioritized if inner > outer)."""
    h, w = weight.shape[-2], weight.shape[-1]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy = torch.arange(h, dtype=torch.float32)
    xx = torch.arange(w, dtype=torch.float32)
    yy, xx = torch.meshgrid(yy, xx, indexing="ij")
    r = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    w_np = weight.detach().cpu().numpy()
    r_np = r.numpy()

    radii = tuple(ring_radii)
    bounds = [0.0, *radii, float("inf")]
    labels = (
        [f"r≤{radii[0]:g}"]
        + [f"{radii[i]:g}<r≤{radii[i + 1]:g}" for i in range(len(radii) - 1)]
        + [f"r>{radii[-1]:g}"]
    )
    out: dict[str, float] = {}
    for label, lo, hi in zip(labels, bounds[:-1], bounds[1:]):
        band = (r_np >= lo) & (r_np < hi)
        if np.any(band):
            out[label] = float(np.mean(w_np[band]))
    return out


def expand_invalid_mask_for_loss(mask: torch.Tensor, buffer_px: int) -> torch.Tensor:
    """Dilate land/invalid pixels (mask == 1) for loss weighting near coastlines."""
    if buffer_px <= 0:
        return mask
    squeeze_batch = mask.dim() == 2
    m = (mask > 0).to(dtype=torch.float32)
    if m.dim() == 2:
        m = m.unsqueeze(0)
    k = 2 * int(buffer_px) + 1
    expanded = F.max_pool2d(m.unsqueeze(1), kernel_size=k, stride=1, padding=int(buffer_px))
    out = expanded.squeeze(1).to(dtype=mask.dtype)
    if squeeze_batch:
        out = out.squeeze(0)
    return out


def _spatial_weights(
    mask: torch.Tensor,
    spatial_weight: torch.Tensor | None,
    field: torch.Tensor,
) -> torch.Tensor:
    """Per-pixel weights [B, 1, H, W]: valid land mask × optional Gaussian center."""
    valid = (mask == 0).unsqueeze(1).to(dtype=field.dtype)
    if spatial_weight is None:
        return valid
    w = spatial_weight.to(device=field.device, dtype=field.dtype)
    if w.dim() == 2:
        w = w.unsqueeze(0).unsqueeze(0)
    return valid * w


def masked_loss(pred, target, mask, spatial_weight: torch.Tensor | None = None):
    """Masked L1 on valid pixels (mask == 0), optionally Gaussian-weighted at center."""
    valid = (mask == 0).unsqueeze(1)
    if spatial_weight is None:
        valid = valid.expand_as(pred)
        return F.l1_loss(pred[valid], target[valid])
    weights = _spatial_weights(mask, spatial_weight, pred)
    diff = torch.abs(pred - target)
    denom = weights.sum().clamp_min(1e-8)
    return (diff * weights).sum() / denom


def masked_rmse(pred, target, mask, spatial_weight: torch.Tensor | None = None):
    """Masked RMSE on valid pixels (mask == 0), optionally Gaussian-weighted at center."""
    valid = (mask == 0).unsqueeze(1)
    if spatial_weight is None:
        valid = valid.expand_as(pred)
        mse = F.mse_loss(pred[valid], target[valid])
    else:
        weights = _spatial_weights(mask, spatial_weight, pred)
        sq = (pred - target) ** 2
        denom = weights.sum().clamp_min(1e-8)
        mse = (sq * weights).sum() / denom
    return torch.sqrt(mse.clamp_min(1e-8))


def speed_magnitude_uv(field: torch.Tensor) -> torch.Tensor:
    """Speed |U,V| = sqrt(u^2 + v^2) from [B, C, H, W] with u,v at channels 0,1."""
    u, v = field[:, 0], field[:, 1]
    return torch.sqrt(u * u + v * v + 1e-8)


def masked_speed_loss(
    pred, target, mask, spatial_weight: torch.Tensor | None = None
) -> torch.Tensor:
    """Masked L1 on speed magnitude (channels 0,1 = u,v)."""
    pred_spd = speed_magnitude_uv(pred)
    tgt_spd = speed_magnitude_uv(target)
    weights = _spatial_weights(mask, spatial_weight, pred)
    w2d = weights[:, 0]
    denom = w2d.sum().clamp_min(1e-8)
    return (torch.abs(pred_spd - tgt_spd) * w2d).sum() / denom


def masked_speed_rmse(
    pred, target, mask, spatial_weight: torch.Tensor | None = None
) -> torch.Tensor:
    """Masked RMSE on speed magnitude (channels 0,1 = u,v)."""
    pred_spd = speed_magnitude_uv(pred)
    tgt_spd = speed_magnitude_uv(target)
    weights = _spatial_weights(mask, spatial_weight, pred)
    w2d = weights[:, 0]
    denom = w2d.sum().clamp_min(1e-8)
    mse = ((pred_spd - tgt_spd) ** 2 * w2d).sum() / denom
    return torch.sqrt(mse.clamp_min(1e-8))


def reconstruction_loss(
    pred,
    target,
    mask,
    *,
    metric: str = "components",
    spatial_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if metric == "speed":
        return masked_speed_loss(pred, target, mask, spatial_weight)
    if metric == "components":
        return masked_loss(pred, target, mask, spatial_weight)
    raise ValueError(f"Unknown loss metric {metric!r}; use 'components' or 'speed'")


def apply_field_mask(
    field: torch.Tensor,
    mask: torch.Tensor,
    spatial_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Zero invalid pixels; optionally taper with Gaussian center weight for loss terms."""
    weights = _spatial_weights(mask, spatial_weight, field)
    return field * weights


def my_loss(
    pred,
    target,
    mask,
    spatial_weight=None,
    eps: float = 1e-3,
    *,
    metric: str = "components",
) -> torch.Tensor:
    del eps
    return reconstruction_loss(pred, target, mask, metric=metric, spatial_weight=spatial_weight)


def ragans_generator_loss(
    pred_real: torch.Tensor,
    pred_fake: torch.Tensor,
    criterion: nn.Module,
    valid: torch.Tensor,
    fake: torch.Tensor,
) -> torch.Tensor:
    """Relativistic average GAN generator term (Eq. 7 in Oommen et al. 2026)."""
    return (
        criterion(pred_fake - pred_real.mean(dim=0, keepdim=True), valid)
        + criterion(pred_real - pred_fake.mean(dim=0, keepdim=True), fake)
    ) / 2


def ragans_discriminator_loss(
    pred_real: torch.Tensor,
    pred_fake: torch.Tensor,
    criterion: nn.Module,
    valid: torch.Tensor,
    fake: torch.Tensor,
) -> torch.Tensor:
    """Relativistic average GAN discriminator term (Eq. 6)."""
    loss_real = criterion(pred_real - pred_fake.mean(dim=0, keepdim=True), valid)
    loss_fake = criterion(pred_fake - pred_real.mean(dim=0, keepdim=True), fake)
    return (loss_real + loss_fake) / 2


class AdvNOLoss(nn.Module):
    """L_advNO = L1 + lambda_perc * perceptual + beta * L_G (Eq. 8 + global perc scale)."""

    def __init__(
        self,
        perceptual_loss: nn.Module | None = None,
        lambda_perc: float = 0.1,
        beta: float = 0.1,
        loss_metric: str = "components",
        *,
        val_reconstruction_metric: str = "l1",
        feature_extractor: nn.Module | None = None,
        perceptual_weights: tuple[float, ...] = (0.1, 0.1, 1.0, 1.0, 1.0),
    ):
        super().__init__()
        if perceptual_loss is not None:
            self.perceptual_loss_fn = perceptual_loss
        elif feature_extractor is not None:
            # Legacy: raw VGG extractor + external weights
            self.perceptual_loss_fn = VGGPerceptualLoss2D(
                perceptual_weights=perceptual_weights,
            )
        else:
            self.perceptual_loss_fn = None
        self.lambda_perc = lambda_perc
        self.beta = beta
        self.loss_metric = loss_metric
        self.val_reconstruction_metric = val_reconstruction_metric
        self.gan = nn.BCEWithLogitsLoss()

    def perceptual_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.perceptual_loss_fn is None:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        return self.perceptual_loss_fn(pred, target)

    def generator_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        discriminator: nn.Module,
        spatial_weight: torch.Tensor | None = None,
        *,
        use_perceptual: bool = True,
        use_adversarial: bool = True,
    ) -> dict[str, torch.Tensor]:
        loss_l1 = reconstruction_loss(
            pred, target, mask, metric=self.loss_metric, spatial_weight=spatial_weight
        )

        zero = torch.zeros((), device=pred.device, dtype=pred.dtype)
        loss_perceptual = zero
        loss_adv = zero

        if use_perceptual or use_adversarial:
            pred_m = apply_field_mask(pred, mask, spatial_weight)
            target_m = apply_field_mask(target, mask, spatial_weight)

            if use_perceptual:
                loss_perceptual = self.lambda_perc * self.perceptual_loss(pred_m, target_m)

            if use_adversarial:
                valid = torch.ones_like(discriminator(target_m))
                fake = torch.zeros_like(valid)
                pred_real = discriminator(target_m).detach()
                pred_fake = discriminator(pred_m)
                loss_adv = ragans_generator_loss(pred_real, pred_fake, self.gan, valid, fake)

        total = loss_l1 + loss_perceptual
        if use_adversarial:
            total = total + self.beta * loss_adv
        return {
            "total": total,
            "l1": loss_l1,
            "perceptual": loss_perceptual,
            "adv": loss_adv,
        }

    def discriminator_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        discriminator: nn.Module,
        spatial_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pred_m = apply_field_mask(pred.detach(), mask, spatial_weight)
        target_m = apply_field_mask(target, mask, spatial_weight)
        valid = torch.ones_like(discriminator(target_m))
        fake = torch.zeros_like(valid)
        pred_real = discriminator(target_m)
        pred_fake = discriminator(pred_m)
        return ragans_discriminator_loss(pred_real, pred_fake, self.gan, valid, fake)

    @torch.no_grad()
    def validation_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        spatial_weight: torch.Tensor | None = None,
    ) -> dict[str, float]:
        pred_m = apply_field_mask(pred, mask, spatial_weight)
        target_m = apply_field_mask(target, mask, spatial_weight)
        val_metric = getattr(self, "val_reconstruction_metric", "l1")
        l1 = float(
            reconstruction_loss(
                pred, target, mask, metric=self.loss_metric, spatial_weight=spatial_weight
            ).item()
        )
        if self.loss_metric == "speed":
            rmse = float(masked_speed_rmse(pred, target, mask, spatial_weight).item())
        else:
            rmse = float(masked_rmse(pred, target, mask, spatial_weight).item())
        if val_metric == "rmse":
            reconstruction = rmse
        elif val_metric == "l1":
            reconstruction = l1
        else:
            raise ValueError(
                f"Unknown val_reconstruction_metric {val_metric!r}; use 'l1' or 'rmse'"
            )
        return {
            "l1": l1,
            "rmse": rmse,
            "reconstruction": reconstruction,
            "perceptual": float((self.lambda_perc * self.perceptual_loss(pred_m, target_m)).item()),
        }
