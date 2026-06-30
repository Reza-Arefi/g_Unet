"""
DW-Net (Depth-Wise U-Net) — PyTorch implementation
====================================================
Architecture (from Figure 2, panel c):
  - N waves, each with its own encoder-decoder (no weight sharing)
  - Intermediate waves do NOT recover full resolution before passing
    features to the next wave
  - Down-sampling  : Average Pooling
  - Up-sampling    : Transposed Convolution (learned)
  - Blue block (→) : DW 7×7 conv → PW 1×1 conv → residual add
  - Brown block (→): same depthwise-separable design, independent params
  - Skip connections via concatenation (all models except SineNet)

Input : (B, C_in,  H, W)
Output: (B, C_out, H, W)
"""

import torch
import torch.nn as nn
from typing import List


# ---------------------------------------------------------------------------
# Primitive building blocks
# ---------------------------------------------------------------------------

class DWSepBlock(nn.Module):
    """
    Depthwise-separable residual block used throughout DW-Net.
    DW 7×7 (groups=C) → BN → ReLU → PW 1×1 → BN → (+residual) → ReLU

    A learnable 1×1 projection is added on the skip when in_ch != out_ch
    so every block is independently parameterised.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=7, padding=3,
                            groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

        # projection shortcut (used when channel dims differ)
        self.proj = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                          nn.BatchNorm2d(out_ch))
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        out = self.act(self.bn1(self.dw(x)))
        out = self.bn2(self.pw(out))
        return self.act(out + residual)


class DownBlock(nn.Module):
    """Average-pool down-sampling followed by an independent DWSepBlock."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool  = nn.AvgPool2d(kernel_size=2, stride=2)
        self.block = DWSepBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(x))


class UpBlock(nn.Module):
    """
    Transposed-conv up-sampling then concat with skip, then DWSepBlock.
    After concat the channel count is (up_ch + skip_ch); the block maps
    that to out_ch independently for each level / wave.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up    = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.block = DWSepBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # guard against odd spatial sizes
        if x.shape != skip.shape:
            x = nn.functional.interpolate(x, size=skip.shape[2:],
                                           mode='bilinear', align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


# ---------------------------------------------------------------------------
# Single Wave
# ---------------------------------------------------------------------------

class DWWave(nn.Module):
    """
    One encoder-decoder wave.

    Parameters
    ----------
    in_ch       : number of input channels (from previous wave or stem)
    out_ch      : number of output channels produced at *full resolution*
                  (only the last wave reaches full resolution in DW-Net;
                   intermediate waves output at their *bottleneck* resolution)
    channels    : list of channel widths for each resolution level, e.g.
                  [64, 128, 256, 512].  len(channels) == number of levels.
    full_res_out: if True the wave includes an up-sampling path back to the
                  original resolution (used for the last wave).
                  Intermediate waves stop at the bottleneck.
    """

    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 channels: List[int],
                 full_res_out: bool = True):
        super().__init__()
        assert len(channels) >= 2, "Need at least 2 levels."

        self.full_res_out = full_res_out
        n_levels = len(channels)

        # ---- encoder -------------------------------------------------------
        # stem: single 3×3 conv (stride 1, padding 1) as shown in Figure 2
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        self.down_blocks = nn.ModuleList()
        for i in range(n_levels - 1):
            self.down_blocks.append(DownBlock(channels[i], channels[i + 1]))

        # ---- decoder (only built when we need full-res output) -------------
        if full_res_out:
            self.up_blocks = nn.ModuleList()
            # iterate from deepest up-block to shallowest
            # level indices going up: (n_levels-1) -> ... -> 1
            for i in range(n_levels - 2, -1, -1):
                skip_ch = channels[i]
                deep_ch = channels[i + 1]
                up_out  = channels[i] if i > 0 else out_ch
                self.up_blocks.append(UpBlock(deep_ch, skip_ch, up_out))

        # ---- bottleneck output projection (used by intermediate waves) -----
        # maps bottleneck features to out_ch so the next wave can use them
        if not full_res_out:
            self.bottleneck_proj = DWSepBlock(channels[-1], out_ch)

    # ------------------------------------------------------------------ fwd
    def forward(self, x: torch.Tensor):
        """
        Returns
        -------
        If full_res_out : full-resolution feature map  (B, out_ch, H, W)
        Else            : bottleneck feature map        (B, out_ch, H/2^k, W/2^k)
        """
        # encoder
        skips = []
        feat = self.stem(x)
        skips.append(feat)                          # level 0 skip

        for down in self.down_blocks:
            feat = down(feat)
            skips.append(feat)                      # levels 1..N-1

        bottleneck = skips.pop()                    # deepest feature

        if not self.full_res_out:
            return self.bottleneck_proj(bottleneck)

        # decoder
        feat = bottleneck
        for up in self.up_blocks:
            skip = skips.pop()
            feat = up(feat, skip)

        return feat                                  # (B, out_ch, H, W)


# ---------------------------------------------------------------------------
# DW-UNet  (N waves)
# ---------------------------------------------------------------------------

class DWUNet(nn.Module):
    """
    DW-Net with N waves.

    Key property (panel c caption):
        "No full-resolution recovery in intermediate waves."
    Waves 1 … N-1 terminate at the bottleneck and pass those features
    (upsampled back to full resolution with a single learned transposed conv)
    as the input to the next wave.  Only wave N performs the full decoder.

    Parameters
    ----------
    in_channels   : C_in  (input image / volume channels)
    out_channels  : C_out (final prediction channels)
    n_waves       : N     (number of waves, ≥ 1)
    channels      : channel progression per wave, e.g. [64, 128, 256, 512]
    inter_upsample: how to bring intermediate bottleneck back to full res
                    before feeding the next wave.  Uses ConvTranspose2d chain.
    """

    def __init__(self,
                 in_channels:  int = 1,
                 out_channels: int = 1,
                 n_waves:      int = 2,
                 channels:     List[int] = None):
        super().__init__()

        if channels is None:
            channels = [64, 128, 256, 512]

        self.n_waves   = n_waves
        self.channels  = channels
        n_levels       = len(channels)
        stride         = 2 ** (n_levels - 1)   # total spatial reduction

        self.waves          = nn.ModuleList()
        self.inter_upsample = nn.ModuleList()   # one per intermediate wave

        for w in range(n_waves):
            is_last      = (w == n_waves - 1)
            wave_in_ch   = in_channels if w == 0 else channels[0]
            wave_out_ch  = out_channels if is_last else channels[0]

            wave = DWWave(
                in_ch       = wave_in_ch,
                out_ch      = wave_out_ch,
                channels    = channels,
                full_res_out= is_last,
            )
            self.waves.append(wave)

            # For intermediate waves we need to bring the bottleneck feature
            # (at H/stride, W/stride) back to full resolution so the next
            # wave receives a (B, channels[0], H, W) tensor.
            if not is_last:
                # Stack of transposed convs, one per down-sampling level
                up_layers = []
                ch = channels[0]                # bottleneck_proj output ch
                for _ in range(n_levels - 1):
                    up_layers += [
                        nn.ConvTranspose2d(ch, ch, kernel_size=2, stride=2),
                        nn.BatchNorm2d(ch),
                        nn.ReLU(inplace=True),
                    ]
                self.inter_upsample.append(nn.Sequential(*up_layers))

    # ------------------------------------------------------------------ fwd
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C_in, H, W)
        returns (B, C_out, H, W)
        """
        feat = x
        for w, wave in enumerate(self.waves):
            feat = wave(feat)

            is_last = (w == self.n_waves - 1)
            if not is_last:
                # bring bottleneck back to full resolution
                feat = self.inter_upsample[w](feat)

                # guard spatial size matches input
                if feat.shape[2:] != x.shape[2:]:
                    feat = nn.functional.interpolate(
                        feat, size=x.shape[2:],
                        mode='bilinear', align_corners=False)

        return feat


# ---------------------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    B, C_in, C_out = 2, 1, 2
    H, W = 256, 256

    model = DWUNet(
        in_channels  = C_in,
        out_channels = C_out,
        n_waves      = 2,
        channels     = [64, 128, 256, 512],
    )

    x   = torch.randn(B, C_in, H, W)
    out = model(x)

    print(f"Input  shape : {x.shape}")
    print(f"Output shape : {out.shape}")   # expected (2, 2, 256, 256)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params : {total_params:,}")