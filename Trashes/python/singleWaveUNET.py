import torch
import torch.nn as nn
import torch.nn.functional as F


def get_num_groups(num_channels, max_groups=8):
    """
    Returns a valid number of groups for GroupNorm.
    """
    for g in reversed(range(1, max_groups + 1)):
        if num_channels % g == 0:
            return g
    return 1


class ConvBlock(nn.Module):
    """
    Two 3x3 convolutions with:
        - stride = 1
        - padding = 1
        - bias enabled
        - GroupNorm
        - ReLU
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        groups1 = get_num_groups(out_channels)
        groups2 = get_num_groups(out_channels)

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.GroupNorm(groups1, out_channels),
            nn.ReLU(inplace=True),

            # nn.Conv2d(
            #     out_channels,
            #     out_channels,
            #     kernel_size=3,
            #     stride=1,
            #     padding=1,
            #     bias=True,
            # ),
            # nn.GroupNorm(groups2, out_channels),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class singleWaveUNet(nn.Module):
    """
    Standard U-Net (single-wave U-Net from Fig. 2a)

    Features:
    ----------
    - Variable depth
    - 2x2 MaxPool downsampling
    - Transposed convolution upsampling
    - GroupNorm instead of BatchNorm
    - Conv bias enabled
    - Reduced bottleneck width to approximately match
      original U-Net parameter count

    Input:
        9 x 128 x 128

    Output:
        3 x 128 x 128
    """

    def __init__(
        self,
        in_channels=9,
        out_channels=3,
        depth=4,
        base_channels=64,
        bottleneck_reduction=2,
    ):
        super().__init__()

        self.depth = depth

        # -------------------------
        # Encoder
        # -------------------------
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        ch_in = in_channels
        ch_out = base_channels

        encoder_channels = []

        for _ in range(depth):
            self.encoders.append(ConvBlock(ch_in, ch_out))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))

            encoder_channels.append(ch_out)

            ch_in = ch_out
            ch_out *= 2

        # -------------------------
        # Bottleneck
        # -------------------------
        bottleneck_channels = ch_in * 2 // bottleneck_reduction

        self.bottleneck = ConvBlock(ch_in, bottleneck_channels)

        # -------------------------
        # Decoder
        # -------------------------
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        decoder_in = bottleneck_channels

        for enc_ch in reversed(encoder_channels):

            self.upconvs.append(
                nn.ConvTranspose2d(
                    decoder_in,
                    enc_ch,
                    kernel_size=2,
                    stride=2,
                    bias=True,
                )
            )

            self.decoders.append(
                ConvBlock(enc_ch * 2, enc_ch)
            )

            decoder_in = enc_ch

        # -------------------------
        # Final projection
        # -------------------------
        self.final_conv = nn.Conv2d(
            base_channels,
            out_channels,
            kernel_size=1,
            bias=True,
        )

    def forward(self, x):

        skips = []

        # -------------------------
        # Encoder
        # -------------------------
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        # -------------------------
        # Bottleneck
        # -------------------------
        x = self.bottleneck(x)

        # -------------------------
        # Decoder
        # -------------------------
        skips = skips[::-1]

        for upconv, decoder, skip in zip(
            self.upconvs,
            self.decoders,
            skips
        ):
            x = upconv(x)

            # Handle odd spatial sizes if necessary
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            x = torch.cat([skip, x], dim=1)
            x = decoder(x)

        # -------------------------
        # Output
        # -------------------------
        x = self.final_conv(x)

        return x


# ------------------------------------------------------------------
# Example
# ------------------------------------------------------------------
if __name__ == "__main__":

    model = UNet(
        in_channels=9,
        out_channels=3,
        depth=4,                 # variable depth
        base_channels=64,
        bottleneck_reduction=2,  # reduced bottleneck
    )

    x = torch.randn(2, 9, 128, 128)

    y = model(x)

    print("Input :", x.shape)
    print("Output:", y.shape)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")