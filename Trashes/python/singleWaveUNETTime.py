import torch
import torch.nn as nn
import torch.nn.functional as F
from timeEmbeding import FiLMBlock,TimeEmbedder,get_num_groups


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

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.GroupNorm(groups2, out_channels),
            nn.ReLU(inplace=True),
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
        t_in                : int = 9,
        time_proj_dim       : int = 64,
        time_emb_dim        : int = 128,
    ):
        super().__init__()

        self.depth = depth

        # ── Time embedder ────────────────────────────────────────
        self.time_embedder = TimeEmbedder(
            t_in     = t_in,
            proj_dim = time_proj_dim,
            out_dim  = time_emb_dim,
        )
        # ── Time embedder ────────────────────────────────────────


        # -------------------------
        # Encoder
        # -------------------------
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.enc_films = nn.ModuleList()

        ch_in = in_channels
        ch_out = base_channels

        encoder_channels = []

        for _ in range(depth):
            self.encoders.append(ConvBlock(ch_in, ch_out))
            self.enc_films.append(FiLMBlock(ch_out, time_emb_dim))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))

            encoder_channels.append(ch_out)

            ch_in = ch_out
            ch_out *= 2

        # -------------------------
        # Bottleneck
        # -------------------------
        bottleneck_channels = ch_in * 2 // bottleneck_reduction
        self.bottleneck = ConvBlock(ch_in, bottleneck_channels)
        self.bottleneck_film = FiLMBlock(bottleneck_channels, time_emb_dim)

        # -------------------------
        # Decoder
        # -------------------------
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.dec_films = nn.ModuleList()

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

            self.dec_films.append(FiLMBlock(enc_ch, time_emb_dim))

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

        ##-- MS --------------------------------------------------------------------
        self.activation = nn.ReLU()
        self.ms1_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=5, padding='same'),
            self.activation,
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=5, padding='same'),
            self.activation,
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, padding='same'),
            self.activation,
        )

        self.ms2_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=9, padding='same'),
            self.activation,
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=9, padding='same'),
            self.activation,
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=9, padding='same'),
            self.activation,
        )

        self.ms3_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=13, padding='same'),
            self.activation,
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=13, padding='same'),
            self.activation,
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=13, padding='same'),
            self.activation,
        )

        self.ms4_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=(8 * 3 + in_channels),
                out_channels=8,
                kernel_size=7,
                padding=3,
            ),
            self.activation,
            nn.Conv2d(in_channels=8, out_channels=out_channels, kernel_size=5, padding=2),
            self.activation,
        )

        # After concatenating DSC and MS
        self.final_layers = nn.Conv2d(
            in_channels = 32 + base_channels, out_channels=out_channels, kernel_size=3, padding=1
        )   


    def _ms(self, x):
        x1 = self.ms1_layers(x)        
        x2 = self.ms2_layers(x)
        x3 = self.ms3_layers(x)        
        return self.ms4_layers(torch.cat([x, x1, x2, x3], dim=1))
    ##-- MS --------------------------------------------------------------------

    def forward(self, x: torch.Tensor, t: torch.Tensor):

        skips = []

        # 1. Compute global time embedding  [B, time_emb_dim]
        t_emb = self.time_embedder(t)

        
        msX = self._ms(x)

        # -------------------------
        # Encoder
        # -------------------------
        for enc, film, pool in zip(self.encoders, self.enc_films, self.pools):
            x = enc(x)
            x = film(x, t_emb)
            skips.append(x)
            x = pool(x)

        # -------------------------
        # Bottleneck
        # -------------------------
        x = self.bottleneck(x)
        x = self.bottleneck_film(x, t_emb)

        # -------------------------
        # Decoder
        # -------------------------
        skips = skips[::-1]

        for upconv, decoder, film, skip in zip(
            self.upconvs, self.decoders, self.dec_films, skips
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
            x = film(x, t_emb)

        # -------------------------
        # Output
        # -------------------------        

        x = self.final_conv(x)
        #print("MS>",msX.shape,x.shape)
        x = x + msX

        #x3 = self.final_layers(torch.cat([x1, x2], dim=1))

        return x


# ------------------------------------------------------------------
# Example
# ------------------------------------------------------------------
if __name__ == "__main__":
    B, H, W = 2, 128, 128
 
    model = singleWaveUNet(
        in_channels          = 9,
        out_channels         = 3,
        depth                = 4,
        base_channels        = 64,
        bottleneck_reduction = 2,
        t_in                 = 9,     # 3 features × 3 frames
        time_proj_dim        = 64,
        time_emb_dim         = 128,
    )
 
    x = torch.randn(B, 9, H, W)
 
    # Build t:  [day_of_year/365, hour_of_day/24, delta_t_hours/24]
    #           repeated for each of the 3 input frames  →  [B, 9]
    day   = torch.full((B, 3), 172 / 365)
    hour  = torch.full((B, 3),  12 / 24)
    delta = torch.full((B, 3),   6 / 24)
    t = torch.stack([day, hour, delta], dim=-1).reshape(B, -1)  # [B, 9]
 
    y = model(x, t)
 
    print(f"Input  x : {x.shape}")
    print(f"Input  t : {t.shape}")
    print(f"Output y : {y.shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")
