import torch
from torch import nn


class DSCMS(nn.Module):
    """Hybrid Downsampled Skip-Connection/Multi-Scale model proposed by Fukami et al. (2019, JFM).
    Ref: http://www.seas.ucla.edu/fluidflow/lib/hDSC_MS.py
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(DSCMS, self).__init__()

        # Down-sampled skip-connection model (DSC)
        self.activation = nn.ReLU()
        
        self.down_1 = nn.MaxPool2d(kernel_size=8, padding=0)
        self.dsc1_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding='same'),
            self.activation,
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same'),
            self.activation,
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )

        self.down_2 = nn.MaxPool2d(kernel_size=4, padding=0)
        self.dsc2_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels + 32, out_channels=32, kernel_size=3, padding='same'
            ),
            self.activation,
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same'),
            self.activation,
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )

        self.down_3 = nn.MaxPool2d(kernel_size=2, padding=0)
        self.dsc3_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels + 32, out_channels=32, kernel_size=3, padding='same'
            ),
            self.activation,
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same'),
            self.activation,
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )

        self.dsc4_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels + 32, out_channels=32, kernel_size=3, padding='same'
            ),
            self.activation,
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same'),
            self.activation,
        )

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
            nn.Conv2d(in_channels=8, out_channels=3, kernel_size=5, padding=2),
            self.activation,
        )

        # After concatenating DSC and MS
        self.final_layers = nn.Conv2d(
            in_channels=32 + 3, out_channels=out_channels, kernel_size=3, padding=1
        )

    def _dsc(self, x):
        x1 = self.dsc1_layers(self.down_1(x))
        mp2 = self.down_2(x)
        x2 = self.dsc2_layers(torch.cat([x1, mp2], dim=1))
        mp3 = self.down_3(x)
        x3 = self.dsc3_layers(torch.cat([x2, mp3], dim=1))
        return self.dsc4_layers(torch.cat([x, x3], dim=1))

    def _ms(self, x):
        x1 = self.ms1_layers(x)        
        x2 = self.ms2_layers(x)
        x3 = self.ms3_layers(x)        
        return self.ms4_layers(torch.cat([x, x1, x2, x3], dim=1))

    def forward(self, x):
        x1 = self._dsc(x)
        x2 = self._ms(x)
        x3 = self.final_layers(torch.cat([x1, x2], dim=1))
        return x3