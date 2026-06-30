import math
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

##################################################################################
# ──────────────────────────────────────────────────────────────────
# Time Embedding  (sinusoidal projection → MLP)
# ──────────────────────────────────────────────────────────────────
 
class TimeEmbedder(nn.Module):
    """
    Encodes a raw time feature vector into a dense embedding.
 
    Input shape : [B, t_in]
        Suggested features per frame (concatenated across 3 frames):
            • day_of_year  / 365          – seasonal signal
            • hour_of_day  / 24           – diurnal signal
            • Δt to next frame (hours)    – inter-frame gap
        → t_in = 3 features × 3 frames = 9  (or whatever you pass)
 
    Each scalar feature is first lifted to `proj_dim` dimensions via
    sinusoidal projection (no learned params), then the full vector is
    passed through a two-layer MLP to produce the final embedding used
    by FiLM blocks.
    """
 
    def __init__(self, t_in: int = 9, proj_dim: int = 64, out_dim: int = 128):
        super().__init__()
        self.proj_dim = proj_dim
        self.t_in     = t_in
 
        # Learnable MLP that maps sinusoidal features → embedding
        mlp_in = t_in * proj_dim * 2          # *2 because sin + cos
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, out_dim * 2),
            nn.SiLU(),
            nn.Linear(out_dim * 2, out_dim),
            nn.SiLU(),
        )
 
        # Fixed frequency bank  [proj_dim]
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(proj_dim) / (proj_dim - 1)
        )
        self.register_buffer("freqs", freqs)   # not a parameter
 
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t : [B, t_in]  – normalised raw time features
        returns: [B, out_dim]
        """
        # [B, t_in, 1] × [1, 1, proj_dim]  →  [B, t_in, proj_dim]
        args = t.unsqueeze(-1) * self.freqs.unsqueeze(0).unsqueeze(0)
        # sin + cos  →  [B, t_in, proj_dim*2]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        # flatten  →  [B, t_in * proj_dim * 2]
        emb = emb.flatten(1)
        return self.mlp(emb)                   # [B, out_dim]
##################################################################################
# ──────────────────────────────────────────────────────────────────
# FiLM Block
# ──────────────────────────────────────────────────────────────────
 
class FiLMBlock(nn.Module):
    """
    Feature-wise Linear Modulation.
 
    After a ConvBlock produces feature map `x` with `channels`
    feature maps, this block applies an affine transformation whose
    parameters (scale γ, shift β) are *predicted from the time
    embedding* via a single linear layer:
 
        γ, β  =  split( Linear(t_emb), dim=channels )
        out   =  GroupNorm(x) * (1 + γ) + β
 
    The residual `1 + γ` (instead of just γ) keeps the initial
    behaviour close to identity, stabilising early training.
    """
 
    def __init__(self, channels: int, time_dim: int):
        super().__init__()
        groups = get_num_groups(channels)
        self.norm           = nn.GroupNorm(groups, channels)
        self.to_scale_shift = nn.Linear(time_dim, channels * 2)
 
        # Initialise to near-identity
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)
 
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        x     : [B, C, H, W]
        t_emb : [B, time_dim]
        """
        scale, shift = self.to_scale_shift(t_emb).chunk(2, dim=-1)
        scale = scale[:, :, None, None]        # broadcast → [B, C, 1, 1]
        shift = shift[:, :, None, None]
        return self.norm(x) * (1.0 + scale) + shift
##################################################################################
##################################################################################
##################################################################################
##################################################################################
