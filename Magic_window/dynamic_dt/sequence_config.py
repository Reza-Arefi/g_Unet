"""Shared sequence layout for dynamic_dt: 5 conditioning frames → 1 forecast."""

from __future__ import annotations

# Frames: t-4, t-3, t-2, t-1, t (conditioning) → predict t+1
CONDITIONING_STEPS = 5
FORECAST_STEPS = 1

# Model input: lb * nf channels (5 timesteps × 2 uv channels)
INPUT_CHANNELS = CONDITIONING_STEPS * 2  # iD=2
