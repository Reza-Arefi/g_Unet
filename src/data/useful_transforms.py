from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.ndimage import rotate as ndi_rotate


class FixedCrop:
    """Deterministic spatial crop for tensors with trailing `(H, W)` dimensions."""

    def __init__(self, size: int | tuple[int, int], top: int = 0, left: int = 0):
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        if len(self.size) != 2:
            raise ValueError("size must be an int or a (height, width) pair")

        self.top = int(top)
        self.left = int(left)
        if self.top < 0 or self.left < 0:
            raise ValueError("top and left must be non-negative")
        if self.size[0] <= 0 or self.size[1] <= 0:
            raise ValueError("crop height and width must be positive")

    def __call__(self, x: Any):
        crop_h, crop_w = self.size
        h, w = x.shape[-2:]
        if self.top + crop_h > h or self.left + crop_w > w:
            raise ValueError(
                f"FixedCrop(size={self.size}, top={self.top}, left={self.left}) "
                f"does not fit tensor spatial shape {(h, w)}"
            )
        return x[..., self.top : self.top + crop_h, self.left : self.left + crop_w]

    def __repr__(self) -> str:
        return f"FixedCrop(size={self.size}, top={self.top}, left={self.left})"


def rotate_spatial_planes(x: Any, angle_deg: float, *, cval: float = 0.0) -> torch.Tensor:
    """Rotate trailing ``(H, W)`` planes in place (``reshape=False`` keeps size)."""
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.float32)
    arr = x.detach().cpu().numpy()
    leading = arr.shape[:-2]
    h, w = arr.shape[-2:]
    flat = arr.reshape(-1, h, w)
    rotated = np.stack(
        [
            ndi_rotate(plane, angle=angle_deg, reshape=False, order=1, mode="constant", cval=cval)
            for plane in flat
        ],
        axis=0,
    )
    rotated = rotated.reshape(*leading, h, w)
    return torch.as_tensor(rotated, dtype=x.dtype, device=x.device)


class CutThenRotateCrop:
    """Axis-aligned crop, then rotate the patch (same workflow as big-window export)."""

    def __init__(
        self,
        size: int | tuple[int, int],
        top: int,
        left: int,
        angle_deg: float,
    ):
        self.fixed = FixedCrop(size, top=top, left=left)
        self.angle_deg = float(angle_deg)
        self.size = self.fixed.size
        self.top = self.fixed.top
        self.left = self.fixed.left

    def __call__(self, x: Any):
        return rotate_spatial_planes(self.fixed(x), self.angle_deg)

    def __repr__(self) -> str:
        return (
            f"CutThenRotateCrop(size={self.size}, top={self.top}, "
            f"left={self.left}, angle_deg={self.angle_deg})"
        )


class RotatedFixedCrop:
    """
    Sample an oriented rectangle on the full grid into an axis-aligned patch.

    ``(top, left)`` is the anchor corner on the full grid. ``angle_deg`` sets the
    rectangle side orientation there.     Output shape is ``(height, width)`` with horizontal and vertical edges
    (straightened training window). ``size`` may be an int (square) or ``(h, w)``.
    """

    def __init__(
        self,
        size: int | tuple[int, int],
        top: int,
        left: int,
        angle_deg: float,
    ):
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        if len(self.size) != 2:
            raise ValueError("size must be an int or a (height, width) pair")

        self.top = int(top)
        self.left = int(left)
        self.angle_deg = float(angle_deg)
        if self.top < 0 or self.left < 0:
            raise ValueError("top and left must be non-negative")
        if self.size[0] <= 0 or self.size[1] <= 0:
            raise ValueError("crop height and width must be positive")

    def _sample_grid(self, h: int, w: int, device, dtype):
        crop_h, crop_w = self.size
        angle = float(np.deg2rad(self.angle_deg))
        cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
        col_off, row_off = torch.meshgrid(
            torch.arange(crop_w, device=device, dtype=dtype),
            torch.arange(crop_h, device=device, dtype=dtype),
            indexing="xy",
        )
        src_col = self.left + col_off * cos_a - row_off * sin_a
        src_row = self.top + col_off * sin_a + row_off * cos_a
        grid_x = 2.0 * src_col / max(w - 1, 1) - 1.0
        grid_y = 2.0 * src_row / max(h - 1, 1) - 1.0
        return torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

    def __call__(self, x: Any):
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)

        leading = x.shape[:-2]
        h, w = x.shape[-2:]
        flat = x.reshape(-1, 1, h, w)
        grid = self._sample_grid(h, w, flat.device, flat.dtype).expand(flat.shape[0], -1, -1, -1)
        sampled = torch.nn.functional.grid_sample(
            flat,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return sampled.reshape(*leading, self.size[0], self.size[1])

    def __repr__(self) -> str:
        return (
            f"RotatedFixedCrop(size={self.size}, top={self.top}, "
            f"left={self.left}, angle_deg={self.angle_deg})"
        )


def rotated_window_corners(
    top: int,
    left: int,
    height: int,
    angle_deg: float,
    width: int | None = None,
) -> np.ndarray:
    """Return polygon vertices `(cols, rows)` for a rotated rectangle at `(left, top)`.

    For a square, pass the same value for ``height`` and ``width``, or omit ``width``.
    """
    if width is None:
        width = height
    angle = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    col_off = np.array([0.0, width, width, 0.0], dtype=np.float64)
    row_off = np.array([0.0, 0.0, height, height], dtype=np.float64)
    cols = left + col_off * cos_a - row_off * sin_a
    rows = top + col_off * sin_a + row_off * cos_a
    return np.column_stack([cols, rows])


def bottom_right_crop(crop_h: int, crop_w: int):
    """Fixed crop anchored at the bottom-right corner of the `(H, W)` plane."""

    def crop(x: Any):
        h, w = x.shape[-2:]
        top = h - int(crop_h)
        left = w - int(crop_w)
        if top < 0 or left < 0:
            raise ValueError(
                f"bottom_right_crop({crop_h}, {crop_w}) does not fit spatial shape {(h, w)}"
            )
        return x[..., top:top + int(crop_h), left:left + int(crop_w)]

    crop.crop_h = int(crop_h)
    crop.crop_w = int(crop_w)
    crop.__doc__ = f"bottom_right_crop(h={crop_h}, w={crop_w})"
    return crop


def max_bottom_right_crop():
    """Largest bottom-right window: uses the full height and width of each tensor."""

    def crop(x: Any):
        return x

    crop.crop_h = None
    crop.crop_w = None
    crop.__doc__ = "max_bottom_right_crop()"
    return crop


def max_bottom_right_valid_window(valid_mask):
    """
    Largest-area rectangle inside ``valid_mask`` anchored at the bottom-right corner.

    Returns dict with keys ``crop_h``, ``crop_w``, ``top``, ``left``.
    """
    mask = np.asarray(valid_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("valid_mask must be 2-D")

    h, w = mask.shape
    best_area = 0
    best = {"crop_h": 0, "crop_w": 0, "top": h, "left": w}

    for top in range(h):
        min_w = w
        for row in range(top, h):
            run = 0
            for col in range(w - 1, -1, -1):
                if mask[row, col]:
                    run += 1
                else:
                    break
            min_w = min(min_w, run)
            if min_w == 0:
                break
            crop_h = row - top + 1
            crop_w = min_w
            area = crop_h * crop_w
            if area > best_area:
                best_area = area
                best = {
                    "crop_h": int(crop_h),
                    "crop_w": int(crop_w),
                    "top": int(top),
                    "left": int(w - crop_w),
                }
    return best


__all__ = [
    "FixedCrop",
    "CutThenRotateCrop",
    "rotate_spatial_planes",
    "RotatedFixedCrop",
    "rotated_window_corners",
    "random_crop",
    "bottom_right_crop",
    "max_bottom_right_crop",
    "max_bottom_right_valid_window",
]


def random_crop(size):
    """Return a transform that applies a random spatial crop of shape `size` to a (..., H, W) tensor."""
    size = (size, size) if isinstance(size, int) else size

    def crop(x):
        """Crop x to a random (crop_h, crop_w) window."""
        crop_h, crop_w = size
        h, w = x.shape[-2:]
        i = torch.randint(0, h - crop_h + 1, ()).item()
        j = torch.randint(0, w - crop_w + 1, ()).item()
        return x[..., i:i + crop_h, j:j + crop_w]

    return crop
