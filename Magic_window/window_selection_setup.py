"""Shared setup for window_selection.ipynb — auto-loaded by window cells."""
from pathlib import Path
import os
import sys

# Always anchor to this package directory (works when imported from Shalow notebooks).
MW_DIR = Path(__file__).resolve().parent
ROOT = MW_DIR.parent
NOTEBOOK_DIR = Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import zarr
from matplotlib.patches import Polygon, Rectangle

from src.data.useful_transforms import FixedCrop, RotatedFixedCrop, rotated_window_corners

ZARR_PATH = os.getenv("GUIANA_ZARR_PATH", "s3://oceangen-data/margem_equatorial_tensor.zarr")
VIZ_TIME_IDX = 0
_SETUP_DONE = False

MAGIC_TOP = MAGIC_LEFT = MAGIC_H = MAGIC_W = None
MAGIC_CENTER_ROW = MAGIC_CENTER_COL = None
EXPORT_ROW_SLICE = EXPORT_COL_SLICE = None


def ensure_setup():
    global _SETUP_DONE, MAGIC_TOP, MAGIC_LEFT, MAGIC_H, MAGIC_W
    global MAGIC_CENTER_ROW, MAGIC_CENTER_COL, EXPORT_ROW_SLICE, EXPORT_COL_SLICE
    if _SETUP_DONE:
        return
    pt_path = MW_DIR / "magic_window_lat_lon_bathymetry_wh(1).pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"Expected file at {pt_path}")
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    EXPORT_ROW_SLICE = tuple(data["row_slice"])
    EXPORT_COL_SLICE = (85, 213)
    MAGIC_TOP = int(EXPORT_ROW_SLICE[0])
    MAGIC_LEFT = int(EXPORT_COL_SLICE[0])
    MAGIC_H = int(EXPORT_ROW_SLICE[1] - EXPORT_ROW_SLICE[0])
    MAGIC_W = int(EXPORT_COL_SLICE[1] - EXPORT_COL_SLICE[0])
    MAGIC_CENTER_ROW = MAGIC_TOP + (MAGIC_H - 1) / 2.0
    MAGIC_CENTER_COL = MAGIC_LEFT + (MAGIC_W - 1) / 2.0
    _SETUP_DONE = True


def bathymetry_cmap(base="terrain_r"):
    cmap = plt.get_cmap(base).copy()
    cmap.set_bad(color="black")
    cmap.set_under(color="black")
    return cmap


def mask_land(field):
    return np.ma.masked_where(~np.isfinite(field) | (field == 0), field)


def _top_left_from_center(pivot_col, pivot_row, height, width, angle_deg):
    a = np.deg2rad(float(angle_deg))
    cc, rc = (width - 1) / 2.0, (height - 1) / 2.0
    left = pivot_col - cc * np.cos(a) + rc * np.sin(a)
    top = pivot_row - cc * np.sin(a) - rc * np.cos(a)
    return int(round(top)), int(round(left))


def resolve_window(cfg: dict) -> dict:
    h = int(cfg["crop_height"])
    w = int(cfg["crop_width"])
    ang = float(cfg["window_angle"])
    pivot = str(cfg.get("rotation_pivot", "top_left"))
    pivot_col = cfg.get("pivot_col")
    pivot_row = cfg.get("pivot_row")
    top = cfg.get("window_top")
    left = cfg.get("window_left")
    if pivot == "center":
        if pivot_col is None or pivot_row is None:
            if top is None or left is None:
                raise ValueError("center pivot needs pivot_col/pivot_row or window_top/window_left")
            pivot_col = float(left) + (w - 1) / 2.0
            pivot_row = float(top) + (h - 1) / 2.0
        top, left = _top_left_from_center(pivot_col, pivot_row, h, w, ang)
    else:
        if top is None or left is None:
            raise ValueError("top_left pivot needs window_top and window_left")
        top, left = int(top), int(left)
    return {
        "export_key": str(cfg["export_key"]),
        "crop_height": h,
        "crop_width": w,
        "window_top": int(top),
        "window_left": int(left),
        "window_angle": ang,
        "rotation_pivot": pivot,
        "pivot_col": float(pivot_col) if pivot_col is not None else None,
        "pivot_row": float(pivot_row) if pivot_row is not None else None,
        "color": cfg.get("color", "#FFEB3B"),
    }


def load_fields(time_idx=VIZ_TIME_IDX):
    zg = zarr.open_group(ZARR_PATH, mode="r")
    ch = [c.decode() if isinstance(c, bytes) else str(c) for c in zg["channel"][:]]
    arr = np.asarray(zg["ocean_tensor"][int(time_idx)])
    if arr.ndim == 5:
        arr = arr[0]
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    idx = {n: i for i, n in enumerate(ch)}
    fields = {n: arr[idx[n]] for n in ("bathymetry", "u", "v")}
    return fields, str(pd.to_datetime(zg["time"][int(time_idx)]))


def print_window_info(spec):
    rows = [
        ("export_key", spec["export_key"]),
        ("window_top", spec["window_top"]),
        ("window_left", spec["window_left"]),
        ("window_angle", spec["window_angle"]),
        ("rotation_pivot", spec["rotation_pivot"]),
        ("data_dir", f"Magic_window/data/{spec['export_key']}"),
    ]
    df = pd.DataFrame(rows, columns=["field", "value"])
    try:
        from IPython.display import display
        display(df.style.hide(axis="index"))
    except Exception:
        print(df.to_string(index=False))
    print(
        f"Copy: CROP_HEIGHT={spec['crop_height']}, CROP_WIDTH={spec['crop_width']}, "
        f"WINDOW_TOP={spec['window_top']}, WINDOW_LEFT={spec['window_left']}, "
        f"WINDOW_ANGLE={spec['window_angle']}"
    )
    return spec


def _outline(ax, spec):
    t, l = spec["window_top"], spec["window_left"]
    h, w, a = spec["crop_height"], spec["crop_width"], spec["window_angle"]
    if a == 0:
        ax.add_patch(Rectangle((l, t), w, h, fill=False, edgecolor=spec["color"], lw=3, label=spec["export_key"]))
    else:
        ax.add_patch(Polygon(rotated_window_corners(t, l, h, a, w), fill=False, edgecolor=spec["color"], lw=3, label=spec["export_key"]))
    ax.add_patch(Rectangle((MAGIC_LEFT, MAGIC_TOP), MAGIC_W, MAGIC_H, fill=False, edgecolor="#FF4444", lw=2, ls="--", label="magic ref"))


def visualize_window(spec, time_idx=VIZ_TIME_IDX):
    fields, t_str = load_fields(time_idx)
    bathy, u, v = fields["bathymetry"], fields["u"], fields["v"]
    b_ok = bathy[np.isfinite(bathy) & (bathy != 0)]
    b_vmin, b_vmax = (float(b_ok.min()), float(b_ok.max())) if b_ok.size else (None, None)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (fld, title, cmap, vmin, vmax) in zip(axes[0], [
        (bathy, "Bathymetry", bathymetry_cmap(), b_vmin, b_vmax),
        (u, "u", "RdBu_r", None, None),
        (v, "v", "RdBu_r", None, None),
    ]):
        im = ax.imshow(mask_land(fld) if title == "Bathymetry" else fld, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
        _outline(ax, spec)
        ax.set_title(f"Full — {title}")
        fig.colorbar(im, ax=ax, shrink=0.7)
    axes[0, 0].legend(fontsize=7, loc="lower right")
    stack = torch.tensor(np.stack([u, v, bathy]), dtype=torch.float32)
    h, w = spec["crop_height"], spec["crop_width"]
    if spec["window_angle"] == 0:
        crop_t = FixedCrop(size=(h, w), top=spec["window_top"], left=spec["window_left"])
    else:
        crop_t = RotatedFixedCrop(size=(h, w), top=spec["window_top"], left=spec["window_left"], angle_deg=spec["window_angle"])
    cropped = crop_t(stack).numpy()
    for ax, (fld, title, cmap) in zip(axes[1], [
        (cropped[2], "Bathymetry crop", bathymetry_cmap()),
        (cropped[0], "u crop", "RdBu_r"),
        (cropped[1], "v crop", "RdBu_r"),
    ]):
        im = ax.imshow(mask_land(fld) if "Bathymetry" in title else fld, cmap=cmap, vmin=b_vmin if "Bathymetry" in title else None, vmax=b_vmax if "Bathymetry" in title else None, origin="upper")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle(
        f"[{spec['export_key']}] top={spec['window_top']}, left={spec['window_left']}, angle={spec['window_angle']}° | t={t_str}",
        fontsize=13,
    )
    plt.tight_layout()
    plt.show()


def preview_window(cfg: dict, time_idx=VIZ_TIME_IDX):
    ensure_setup()
    spec = resolve_window(cfg)
    print_window_info(spec)
    visualize_window(spec, time_idx=time_idx)
    return spec


def bootstrap_notebook(ns: dict | None = None) -> dict:
    """Load setup into a namespace (notebook cell globals)."""
    import importlib.util
    target = ns if ns is not None else globals()
    if "preview_window" in target and target.get("MAGIC_TOP") is not None:
        target["ensure_setup"]()
        return target
    setup_path = MW_DIR / "window_selection_setup.py"
    spec = importlib.util.spec_from_file_location("window_selection_setup", setup_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ensure_setup()
    target.update({k: v for k, v in vars(mod).items() if not k.startswith("__")})
    return target
