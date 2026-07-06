#!/usr/bin/env python3
"""
Lag-dependent dynamics analysis for catalog windows (pool12 export).

Reuses Shalow ``select_window_12`` pool arrays (13–22 train period) — no re-export.
Metrics mirror ``Graph_UNET/Mesh/analyze_mesh_export_dynamics.py``.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PREFIX = ROOT / "Shalow" / "data" / "select_window_12" / "sample_13_22_uv_dt360"
DEFAULT_FIXWIN_DATA_PREFIX = ROOT / "data" / "sample_13_22_smallfixwin_uv_dt360"
DEFAULT_OUT_ROOT = Path(__file__).resolve().parent / "data"


def setup_logging(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "analysis.log"
    logger = logging.getLogger(f"window_dynamics.{out_dir.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def load_window_catalog(data_prefix: Path) -> dict[str, Any]:
    path = Path(data_prefix).with_name(f"{Path(data_prefix).name}_window_catalog.json")
    if not path.is_file():
        raise FileNotFoundError(f"Window catalog not found: {path}")
    return json.loads(path.read_text())


def catalog_entry(catalog: dict[str, Any], window: int) -> dict[str, Any]:
    if not 1 <= window <= 12:
        raise ValueError(f"window must be 1..12, got {window}")
    slot = window - 1
    for entry in catalog["windows"]:
        if int(entry["slot"]) == slot:
            return entry
    raise KeyError(f"No catalog entry for window {window} (slot {slot})")


def load_export_config(data_prefix: Path) -> dict[str, Any]:
    path = Path(data_prefix).with_name(f"{Path(data_prefix).name}_export_config.json")
    if not path.is_file():
        raise FileNotFoundError(f"Export config not found: {path}")
    return json.loads(path.read_text())


def load_pool12_window_series(
    data_prefix: Path | str,
    window: int,
    *,
    anchor_step: int | None = None,
    pool_tag: str = "k1",
) -> dict[str, Any]:
    """
    Build a continuous (T, H, W) u/v series for one catalog window from pool12 export.

    Uses the conditioning anchor frame (step ``conditioning_steps - 1``) at each daily snapshot.
  Rows are sorted by Zarr index (chronological 2013–2022, ~1-day spacing).
    """
    prefix = Path(data_prefix)
    export_cfg = load_export_config(prefix)
    catalog = load_window_catalog(prefix)
    entry = catalog_entry(catalog, window)
    slot = window - 1

    cond_steps = int(export_cfg["conditioning_steps"])
    step = cond_steps - 1 if anchor_step is None else int(anchor_step)

    sims_path = Path(f"{prefix}_sims_pool_{pool_tag}.npy")
    mask_path = Path(f"{prefix}_mask_pool_{pool_tag}.npy")
    indices_path = Path(f"{prefix}_indices.npy")
    for p in (sims_path, mask_path, indices_path):
        if not p.is_file():
            raise FileNotFoundError(f"Missing export array: {p}")

    sims_pool = np.load(sims_path, mmap_mode="r")
    mask_pool = np.load(mask_path, mmap_mode="r")
    indices = np.load(indices_path)

    order = np.argsort(indices, kind="stable")
    u = np.asarray(sims_pool[order, slot, step, 0], dtype=np.float32)
    v = np.asarray(sims_pool[order, slot, step, 1], dtype=np.float32)
    masks = np.asarray(mask_pool[order, slot], dtype=np.float32)

    # Export mask: 1 = land (invalid), 0 = ocean (valid).
    valid = (masks.mean(axis=0) < 0.5)

    dt_hours = int(export_cfg.get("dt", 360))
    # Between daily anchors, spacing is 24 h (see fixed_dataset_indices stride).
    anchor_dt_hours = 24

    return {
        "u": u,
        "v": v,
        "valid": valid,
        "window": window,
        "slot": slot,
        "export_key": entry["export_key"],
        "catalog_entry": entry,
        "n_times": int(u.shape[0]),
        "dt_hours": anchor_dt_hours,
        "dt_days": anchor_dt_hours / 24.0,
        "anchor_step": step,
        "export_config": export_cfg,
        "data_prefix": str(prefix),
        "sorted_indices": indices[order],
        "data_mode": "catalog",
    }


def load_fix_window_series(
    data_prefix: Path | str,
    *,
    export_key: str = "smallfixwin",
    anchor_step: int | None = None,
) -> dict[str, Any]:
    """
    Build a continuous (T, H, W) u/v series from a single-window fixwin export.

    Uses ``_sims.npy`` (not pool arrays). Rows sorted by Zarr index (~1-day spacing).
    """
    prefix = Path(data_prefix)
    export_cfg = load_export_config(prefix)

    cond_steps = int(export_cfg["conditioning_steps"])
    step = cond_steps - 1 if anchor_step is None else int(anchor_step)

    sims_path = Path(f"{prefix}_sims.npy")
    mask_path = Path(f"{prefix}_mask.npy")
    indices_path = Path(f"{prefix}_indices.npy")
    window_path = Path(f"{prefix}_window.npy")
    for p in (sims_path, mask_path, indices_path, window_path):
        if not p.is_file():
            raise FileNotFoundError(f"Missing export array: {p}")

    sims = np.load(sims_path, mmap_mode="r")
    mask_arr = np.load(mask_path, mmap_mode="r")
    indices = np.load(indices_path)
    window_meta = np.load(window_path)

    order = np.argsort(indices, kind="stable")
    u = np.asarray(sims[order, step, 0], dtype=np.float32)
    v = np.asarray(sims[order, step, 1], dtype=np.float32)
    masks = np.asarray(mask_arr[order], dtype=np.float32)

    valid = masks.mean(axis=0) < 0.5

    entry = {
        "export_key": export_key,
        "window_top": int(export_cfg.get("window_top", window_meta[0])),
        "window_left": int(export_cfg.get("window_left", window_meta[1])),
        "window_angle": float(export_cfg.get("window_angle", window_meta[4])),
        "crop_height": int(export_cfg.get("crop_height", window_meta[2])),
        "crop_width": int(export_cfg.get("crop_width", window_meta[3])),
    }

    anchor_dt_hours = 24

    return {
        "u": u,
        "v": v,
        "valid": valid,
        "window": None,
        "slot": None,
        "export_key": export_key,
        "catalog_entry": entry,
        "n_times": int(u.shape[0]),
        "dt_hours": anchor_dt_hours,
        "dt_days": anchor_dt_hours / 24.0,
        "anchor_step": step,
        "export_config": export_cfg,
        "data_prefix": str(prefix),
        "sorted_indices": indices[order],
        "data_mode": "fixwin",
    }


def result_out_dir(out_root: Path | str, summary: dict[str, Any]) -> Path:
    """Output directory for a dynamics summary (catalog or fixwin)."""
    root = Path(out_root)
    if summary.get("data_mode") == "fixwin":
        return root / f"fixwin_{summary['export_key']}"
    return root / f"window_{summary['window']:02d}_{summary['export_key']}"


def spatial_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    a = a.astype(np.float64) - float(np.mean(a))
    b = b.astype(np.float64) - float(np.mean(b))
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if denom < 1e-12:
        return float("nan")
    return float(np.sum(a * b) / denom)


def domain_mean_series(field: np.ndarray, valid: np.ndarray) -> np.ndarray:
    vals = field[:, valid]
    return np.mean(vals, axis=1).astype(np.float64)


def lag_autocorrelation(series: np.ndarray, max_lag: int) -> np.ndarray:
    x = series - np.mean(series)
    var = float(np.mean(x * x))
    if var < 1e-12:
        return np.full(max_lag + 1, np.nan)
    out = np.empty(max_lag + 1, dtype=np.float64)
    out[0] = 1.0
    for lag in range(1, max_lag + 1):
        out[lag] = float(np.mean(x[:-lag] * x[lag:]) / var)
    return out


def persistence_rmse(field: np.ndarray, valid: np.ndarray, lag: int) -> float:
    m = valid
    se = 0.0
    n_pix = 0
    for t in range(field.shape[0] - lag):
        d = (field[t + lag] - field[t])[m]
        se += float(np.sum(d * d))
        n_pix += int(d.size)
    return float(np.sqrt(se / n_pix)) if n_pix else float("nan")


def spatial_pattern_correlation(field: np.ndarray, valid: np.ndarray, lag: int) -> float:
    m = valid
    corrs: list[float] = []
    for t in range(field.shape[0] - lag):
        corrs.append(spatial_corr(field[t][m], field[t + lag][m]))
    finite = [c for c in corrs if np.isfinite(c)]
    return float(np.mean(finite)) if finite else float("nan")


def integral_timescale_days(ac: np.ndarray, dt_days: float) -> float:
    total = 0.0
    for lag in range(1, len(ac)):
        if not np.isfinite(ac[lag]) or ac[lag] <= 0.0:
            break
        total += float(ac[lag])
    return dt_days * total


def radial_kinetic_energy_spectrum(u: np.ndarray, v: np.ndarray, valid: np.ndarray) -> np.ndarray:
    u_m = np.where(valid, u, 0.0)
    v_m = np.where(valid, v, 0.0)
    fu = np.fft.fft2(u_m)
    fv = np.fft.fft2(v_m)
    power = 0.5 * (np.abs(fu) ** 2 + np.abs(fv) ** 2)
    h, w = u.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    r_max = min(cy, cx)
    bins = np.bincount(r.ravel(), power.ravel(), minlength=r_max + 1)[: r_max + 1]
    counts = np.bincount(r.ravel(), minlength=r_max + 1)[: r_max + 1]
    counts = np.maximum(counts, 1)
    return (bins / counts).astype(np.float64)


def spectral_energy_correlation(
    u: np.ndarray,
    v: np.ndarray,
    valid: np.ndarray,
    lag: int,
) -> float:
    specs = [radial_kinetic_energy_spectrum(u[t], v[t], valid) for t in range(u.shape[0])]
    corrs: list[float] = []
    for t in range(u.shape[0] - lag):
        a = specs[t][1:]
        b = specs[t + lag][1:]
        if a.size < 4:
            continue
        corrs.append(spatial_corr(a, b))
    finite = [c for c in corrs if np.isfinite(c)]
    return float(np.mean(finite)) if finite else float("nan")


def eke_field(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return (0.5 * (u * u + v * v)).astype(np.float32)


def compute_lag_table(
    u: np.ndarray,
    v: np.ndarray,
    valid: np.ndarray,
    *,
    max_lag_days: int,
    dt_days: float,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    lags = list(range(1, max_lag_days + 1))
    u_mean = domain_mean_series(u, valid)
    v_mean = domain_mean_series(v, valid)
    eke = eke_field(u, v)
    eke_mean = domain_mean_series(eke, valid)

    u_ac = lag_autocorrelation(u_mean, max_lag_days)
    v_ac = lag_autocorrelation(v_mean, max_lag_days)
    eke_ac = lag_autocorrelation(eke_mean, max_lag_days)

    rows: list[dict[str, Any]] = []
    for lag in lags:
        row = {
            "lag_days": lag,
            "lag_hours": lag * dt_days * 24.0,
            "u_lag_autocorr": float(u_ac[lag]),
            "v_lag_autocorr": float(v_ac[lag]),
            "eke_lag_autocorr": float(eke_ac[lag]),
            "u_persistence_rmse": persistence_rmse(u, valid, lag),
            "v_persistence_rmse": persistence_rmse(v, valid, lag),
            "u_spatial_pattern_corr": spatial_pattern_correlation(u, valid, lag),
            "v_spatial_pattern_corr": spatial_pattern_correlation(v, valid, lag),
            "eke_spatial_pattern_corr": spatial_pattern_correlation(eke, valid, lag),
            "spectral_energy_corr": spectral_energy_correlation(u, v, valid, lag),
        }
        rows.append(row)
        logger.info(
            "lag=%2dd | u_ac=%.3f v_ac=%.3f | u_rmse=%.4f v_rmse=%.4f | u_pat=%.3f spec=%.3f",
            lag,
            row["u_lag_autocorr"],
            row["v_lag_autocorr"],
            row["u_persistence_rmse"],
            row["v_persistence_rmse"],
            row["u_spatial_pattern_corr"],
            row["spectral_energy_corr"],
        )

    integral = {
        "u_integral_timescale_days": integral_timescale_days(u_ac, dt_days),
        "v_integral_timescale_days": integral_timescale_days(v_ac, dt_days),
        "eke_integral_timescale_days": integral_timescale_days(eke_ac, dt_days),
        "dt_days": dt_days,
        "definition": "sum of positive lag autocorrelations (domain-mean anomaly) × dt",
    }
    return pd.DataFrame(rows), integral


def save_plots(
    df: pd.DataFrame,
    integral: dict[str, Any],
    out_dir: Path,
    *,
    title_suffix: str,
    period: str,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    days = df["lag_days"].to_numpy()

    def _save(fig, name: str) -> None:
        path = plots_dir / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(days, df["u_lag_autocorr"], "o-", label="u")
    ax.plot(days, df["v_lag_autocorr"], "s-", label="v")
    ax.plot(days, df["eke_lag_autocorr"], "^-", label="EKE (domain mean)")
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.4)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(f"Lag autocorrelation (domain-mean anomaly)\n{period}\n{title_suffix}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "01_lag_autocorrelation.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(days, df["u_persistence_rmse"], "o-", label="u")
    ax.plot(days, df["v_persistence_rmse"], "s-", label="v")
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("RMSE (physical units)")
    ax.set_title(f"Persistence RMSE: forecast = field at t\n{period}\n{title_suffix}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "02_persistence_rmse.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(days, df["u_spatial_pattern_corr"], "o-", label="u")
    ax.plot(days, df["v_spatial_pattern_corr"], "s-", label="v")
    ax.plot(days, df["eke_spatial_pattern_corr"], "^-", label="EKE")
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Spatial pattern correlation")
    ax.set_title(f"Spatial pattern correlation\n{period}\n{title_suffix}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "03_spatial_pattern_correlation.png")

    labels = ["u", "v", "EKE"]
    vals = [
        integral["u_integral_timescale_days"],
        integral["v_integral_timescale_days"],
        integral["eke_integral_timescale_days"],
    ]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(labels, vals, color=["C0", "C1", "C2"], alpha=0.85)
    ax.set_ylabel("Integral timescale (days)")
    ax.set_title("Integral timescale (from domain-mean autocorrelation)")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "04_integral_timescale.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(days, df["eke_lag_autocorr"], "o-", label="domain-mean AC")
    ax.plot(days, df["eke_spatial_pattern_corr"], "s-", label="spatial pattern corr")
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Correlation")
    ax.set_title(f"EKE decorrelation  EKE=½(u²+v²)\n{period}\n{title_suffix}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "05_eke_decorrelation.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(days, df["spectral_energy_corr"], "o-", color="C4")
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Spectrum correlation")
    ax.set_title(f"Spectral-energy decorrelation (radial KE spectrum)\n{period}\n{title_suffix}")
    ax.grid(True, alpha=0.3)
    _save(fig, "06_spectral_energy_decorrelation.png")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes[0, 0].plot(days, df["u_lag_autocorr"], label="u")
    axes[0, 0].plot(days, df["v_lag_autocorr"], label="v")
    axes[0, 0].set_title("Lag autocorrelation")
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].bar(labels, vals, color=["C0", "C1", "C2"])
    axes[1, 0].set_title("Integral timescale (days)")
    axes[1, 1].plot(days, df["eke_lag_autocorr"], label="AC")
    axes[1, 1].plot(days, df["eke_spatial_pattern_corr"], label="pattern")
    axes[1, 1].set_title("EKE decorrelation")
    axes[1, 1].legend(fontsize=8)
    axes[1, 2].plot(days, df["spectral_energy_corr"], color="C4")
    axes[1, 2].set_title("Spectral-energy decorr")
    axes[0, 1].plot(days, df["u_persistence_rmse"], label="u")
    axes[0, 1].plot(days, df["v_persistence_rmse"], label="v")
    axes[0, 1].set_title("Persistence RMSE")
    axes[0, 1].legend(fontsize=8)
    axes[0, 2].plot(days, df["u_spatial_pattern_corr"], label="u")
    axes[0, 2].plot(days, df["v_spatial_pattern_corr"], label="v")
    axes[0, 2].set_title("Spatial pattern corr")
    axes[0, 2].legend(fontsize=8)
    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Lag (days)")
    fig.suptitle(f"Window dynamics analysis | {title_suffix}", fontsize=12)
    fig.tight_layout()
    _save(fig, "00_dashboard.png")

    return saved


def run_window_dynamics_analysis(
    series: dict[str, Any],
    out_dir: Path | str,
    *,
    max_lag_days: int = 30,
    cache_npz: bool = True,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    logger = setup_logging(out_dir)

    u, v, valid = series["u"], series["v"], series["valid"]
    dt_days = float(series["dt_days"])
    export_cfg = series["export_config"]
    entry = series["catalog_entry"]
    data_mode = series.get("data_mode", "catalog")
    window = series.get("window")

    if cache_npz:
        cache_dir = out_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_dir / "uv_series.npz",
            u=u,
            v=v,
            valid=valid,
            sorted_indices=series["sorted_indices"],
        )

    period = f"{export_cfg.get('start_time')} → {export_cfg.get('end_time')}"
    hw = f"{entry['crop_height']}×{entry['crop_width']}"
    if data_mode == "fixwin":
        title_suffix = f"{entry['export_key']} (fixwin) | {hw} ocean mask"
        log_label = entry["export_key"]
    else:
        title_suffix = f"W{window:02d} {entry['export_key']} | {hw} ocean mask"
        log_label = f"W{window:02d} {entry['export_key']}"

    logger.info("Window dynamics analysis %s", log_label)
    logger.info("Data prefix: %s", series["data_prefix"])
    logger.info("Output -> %s", out_dir)
    logger.info(
        "Time series u=%s valid_pixels=%d n_times=%d dt_days=%.1f",
        u.shape,
        int(valid.sum()),
        u.shape[0],
        dt_days,
    )

    max_lag = min(int(max_lag_days), u.shape[0] - 2)
    df, integral = compute_lag_table(
        u, v, valid,
        max_lag_days=max_lag,
        dt_days=dt_days,
        logger=logger,
    )
    plots = save_plots(
        df,
        integral,
        out_dir,
        title_suffix=title_suffix,
        period=period,
    )

    df.to_csv(out_dir / "lag_metrics.csv", index=False)
    (out_dir / "integral_timescales.json").write_text(json.dumps(integral, indent=2) + "\n")

    lag_snap = df[df["lag_days"] == max_lag].iloc[0].to_dict() if len(df) else {}
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_mode": data_mode,
        "window": window,
        "slot": series.get("slot"),
        "export_key": entry["export_key"],
        "data_prefix": series["data_prefix"],
        "period": period,
        "spatial": {
            "crop_hw": [entry["crop_height"], entry["crop_width"]],
            "window_top": entry["window_top"],
            "window_left": entry["window_left"],
            "window_angle": entry.get("window_angle", 0.0),
            "valid_ocean_pixels": int(valid.sum()),
            "land_fraction_mean": float(1.0 - valid.mean()),
        },
        "n_times": int(u.shape[0]),
        "anchor_step": int(series["anchor_step"]),
        "dt_days": dt_days,
        "max_lag_days": max_lag,
        "integral_timescales_days": integral,
        "lag_30d_snapshot": lag_snap,
        "outputs": {
            "lag_metrics_csv": str(out_dir / "lag_metrics.csv"),
            "integral_timescales_json": str(out_dir / "integral_timescales.json"),
            "plots": [str(p) for p in plots],
            "log": str(out_dir / "analysis.log"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("Wrote %s", out_dir / "summary.json")
    return summary


def analyze_selected_windows(
    windows: list[int],
    *,
    data_prefix: Path | str | None = None,
    out_root: Path | str | None = None,
    max_lag_days: int = 30,
) -> list[dict[str, Any]]:
    prefix = Path(data_prefix or DEFAULT_DATA_PREFIX)
    root = Path(out_root or DEFAULT_OUT_ROOT)
    summaries: list[dict[str, Any]] = []
    for w in windows:
        series = load_pool12_window_series(prefix, w)
        out_dir = root / f"window_{w:02d}_{series['export_key']}"
        summaries.append(
            run_window_dynamics_analysis(series, out_dir, max_lag_days=max_lag_days)
        )
    return summaries


def analyze_fix_window(
    *,
    data_prefix: Path | str | None = None,
    export_key: str = "smallfixwin",
    out_root: Path | str | None = None,
    max_lag_days: int = 30,
) -> dict[str, Any]:
    prefix = Path(data_prefix or DEFAULT_FIXWIN_DATA_PREFIX)
    root = Path(out_root or DEFAULT_OUT_ROOT)
    series = load_fix_window_series(prefix, export_key=export_key)
    out_dir = root / f"fixwin_{export_key}"
    return run_window_dynamics_analysis(series, out_dir, max_lag_days=max_lag_days)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Lag dynamics for catalog windows (pool12)")
    p.add_argument("--windows", type=int, nargs="+", default=list(range(1, 13)))
    p.add_argument("--data-prefix", type=Path, default=DEFAULT_DATA_PREFIX)
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--max-lag-days", type=int, default=30)
    args = p.parse_args()

    summaries = analyze_selected_windows(
        args.windows,
        data_prefix=args.data_prefix,
        out_root=args.out_root,
        max_lag_days=args.max_lag_days,
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
