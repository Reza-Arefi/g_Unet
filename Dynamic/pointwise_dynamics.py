#!/usr/bin/env python3
"""
Pointwise (per-grid-cell) temporal dynamics for catalog windows.

For each ocean pixel computes:
  - Autocorrelation curve ρ_ij(τ)
  - Integral timescale T_ij
  - First lag where ρ_ij drops below 0.7, 0.5, 0.3
  - Mutual information curve I_ij(τ)  (Gaussian proxy from ρ, optional binned)
  - Dominant oscillation period from temporal power spectrum

Reuses pool12 export via ``window_dynamics.load_pool12_window_series``.
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

from window_dynamics import (
    DEFAULT_DATA_PREFIX,
    DEFAULT_OUT_ROOT,
    load_pool12_window_series,
    setup_logging,
)

POINTWISE_OUT_ROOT = Path(__file__).resolve().parent / "pointwise_data"
DECORR_THRESHOLDS = (0.7, 0.5, 0.3)


def _field_stack(u: np.ndarray, v: np.ndarray, field: str) -> np.ndarray:
    if field == "u":
        return u
    if field == "v":
        return v
    if field == "speed":
        return np.sqrt(u * u + v * v).astype(np.float32)
    raise ValueError(f"field must be u|v|speed, got {field!r}")


def pointwise_autocorrelation(
    series: np.ndarray,
    valid: np.ndarray,
    max_lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized lag autocorrelation per pixel.

    Returns
    -------
    rho : (max_lag, H, W)  ρ(τ), τ = 1..max_lag
    var : (H, W) temporal variance (anomaly)
    """
    t_len, h, w = series.shape
    flat = series.reshape(t_len, -1).astype(np.float64)
    mean = flat.mean(axis=0)
    flat = flat - mean
    var = np.mean(flat * flat, axis=0)
    var = np.where(var < 1e-12, np.nan, var)

    rho = np.full((max_lag, h * w), np.nan, dtype=np.float64)
    for lag in range(1, max_lag + 1):
        cov = np.mean(flat[:-lag] * flat[lag:], axis=0)
        rho[lag - 1] = cov / var

    rho = rho.reshape(max_lag, h, w)
    v2d = var.reshape(h, w)
    invalid = ~valid
    rho[:, invalid] = np.nan
    v2d[invalid] = np.nan
    return rho.astype(np.float32), v2d.astype(np.float32)


def pointwise_integral_timescale(rho: np.ndarray, dt_days: float) -> np.ndarray:
    """T_ij = dt * sum of positive ρ_ij(τ) until first non-positive lag."""
    max_lag, h, w = rho.shape
    tint = np.zeros((h, w), dtype=np.float32)
    active = np.isfinite(rho[0]) if max_lag else np.zeros((h, w), dtype=bool)
    for lag in range(max_lag):
        r = rho[lag]
        add = np.where(active & (r > 0), r * dt_days, 0.0)
        tint += add.astype(np.float32)
        active = active & (r > 0)
    tint[~active & (tint == 0)] = np.nan
    return tint


def pointwise_first_lag_below(
    rho: np.ndarray,
    threshold: float,
    dt_days: float,
) -> np.ndarray:
    """First τ (days) where ρ_ij(τ) < threshold; nan if never within computed lags."""
    max_lag, h, w = rho.shape
    out = np.full((h, w), np.nan, dtype=np.float32)
    for lag in range(max_lag):
        tau_days = (lag + 1) * dt_days
        mask = np.isnan(out) & np.isfinite(rho[lag]) & (rho[lag] < threshold)
        out[mask] = tau_days
    return out


def gaussian_mi_from_rho(rho: np.ndarray) -> np.ndarray:
    """I_ij(τ) ≈ -½ ln(1 - ρ²) nats (Gaussian proxy, fast)."""
    r2 = np.clip(rho.astype(np.float64) ** 2, 0.0, 1.0 - 1e-8)
    return (-0.5 * np.log(1.0 - r2)).astype(np.float32)


def binned_mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 16,
) -> float:
    """Histogram MI for one pixel pair (x[t], y[t])."""
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < n_bins * 4:
        return float("nan")
    hist, _, _ = np.histogram2d(x, y, bins=n_bins)
    total = hist.sum()
    if total <= 0:
        return float("nan")
    pxy = hist / total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    nz = pxy > 0
    denom = px @ py
    return float(np.sum(pxy[nz] * np.log(pxy[nz] / denom[nz])))


def pointwise_binned_mi(
    series: np.ndarray,
    valid: np.ndarray,
    max_lag: int,
    *,
    n_bins: int = 16,
    lag_stride: int = 1,
) -> np.ndarray:
    """
    Exact binned MI per pixel (slow). Only valid ocean pixels are filled.

    lag_stride > 1 computes MI every N lags and linearly fills gaps.
    """
    t_len, h, w = series.shape
    lags = list(range(1, max_lag + 1, max(1, lag_stride)))
    mi = np.full((max_lag, h, w), np.nan, dtype=np.float32)
    flat = series.reshape(t_len, -1)
    valid_idx = np.flatnonzero(valid.ravel())

    for lag in lags:
        x0 = flat[:-lag]
        x1 = flat[lag:]
        for k in valid_idx:
            mi[lag - 1, k // w, k % w] = binned_mutual_information(
                x0[:, k], x1[:, k], n_bins=n_bins
            )
    return mi


def pointwise_power_spectrum(
    series: np.ndarray,
    valid: np.ndarray,
    dt_days: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Temporal FFT per pixel.

    Returns
    -------
    dom_period_days : (H, W) period of peak non-DC power
    peak_freq_cpd : (H, W) peak frequency (cycles/day)
    spec : (n_freq, H, W) one-sided power spectrum
    """
    t_len, h, w = series.shape
    flat = series.reshape(t_len, -1).astype(np.float64)
    flat = flat - flat.mean(axis=0, keepdims=True)
    fft = np.fft.rfft(flat, axis=0)
    power = (np.abs(fft) ** 2).astype(np.float64)
    freq = np.fft.rfftfreq(t_len, d=dt_days)

    dom_period = np.full((h * w,), np.nan, dtype=np.float32)
    peak_freq = np.full((h * w,), np.nan, dtype=np.float32)
    for k in range(h * w):
        p = power[1:, k]
        if not np.any(np.isfinite(p)) or p.max() <= 0:
            continue
        i = int(np.argmax(p)) + 1
        f = float(freq[i])
        if f > 0:
            peak_freq[k] = f
            dom_period[k] = 1.0 / f

    spec = power.reshape(-1, h, w).astype(np.float32)
    dom_period = dom_period.reshape(h, w)
    peak_freq = peak_freq.reshape(h, w)
    dom_period[~valid] = np.nan
    peak_freq[~valid] = np.nan
    spec[:, ~valid] = np.nan
    return dom_period, peak_freq, spec


def summarize_valid_pixels(
    maps: dict[str, np.ndarray],
    valid: np.ndarray,
    *,
    percentiles: tuple[float, ...] = (5, 25, 50, 75, 95),
) -> dict[str, Any]:
    """Distribution stats over valid ocean pixels."""
    out: dict[str, Any] = {}
    pcts = list(percentiles)
    for name, arr in maps.items():
        if arr.ndim == 2:
            vals = arr[valid]
        elif arr.ndim == 3:
            # summarize each lag as domain median curve
            med_curve = np.nanmedian(arr[:, valid], axis=1)
            out[f"{name}_domain_median_curve"] = med_curve.tolist()
            vals = arr[:, valid].ravel()
        else:
            continue
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            out[name] = {"n": 0}
            continue
        out[name] = {
            "n": int(vals.size),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "percentiles": {str(p): float(np.percentile(vals, p)) for p in pcts},
        }
    return out


def save_pointwise_plots(
    out_dir: Path,
    *,
    field: str,
    rho: np.ndarray,
    tint: np.ndarray,
    decorr: dict[float, np.ndarray],
    mi: np.ndarray,
    dom_period: np.ndarray,
    valid: np.ndarray,
    dt_days: float,
    title_suffix: str,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    lags = np.arange(1, rho.shape[0] + 1) * dt_days
    med_rho = np.nanmedian(rho[:, valid], axis=1)
    med_mi = np.nanmedian(mi[:, valid], axis=1)

    def _masked_map(ax, data, title, cmap="viridis", vmin=None, vmax=None):
        show = np.where(valid, data, np.nan)
        im = ax.imshow(show, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 1. Spatial maps dashboard
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    _masked_map(axes[0, 0], tint, f"{field} integral timescale (days)")
    for j, thr in enumerate(DECORR_THRESHOLDS):
        _masked_map(axes[0, j + 1] if j < 2 else axes[1, 0], decorr[thr], f"first lag ρ<{thr} (days)")
    _masked_map(axes[1, 1], dom_period, "dominant period (days)", cmap="magma")
    _masked_map(axes[1, 2], rho[-1], f"ρ at max lag ({lags[-1]:.0f}d)", cmap="coolwarm", vmin=-1, vmax=1)
    fig.suptitle(f"Pointwise dynamics maps | {title_suffix}", fontsize=11)
    fig.tight_layout()
    p = plots_dir / f"01_{field}_spatial_maps.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # 2. Domain median ρ and MI curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(lags, med_rho, "o-", label="median ρ_ij(τ)")
    axes[0].axhline(0.7, color="C1", ls="--", alpha=0.5, label="0.7")
    axes[0].axhline(0.5, color="C2", ls="--", alpha=0.5, label="0.5")
    axes[0].axhline(0.3, color="C3", ls="--", alpha=0.5, label="0.3")
    axes[0].set_xlabel("Lag τ (days)")
    axes[0].set_ylabel("ρ")
    axes[0].set_title(f"{field}: domain-median autocorrelation")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(lags, med_mi, "s-", color="C4", label="median I_ij(τ)")
    axes[1].set_xlabel("Lag τ (days)")
    axes[1].set_ylabel("MI (nats, Gaussian proxy)")
    axes[1].set_title(f"{field}: domain-median mutual information")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(title_suffix, fontsize=10)
    fig.tight_layout()
    p = plots_dir / f"02_{field}_median_rho_mi_curves.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # 3. Histograms of T_int and decorrelation lags
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, data, label in zip(
        axes,
        [tint, decorr[0.5], dom_period],
        ["T_int (days)", "first lag ρ<0.5 (days)", "dominant period (days)"],
    ):
        vals = data[valid]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            ax.hist(vals, bins=40, color="steelblue", alpha=0.85, edgecolor="white")
            ax.axvline(np.median(vals), color="crimson", ls="--", label=f"median={np.median(vals):.1f}")
            ax.legend(fontsize=8)
        ax.set_xlabel(label)
        ax.set_ylabel("pixel count")
    fig.suptitle(f"{field}: pixel-wise distributions | {title_suffix}", fontsize=10)
    fig.tight_layout()
    p = plots_dir / f"03_{field}_histograms.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # 4. Sample pixel ρ curves (fast/slow decorrelation)
    flat_t = tint[valid]
    flat_idx = np.flatnonzero(valid)
    if flat_t.size >= 2:
        order = np.argsort(flat_t)
        slow_k = flat_idx[order[-1]]
        fast_k = flat_idx[order[0]]
        sy, sx = slow_k // tint.shape[1], slow_k % tint.shape[1]
        fy, fx = fast_k // tint.shape[1], fast_k % tint.shape[1]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(lags, rho[:, sy, sx], "o-", label=f"slow pixel ({sy},{sx}) T={tint[sy,sx]:.1f}d")
        ax.plot(lags, rho[:, fy, fx], "s-", label=f"fast pixel ({fy},{fx}) T={tint[fy,fx]:.1f}d")
        ax.plot(lags, med_rho, "k--", alpha=0.6, label="domain median")
        ax.set_xlabel("Lag τ (days)")
        ax.set_ylabel("ρ_ij(τ)")
        ax.set_title(f"{field}: example pointwise autocorrelation curves")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = plots_dir / f"04_{field}_sample_rho_curves.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

    return saved


def analyze_field_pointwise(
    series: np.ndarray,
    valid: np.ndarray,
    *,
    field_name: str,
    max_lag_days: int,
    dt_days: float,
    mi_mode: str = "gaussian",
    mi_bins: int = 16,
    mi_lag_stride: int = 1,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    log = logger or logging.getLogger("pointwise")
    max_lag = min(max_lag_days, series.shape[0] - 2)
    log.info("  %s: autocorrelation (max_lag=%d)", field_name, max_lag)
    rho, _var = pointwise_autocorrelation(series, valid, max_lag)
    tint = pointwise_integral_timescale(rho, dt_days)
    decorr = {thr: pointwise_first_lag_below(rho, thr, dt_days) for thr in DECORR_THRESHOLDS}

    if mi_mode == "binned":
        log.info("  %s: binned MI (stride=%d, bins=%d)", field_name, mi_lag_stride, mi_bins)
        mi = pointwise_binned_mi(
            series, valid, max_lag, n_bins=mi_bins, lag_stride=mi_lag_stride
        )
    else:
        log.info("  %s: Gaussian MI proxy from ρ", field_name)
        mi = gaussian_mi_from_rho(rho)

    log.info("  %s: power spectrum", field_name)
    dom_period, peak_freq, spec = pointwise_power_spectrum(series, valid, dt_days)

    return {
        "field": field_name,
        "rho": rho,
        "integral_timescale": tint,
        "decorrelation_lag_days": decorr,
        "mutual_information": mi,
        "dominant_period_days": dom_period,
        "peak_frequency_cpd": peak_freq,
        "power_spectrum": spec,
        "freq_cpd": np.fft.rfftfreq(series.shape[0], d=dt_days),
    }


def run_pointwise_window_analysis(
    series: dict[str, Any],
    out_dir: Path | str,
    *,
    fields: tuple[str, ...] = ("u", "v", "speed"),
    max_lag_days: int = 30,
    mi_mode: str = "gaussian",
    mi_bins: int = 16,
    mi_lag_stride: int = 3,
    save_arrays: bool = True,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    logger = setup_logging(out_dir)
    valid = series["valid"]
    dt_days = float(series["dt_days"])
    entry = series["catalog_entry"]
    window = int(series["window"])
    export_cfg = series["export_config"]
    period = f"{export_cfg.get('start_time')} → {export_cfg.get('end_time')}"
    title_suffix = f"W{window:02d} {entry['export_key']} | {period}"

    logger.info("Pointwise dynamics W%02d (%s)", window, entry["export_key"])
    logger.info("Output -> %s", out_dir)

    field_results: dict[str, Any] = {}
    all_plots: list[str] = []

    for fname in fields:
        stack = _field_stack(series["u"], series["v"], fname)
        res = analyze_field_pointwise(
            stack,
            valid,
            field_name=fname,
            max_lag_days=max_lag_days,
            dt_days=dt_days,
            mi_mode=mi_mode,
            mi_bins=mi_bins,
            mi_lag_stride=mi_lag_stride,
            logger=logger,
        )
        plots = save_pointwise_plots(
            out_dir,
            field=fname,
            rho=res["rho"],
            tint=res["integral_timescale"],
            decorr=res["decorrelation_lag_days"],
            mi=res["mutual_information"],
            dom_period=res["dominant_period_days"],
            valid=valid,
            dt_days=dt_days,
            title_suffix=title_suffix,
        )
        all_plots.extend(str(p) for p in plots)

        if save_arrays:
            arr_dir = out_dir / "arrays"
            arr_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                arr_dir / f"{fname}_pointwise.npz",
                rho=res["rho"],
                integral_timescale=res["integral_timescale"],
                decorr_lag_07=res["decorrelation_lag_days"][0.7],
                decorr_lag_05=res["decorrelation_lag_days"][0.5],
                decorr_lag_03=res["decorrelation_lag_days"][0.3],
                mutual_information=res["mutual_information"],
                dominant_period_days=res["dominant_period_days"],
                peak_frequency_cpd=res["peak_frequency_cpd"],
                valid=valid,
                freq_cpd=res["freq_cpd"],
            )

        maps_for_stats = {
            "integral_timescale": res["integral_timescale"],
            "dominant_period_days": res["dominant_period_days"],
            "decorrelation_lag_05": res["decorrelation_lag_days"][0.5],
            "autocorrelation": res["rho"],
            "mutual_information": res["mutual_information"],
        }
        field_results[fname] = {
            "statistics": summarize_valid_pixels(maps_for_stats, valid),
            "array_npz": str(out_dir / "arrays" / f"{fname}_pointwise.npz") if save_arrays else None,
        }

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "pointwise_per_grid_cell",
        "window": window,
        "export_key": entry["export_key"],
        "period": period,
        "n_times": int(series["u"].shape[0]),
        "dt_days": dt_days,
        "max_lag_days": min(max_lag_days, series["u"].shape[0] - 2),
        "mi_mode": mi_mode,
        "decorrelation_thresholds": list(DECORR_THRESHOLDS),
        "fields": field_results,
        "outputs": {
            "summary_json": str(out_dir / "pointwise_summary.json"),
            "plots": all_plots,
            "log": str(out_dir / "analysis.log"),
        },
    }
    (out_dir / "pointwise_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("Wrote %s", out_dir / "pointwise_summary.json")
    return summary


def analyze_selected_windows_pointwise(
    windows: list[int],
    *,
    data_prefix: Path | str | None = None,
    out_root: Path | str | None = None,
    fields: tuple[str, ...] = ("u", "v", "speed"),
    max_lag_days: int = 30,
    mi_mode: str = "gaussian",
    **kwargs: Any,
) -> list[dict[str, Any]]:
    prefix = Path(data_prefix or DEFAULT_DATA_PREFIX)
    root = Path(out_root or POINTWISE_OUT_ROOT)
    summaries: list[dict[str, Any]] = []
    for w in windows:
        series = load_pool12_window_series(prefix, w)
        out_dir = root / f"window_{w:02d}_{series['export_key']}"
        summaries.append(
            run_pointwise_window_analysis(
                series,
                out_dir,
                fields=fields,
                max_lag_days=max_lag_days,
                mi_mode=mi_mode,
                **kwargs,
            )
        )
    return summaries


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Pointwise lag dynamics per catalog window")
    p.add_argument("--windows", type=int, nargs="+", default=[7])
    p.add_argument("--data-prefix", type=Path, default=DEFAULT_DATA_PREFIX)
    p.add_argument("--out-root", type=Path, default=POINTWISE_OUT_ROOT)
    p.add_argument("--max-lag-days", type=int, default=30)
    p.add_argument("--fields", nargs="+", default=["u", "v", "speed"])
    p.add_argument("--mi-mode", choices=["gaussian", "binned"], default="gaussian")
    p.add_argument("--mi-bins", type=int, default=16)
    p.add_argument("--mi-lag-stride", type=int, default=3)
    args = p.parse_args()

    summaries = analyze_selected_windows_pointwise(
        args.windows,
        data_prefix=args.data_prefix,
        out_root=args.out_root,
        fields=tuple(args.fields),
        max_lag_days=args.max_lag_days,
        mi_mode=args.mi_mode,
        mi_bins=args.mi_bins,
        mi_lag_stride=args.mi_lag_stride,
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
