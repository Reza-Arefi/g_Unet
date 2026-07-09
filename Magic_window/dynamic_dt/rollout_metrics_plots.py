"""Plot rollout speed metrics from rollout_results.json (per-run and cross-dt comparison)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SPEED_PATCH_SIZE = 4
REGION_KEYS = {
    "center": "metrics_center_4x4",
    "full": "metrics_full_field",
}
REGION_ORDER = ("center", "full")
REGION_TITLES = {
    "center": f"center {SPEED_PATCH_SIZE}×{SPEED_PATCH_SIZE}",
    "full": "full valid field",
}


def test_rollout_root(dt_dir: Path | str) -> Path:
    return Path(dt_dir) / "Test&Rollout"


def comparison_output_dir(dt_dir: Path | str) -> Path:
    return test_rollout_root(dt_dir) / "_comparison"


def discover_rollout_results(
    dt_dir: Path | str,
    *,
    dt_days_list: Sequence[int] | None = None,
) -> list[Path]:
    root = test_rollout_root(dt_dir)
    paths = sorted(root.glob("*/rollout_results.json"))
    paths = [p for p in paths if p.parent.name != "_comparison"]
    if dt_days_list is None:
        return paths
    allowed = {int(d) for d in dt_days_list}
    filtered: list[Path] = []
    for path in paths:
        payload = json.loads(path.read_text())
        if int(payload.get("dt_days", -1)) in allowed:
            filtered.append(path)
    return filtered


def load_rollout_results(paths: Sequence[Path | str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(Path(path).read_text())
        payload["_results_path"] = str(path)
        payload["_run_dir"] = str(Path(path).parent)
        if "run_name" not in payload:
            payload["run_name"] = Path(path).parent.name
        rows.append(payload)
    return rows


def _metric_series(metrics: list[dict[str, Any]]) -> tuple[list[float], list[float], list[float], list[int]]:
    horizons = [int(m["horizon_days"]) for m in metrics]
    persist = [float(m["persistence_rmse"]) for m in metrics]
    model = [float(m["model_rmse"]) for m in metrics]
    skill = [float(m["skill_score"]) for m in metrics]
    return persist, model, skill, horizons


def extract_metric_series(results: dict[str, Any]) -> dict[str, Any]:
    """Unpack center/full persistence, model RMSE, skill, and horizons from one JSON payload."""
    out: dict[str, Any] = {
        "run_name": results.get("run_name", ""),
        "dt_days": int(results["dt_days"]),
        "dt_hours": int(results.get("dt_hours", results["dt_days"] * 24)),
        "horizons_days": list(results.get("horizons_days", [])),
        "n_test": results.get("n_test"),
    }
    for region, key in REGION_KEYS.items():
        persist, model, skill, horizons = _metric_series(results[key])
        out[f"persist_{region}"] = persist
        out[f"model_{region}"] = model
        out[f"skill_{region}"] = skill
        out[f"horizons_{region}"] = horizons
    return out


def plot_speed_rmse_center_and_full(
    *,
    run_name: str,
    horizons: Sequence[int],
    persist_center: Sequence[float],
    model_center: Sequence[float],
    skill_center: Sequence[float],
    persist_full: Sequence[float],
    model_full: Sequence[float],
    skill_full: Sequence[float],
    out_path: Path | str,
) -> Path:
    labels = [f"+{d}d" for d in horizons]
    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, persist, model, skill, title in [
        (axes[0], persist_center, model_center, skill_center, REGION_TITLES["center"]),
        (axes[1], persist_full, model_full, skill_full, REGION_TITLES["full"]),
    ]:
        ax.bar(x - width / 2, persist, width, label="Persistence", color="#ff7f0e")
        ax.bar(x + width / 2, model, width, label="adv-NO model", color="#1f77b4")
        ax.set_xticks(x, labels)
        ax.set_ylabel("Speed RMSE (|U,V|)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        for i, (p, m, s) in enumerate(zip(persist, model, skill)):
            ax.text(i - width / 2, p, f"{p:.3f}", ha="center", va="bottom", fontsize=8)
            ax.text(i + width / 2, m, f"{m:.3f}", ha="center", va="bottom", fontsize=8)
            ax.text(i, max(p, m) * 1.02, f"skill={s:.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(f"Rollout speed error — {run_name}", y=1.02)
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_speed_rmse_by_horizon(
    *,
    run_name: str,
    horizons: Sequence[int],
    persist_center: Sequence[float],
    model_center: Sequence[float],
    skill_center: Sequence[float],
    persist_full: Sequence[float],
    model_full: Sequence[float],
    skill_full: Sequence[float],
    out_path: Path | str,
) -> Path:
    width = 0.55
    n_horizons = len(horizons)
    fig, axes = plt.subplots(2, n_horizons, figsize=(4.3 * n_horizons, 7), sharey="row")
    if n_horizons == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    region_rows = [
        (persist_center, model_center, skill_center, REGION_TITLES["center"]),
        (persist_full, model_full, skill_full, REGION_TITLES["full"]),
    ]
    for row, (persist, model, skill, row_title) in enumerate(region_rows):
        for col, (days, p, m, s) in enumerate(zip(horizons, persist, model, skill)):
            ax = axes[row, col]
            bars = ax.bar(["Persistence", "Model"], [p, m], color=["#ff7f0e", "#1f77b4"], width=width)
            ax.set_title(f"{row_title} | +{days}d")
            ax.set_ylabel("Speed RMSE (|U,V|)" if col == 0 else "")
            ax.grid(True, axis="y", alpha=0.3)
            for bar, val in zip(bars, [p, m]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val,
                    f"{val:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
            ax.text(0.5, 0.95, f"skill={s:.3f}", ha="center", va="top", fontsize=9, transform=ax.transAxes)

    fig.suptitle(f"Persistence vs model speed RMSE by horizon — {run_name}", y=1.02, fontsize=13)
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_speed_rmse_curves(
    *,
    run_name: str,
    horizons: Sequence[int],
    persist_center: Sequence[float],
    model_center: Sequence[float],
    persist_full: Sequence[float],
    model_full: Sequence[float],
    out_path: Path | str,
) -> Path:
    horizons_arr = np.array(horizons, dtype=float)
    xtick_labels = [f"+{int(d)}d" for d in horizons]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, persist, model, title in [
        (axes[0], persist_center, model_center, REGION_TITLES["center"]),
        (axes[1], persist_full, model_full, REGION_TITLES["full"]),
    ]:
        ax.plot(
            horizons_arr,
            persist,
            "-o",
            color="#ff7f0e",
            linewidth=2,
            markersize=9,
            markerfacecolor="#ff7f0e",
            markeredgecolor="white",
            markeredgewidth=1.2,
            label="Persistence",
        )
        ax.plot(
            horizons_arr,
            model,
            "-o",
            color="#1f77b4",
            linewidth=2,
            markersize=9,
            markerfacecolor="#1f77b4",
            markeredgecolor="white",
            markeredgewidth=1.2,
            label="adv-NO model",
        )
        for d, p, m in zip(horizons_arr, persist, model):
            ax.annotate(
                f"{p:.3f}",
                (d, p),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
                color="#cc6600",
            )
            ax.annotate(
                f"{m:.3f}",
                (d, m),
                textcoords="offset points",
                xytext=(0, -14),
                ha="center",
                fontsize=8,
                color="#1a5a9a",
            )
        ax.set_xticks(horizons_arr)
        ax.set_xticklabels(xtick_labels)
        ax.set_xlabel("Forecast horizon")
        ax.set_ylabel("Speed RMSE (|U,V|)")
        ax.set_title(title)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Speed RMSE curves — {run_name}", y=1.02, fontsize=13)
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_rollout_metrics_from_results(
    results: dict[str, Any],
    out_dir: Path | str | None = None,
) -> dict[str, Path]:
    """Save the three standard speed-RMSE figures for one rollout run."""
    out_dir = Path(out_dir or results.get("_run_dir") or ".")
    series = extract_metric_series(results)
    run_name = series["run_name"]
    horizons = series["horizons_days"]

    paths = {
        "center_and_full": plot_speed_rmse_center_and_full(
            run_name=run_name,
            horizons=horizons,
            persist_center=series["persist_center"],
            model_center=series["model_center"],
            skill_center=series["skill_center"],
            persist_full=series["persist_full"],
            model_full=series["model_full"],
            skill_full=series["skill_full"],
            out_path=out_dir / "speed_rmse_center_and_full.png",
        ),
        "by_horizon": plot_speed_rmse_by_horizon(
            run_name=run_name,
            horizons=horizons,
            persist_center=series["persist_center"],
            model_center=series["model_center"],
            skill_center=series["skill_center"],
            persist_full=series["persist_full"],
            model_full=series["model_full"],
            skill_full=series["skill_full"],
            out_path=out_dir / "speed_rmse_by_horizon_center_and_full.png",
        ),
        "curves": plot_speed_rmse_curves(
            run_name=run_name,
            horizons=horizons,
            persist_center=series["persist_center"],
            model_center=series["model_center"],
            persist_full=series["persist_full"],
            model_full=series["model_full"],
            out_path=out_dir / "speed_rmse_curves_center_and_full.png",
        ),
    }
    return paths


def plot_rollout_metrics_from_json(path: Path | str, out_dir: Path | str | None = None) -> dict[str, Path]:
    payload = json.loads(Path(path).read_text())
    payload["_results_path"] = str(path)
    payload["_run_dir"] = str(Path(path).parent)
    if "run_name" not in payload:
        payload["run_name"] = Path(path).parent.name
    return plot_rollout_metrics_from_results(payload, out_dir=out_dir or Path(path).parent)


def metrics_to_long_df(results: list[dict[str, Any]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for res in results:
        run_name = res.get("run_name", Path(res.get("_results_path", "")).parent.name)
        dt_days = int(res["dt_days"])
        for region, key in REGION_KEYS.items():
            for row in res.get(key, []):
                horizon_days = int(row["horizon_days"])
                horizon_step = horizon_days // dt_days if dt_days else None
                records.append(
                    {
                        "run_name": run_name,
                        "dt_days": dt_days,
                        "dt_hours": int(res.get("dt_hours", dt_days * 24)),
                        "region": region,
                        "horizon_days": horizon_days,
                        "horizon_step": horizon_step,
                        "horizon_label": f"+{horizon_days}d",
                        "step_label": f"+{horizon_step}dt" if horizon_step else f"+{horizon_days}d",
                        "persistence_rmse": float(row["persistence_rmse"]),
                        "model_rmse": float(row["model_rmse"]),
                        "skill_score": float(row["skill_score"]),
                        "n_test": res.get("n_test"),
                    }
                )
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    return df.sort_values(["dt_days", "region", "horizon_days"]).reset_index(drop=True)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per dt with full-field metrics at each rollout step."""
    full = df[df["region"] == "full"]
    rows: list[dict[str, Any]] = []
    for dt_days, g in full.groupby("dt_days", sort=True):
        base = g.iloc[0]
        row: dict[str, Any] = {
            "dt_days": int(dt_days),
            "dt_hours": int(base["dt_hours"]),
            "run_name": base["run_name"],
            "n_test": base["n_test"],
        }
        for _, hrow in g.sort_values("horizon_days").iterrows():
            step = int(hrow["horizon_step"])
            for metric in ("persistence_rmse", "model_rmse", "skill_score"):
                row[f"{metric}_full_step{step}"] = hrow[metric]
            row[f"horizon_days_step{step}"] = int(hrow["horizon_days"])
        rows.append(row)
    return pd.DataFrame(rows)


def _ordered_dt_days(df: pd.DataFrame) -> list[int]:
    return sorted(int(d) for d in df["dt_days"].unique())


def _palette(n: int) -> list[str]:
    cmap = plt.cm.tab10(np.linspace(0, 1, max(n, 1)))
    return [plt.matplotlib.colors.to_hex(c) for c in cmap]


def _dataframe_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                cells.append(format(val, floatfmt))
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_grouped_rmse_bars(df: pd.DataFrame, out_path: Path) -> None:
    """Grouped bars: one panel per region; x = dt_days, grouped by rollout step."""
    dt_values = _ordered_dt_days(df)
    labels = [f"dt={d}d" for d in dt_values]
    x = np.arange(len(dt_values))
    steps = sorted(int(s) for s in df["horizon_step"].dropna().unique())
    width = 0.12

    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, 0.35 * len(dt_values) + 2)), sharey=True)
    for ax, region, title in zip(axes, REGION_ORDER, ("Center patch (4×4)", "Full valid field")):
        sub = df[df["region"] == region]
        for i, step in enumerate(steps):
            offset = (i - (len(steps) - 1) / 2) * width
            persist, model = [], []
            for dt in dt_values:
                row = sub[(sub["dt_days"] == dt) & (sub["horizon_step"] == step)].iloc[0]
                persist.append(row["persistence_rmse"])
                model.append(row["model_rmse"])
            days = int(sub[(sub["dt_days"] == dt_values[0]) & (sub["horizon_step"] == step)].iloc[0]["horizon_days"])
            ax.bar(
                x + offset - width / 2,
                persist,
                width * 0.95,
                label=f"Persist +{days}d" if ax is axes[0] else None,
                alpha=0.55,
                color="#ff7f0e",
            )
            ax.bar(
                x + offset + width / 2,
                model,
                width * 0.95,
                label=f"Model +{days}d" if ax is axes[0] else None,
                color="#1f77b4",
            )
        ax.set_xticks(x, labels, rotation=45, ha="right")
        ax.set_ylabel("Speed RMSE (|U,V|)")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].legend(ncol=3, fontsize=8, loc="upper left")
    fig.suptitle("Speed RMSE by forecast step (dt) and horizon", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rmse_curves(df: pd.DataFrame, out_path: Path) -> None:
    """Model RMSE vs absolute horizon days; one line per dt."""
    dt_values = _ordered_dt_days(df)
    colors = _palette(len(dt_values))
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, 0.35 * len(dt_values) + 2)), sharex=False)
    for ax, region, title in zip(axes, REGION_ORDER, ("Center 4×4", "Full field")):
        for color, dt in zip(colors, dt_values):
            sub = df[(df["region"] == region) & (df["dt_days"] == dt)].sort_values("horizon_days")
            ax.plot(
                sub["horizon_days"],
                sub["model_rmse"],
                marker="o",
                label=f"dt={dt}d",
                color=color,
                linewidth=1.8,
            )
            ax.plot(
                sub["horizon_days"],
                sub["persistence_rmse"],
                marker="x",
                linestyle="--",
                color=color,
                alpha=0.45,
                linewidth=1.0,
            )
        ax.set_ylabel("Speed RMSE (|U,V|)")
        ax.set_xlabel("Horizon (days)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.suptitle("Model RMSE vs horizon (solid) · persistence (dashed)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_skill_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    dt_values = _ordered_dt_days(df)
    steps = sorted(int(s) for s in df["horizon_step"].dropna().unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, 0.35 * len(dt_values) + 2)))
    for ax, region, title in zip(axes, REGION_ORDER, ("Center 4×4", "Full field")):
        sub = df[df["region"] == region]
        mat = np.array(
            [
                [
                    float(sub[(sub["dt_days"] == dt) & (sub["horizon_step"] == step)]["skill_score"].iloc[0])
                    for step in steps
                ]
                for dt in dt_values
            ]
        )
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=-0.1, vmax=0.5)
        ax.set_xticks(range(len(steps)), [f"+{s}dt" for s in steps])
        ax.set_yticks(range(len(dt_values)), [f"dt={d}d" for d in dt_values])
        ax.set_title(f"Skill score — {title}")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8, color="black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Skill = 1 − model_RMSE / persistence_RMSE", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_model_rmse_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    dt_values = _ordered_dt_days(df)
    steps = sorted(int(s) for s in df["horizon_step"].dropna().unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, 0.35 * len(dt_values) + 2)))
    for ax, region, title in zip(axes, REGION_ORDER, ("Center 4×4", "Full field")):
        sub = df[df["region"] == region]
        mat = np.array(
            [
                [
                    float(sub[(sub["dt_days"] == dt) & (sub["horizon_step"] == step)]["model_rmse"].iloc[0])
                    for step in steps
                ]
                for dt in dt_values
            ]
        )
        im = ax.imshow(mat, aspect="auto", cmap="viridis_r")
        ax.set_xticks(range(len(steps)), [f"+{s}dt" for s in steps])
        ax.set_yticks(range(len(dt_values)), [f"dt={d}d" for d in dt_values])
        ax.set_title(f"Model RMSE — {title}")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=8, color="white")
        fig.colorbar(im, ax=ax, fraction=0.046, label="RMSE")
    fig.suptitle("adv-NO model speed RMSE by dt and rollout step", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_horizon_facets(df: pd.DataFrame, out_path: Path) -> None:
    """Per rollout-step persistence vs model bars (full field), all dt values."""
    dt_values = _ordered_dt_days(df)
    labels = [f"dt={d}d" for d in dt_values]
    sub = df[df["region"] == "full"]
    steps = sorted(int(s) for s in sub["horizon_step"].dropna().unique())
    fig, axes = plt.subplots(1, len(steps), figsize=(5 * len(steps), max(4.5, 0.35 * len(dt_values) + 2)), sharey=True)
    if len(steps) == 1:
        axes = [axes]
    x = np.arange(len(dt_values))
    width = 0.35
    for ax, step in zip(axes, steps):
        persist, model = [], []
        days = int(sub[sub["horizon_step"] == step].iloc[0]["horizon_days"])
        for dt in dt_values:
            row = sub[(sub["dt_days"] == dt) & (sub["horizon_step"] == step)].iloc[0]
            persist.append(row["persistence_rmse"])
            model.append(row["model_rmse"])
        ax.bar(x - width / 2, persist, width, label="Persistence", color="#ff7f0e")
        ax.bar(x + width / 2, model, width, label="Model", color="#1f77b4")
        ax.set_title(f"+{days}d (+{step}dt)")
        ax.set_xticks(x, labels, rotation=45, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Speed RMSE (|U,V|) — full field")
    axes[0].legend()
    fig.suptitle("Persistence vs model by forecast step (full field)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_tables(long_df: pd.DataFrame, summary_df: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    long_path = out_dir / "rollout_metrics_long.csv"
    long_df.to_csv(long_path, index=False)
    paths["long_csv"] = long_path

    summary_path = out_dir / "rollout_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    paths["summary_csv"] = summary_path

    md_path = out_dir / "rollout_summary.md"
    md = _dataframe_to_markdown(summary_df)
    md_path.write_text(
        "# dynamic_dt rollout comparison (full-field speed metrics)\n\n"
        f"{len(summary_df)} dt values · horizons +1dt / +2dt / +3dt\n\n"
        f"{md}\n"
    )
    paths["summary_md"] = md_path
    return paths


def run_comparison(
    dt_dir: Path | str,
    *,
    dt_days_list: Sequence[int] | None = None,
) -> dict[str, Any]:
    paths = discover_rollout_results(dt_dir, dt_days_list=dt_days_list)
    if not paths:
        raise FileNotFoundError(f"No rollout_results.json under {test_rollout_root(dt_dir)}")

    results = load_rollout_results(paths)
    long_df = metrics_to_long_df(results)
    summary_df = summary_table(long_df)
    out_dir = comparison_output_dir(dt_dir)

    table_paths = save_tables(long_df, summary_df, out_dir)
    plot_paths = {
        "grouped_rmse_bars": out_dir / "compare_grouped_rmse_bars.png",
        "rmse_curves": out_dir / "compare_rmse_curves.png",
        "skill_heatmap": out_dir / "compare_skill_heatmap.png",
        "model_rmse_heatmap": out_dir / "compare_model_rmse_heatmap.png",
        "horizon_facets": out_dir / "compare_horizon_facets_full.png",
    }
    plot_grouped_rmse_bars(long_df, plot_paths["grouped_rmse_bars"])
    plot_rmse_curves(long_df, plot_paths["rmse_curves"])
    plot_skill_heatmap(long_df, plot_paths["skill_heatmap"])
    plot_model_rmse_heatmap(long_df, plot_paths["model_rmse_heatmap"])
    plot_horizon_facets(long_df, plot_paths["horizon_facets"])

    manifest = {
        "n_runs": len(results),
        "dt_days": [int(r["dt_days"]) for r in results],
        "run_names": [r.get("run_name", Path(r["_results_path"]).parent.name) for r in results],
        "output_dir": str(out_dir),
        "tables": {k: str(v) for k, v in table_paths.items()},
        "plots": {k: str(v) for k, v in plot_paths.items()},
    }
    manifest_path = out_dir / "comparison_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest["manifest"] = str(manifest_path)
    return manifest


def plot_all_rollout_results(
    dt_dir: Path | str,
    *,
    dt_days_list: Sequence[int] | None = None,
) -> list[dict[str, Path]]:
    """Generate per-run metric plots for every rollout_results.json."""
    plotted: list[dict[str, Path]] = []
    for path in discover_rollout_results(dt_dir, dt_days_list=dt_days_list):
        plotted.append(plot_rollout_metrics_from_json(path))
    return plotted
