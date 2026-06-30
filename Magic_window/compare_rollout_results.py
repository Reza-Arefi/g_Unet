"""Aggregate rollout_results.json under Test&Rollout into comparison tables and plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_registry import TEST_ROLLOUT_DIR_NAME

HORIZONS = (15, 30, 45)
REGION_ORDER = ("center", "full")
METRIC_COLS = (
    "persistence_rmse",
    "model_rmse",
    "skill_score",
)

# Short labels for crowded x-axes (full export_key kept in tables).
DISPLAY_NAMES: dict[str, str] = {
    "magicwin_reference": "ref",
    "magicwin_high_land": "high_land",
    "magicwin_open_ocean": "open_ocean",
    "magicwin_bathy_east": "bathy_east",
    "magicwin_bathy_mid": "bathy_mid",
    "magicwin_bathy_northeast": "bathy_ne",
    "legacy_smallfixwin_0deg": "fixwin_0°",
    "legacy_smallfixwin_23deg": "fixwin_23°",
    "legacy_smallfixwin_45deg": "fixwin_45°",
    "legacy_smallfixwin_68deg": "fixwin_68°",
}

GROUP_ORDER = (
    "magicwin",
    "legacy_smallfixwin",
    "other",
)


def rollout_results_dir(mw_dir: Path | str) -> Path:
    return Path(mw_dir) / TEST_ROLLOUT_DIR_NAME


def comparison_output_dir(mw_dir: Path | str) -> Path:
    return rollout_results_dir(mw_dir) / "_comparison"


def _display_name(export_key: str) -> str:
    return DISPLAY_NAMES.get(export_key, export_key)


def _export_group(export_key: str) -> str:
    if export_key.startswith("magicwin"):
        return "magicwin"
    if export_key.startswith("legacy_smallfixwin"):
        return "legacy_smallfixwin"
    return "other"


def discover_rollout_results(mw_dir: Path | str) -> list[Path]:
    root = rollout_results_dir(mw_dir)
    paths = sorted(root.glob("*/rollout_results.json"))
    return [p for p in paths if p.parent.name != "_comparison"]


def load_rollout_results(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text())
        payload["_results_path"] = str(path)
        rows.append(payload)
    return rows


def metrics_to_long_df(results: list[dict[str, Any]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for res in results:
        export_key = res["export_key"]
        for region in REGION_ORDER:
            key = f"metrics_{region}"
            for row in res.get(key, []):
                records.append(
                    {
                        "export_key": export_key,
                        "display_name": _display_name(export_key),
                        "group": _export_group(export_key),
                        "model_run_name": res.get("model_run_name"),
                        "rollout_name": res.get("run_name"),
                        "region": region,
                        "horizon_days": int(row["horizon_days"]),
                        "persistence_rmse": float(row["persistence_rmse"]),
                        "model_rmse": float(row["model_rmse"]),
                        "skill_score": float(row["skill_score"]),
                        "run_name": res.get("run_name"),
                        "n_test": res.get("n_test"),
                        "best_val_loss": res.get("best_val_loss"),
                        "best_epoch": res.get("best_epoch"),
                        "evaluated_at": res.get("evaluated_at"),
                    }
                )
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    df["horizon_label"] = df["horizon_days"].map(lambda d: f"+{d}d")
    return df.sort_values(["group", "export_key", "region", "horizon_days"]).reset_index(drop=True)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per export_key with key metrics at +15d and +45d (full field)."""
    full = df[df["region"] == "full"]
    rows: list[dict[str, Any]] = []
    for export_key, g in full.groupby("export_key", sort=False):
        base = g.iloc[0]
        row: dict[str, Any] = {
            "export_key": export_key,
            "display_name": base["display_name"],
            "group": base["group"],
            "n_test": base["n_test"],
            "best_val_loss": base["best_val_loss"],
            "best_epoch": base["best_epoch"],
        }
        for days in HORIZONS:
            h = g[g["horizon_days"] == days]
            if h.empty:
                continue
            h = h.iloc[0]
            for metric in METRIC_COLS:
                row[f"{metric}_full_{days}d"] = h[metric]
        rows.append(row)
    out = pd.DataFrame(rows)
    sort_cols = [c for c in ("model_rmse_full_45d", "skill_score_full_45d") if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=[True, False])
    return out.reset_index(drop=True)


def _ordered_export_keys(df: pd.DataFrame) -> list[str]:
    order_map = {g: i for i, g in enumerate(GROUP_ORDER)}
    keys = df["export_key"].unique().tolist()
    return sorted(
        keys,
        key=lambda k: (
            order_map.get(_export_group(k), 99),
            k,
        ),
    )


def _palette(n: int) -> list[str]:
    cmap = plt.cm.tab20(np.linspace(0, 1, max(n, 1)))
    return [plt.matplotlib.colors.to_hex(c) for c in cmap]


def plot_grouped_rmse_bars(df: pd.DataFrame, out_path: Path) -> None:
    keys = _ordered_export_keys(df)
    labels = [_display_name(k) for k in keys]
    x = np.arange(len(keys))
    width = 0.12
    horizons = list(HORIZONS)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    for ax, region, title in zip(
        axes,
        REGION_ORDER,
        ("Center patch (4×4)", "Full valid field"),
    ):
        sub = df[df["region"] == region]
        for i, days in enumerate(horizons):
            offset = (i - 1) * width
            persist = [
                float(sub[(sub["export_key"] == k) & (sub["horizon_days"] == days)]["persistence_rmse"].iloc[0])
                for k in keys
            ]
            model = [
                float(sub[(sub["export_key"] == k) & (sub["horizon_days"] == days)]["model_rmse"].iloc[0])
                for k in keys
            ]
            ax.bar(x + offset - width / 2, persist, width * 0.95, label=f"Persist +{days}d" if ax is axes[0] else None, alpha=0.55, color="#ff7f0e")
            ax.bar(x + offset + width / 2, model, width * 0.95, label=f"Model +{days}d" if ax is axes[0] else None, color="#1f77b4")
        ax.set_xticks(x, labels, rotation=45, ha="right")
        ax.set_ylabel("Speed RMSE (|U,V|)")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].legend(ncol=3, fontsize=8, loc="upper left")
    fig.suptitle("Speed RMSE by export_key and horizon", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rmse_curves(df: pd.DataFrame, out_path: Path) -> None:
    keys = _ordered_export_keys(df)
    colors = _palette(len(keys))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    for ax, region, title in zip(axes, REGION_ORDER, ("Center 4×4", "Full field")):
        for color, export_key in zip(colors, keys):
            sub = df[(df["region"] == region) & (df["export_key"] == export_key)].sort_values("horizon_days")
            ax.plot(
                sub["horizon_days"],
                sub["model_rmse"],
                marker="o",
                label=_display_name(export_key),
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
        ax.set_xticks(HORIZONS, [f"+{d}d" for d in HORIZONS])
        ax.set_ylabel("Speed RMSE (|U,V|)")
        ax.set_xlabel("Horizon")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.suptitle("Model RMSE vs horizon (solid) · persistence (dashed)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_skill_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, 0.35 * df["export_key"].nunique() + 2)))
    for ax, region, title in zip(axes, REGION_ORDER, ("Center 4×4", "Full field")):
        sub = df[df["region"] == region]
        keys = _ordered_export_keys(sub)
        mat = np.array(
            [
                [
                    float(sub[(sub["export_key"] == k) & (sub["horizon_days"] == d)]["skill_score"].iloc[0])
                    for d in HORIZONS
                ]
                for k in keys
            ]
        )
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=-0.1, vmax=0.5)
        ax.set_xticks(range(len(HORIZONS)), [f"+{d}d" for d in HORIZONS])
        ax.set_yticks(range(len(keys)), [_display_name(k) for k in keys])
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
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, 0.35 * df["export_key"].nunique() + 2)))
    for ax, region, title in zip(axes, REGION_ORDER, ("Center 4×4", "Full field")):
        sub = df[df["region"] == region]
        keys = _ordered_export_keys(sub)
        mat = np.array(
            [
                [
                    float(sub[(sub["export_key"] == k) & (sub["horizon_days"] == d)]["model_rmse"].iloc[0])
                    for d in HORIZONS
                ]
                for k in keys
            ]
        )
        im = ax.imshow(mat, aspect="auto", cmap="viridis_r")
        ax.set_xticks(range(len(HORIZONS)), [f"+{d}d" for d in HORIZONS])
        ax.set_yticks(range(len(keys)), [_display_name(k) for k in keys])
        ax.set_title(f"Model RMSE — {title}")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=8, color="white")
        fig.colorbar(im, ax=ax, fraction=0.046, label="RMSE")
    fig.suptitle("adv-NO model speed RMSE", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_horizon_facets(df: pd.DataFrame, out_path: Path) -> None:
    """Per-horizon persistence vs model bars (full field), all export_keys."""
    keys = _ordered_export_keys(df)
    labels = [_display_name(k) for k in keys]
    sub = df[df["region"] == "full"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    x = np.arange(len(keys))
    width = 0.35
    for ax, days in zip(axes, HORIZONS):
        persist, model = [], []
        for k in keys:
            row = sub[(sub["export_key"] == k) & (sub["horizon_days"] == days)].iloc[0]
            persist.append(row["persistence_rmse"])
            model.append(row["model_rmse"])
        ax.bar(x - width / 2, persist, width, label="Persistence", color="#ff7f0e")
        ax.bar(x + width / 2, model, width, label="Model", color="#1f77b4")
        ax.set_title(f"+{days}d")
        ax.set_xticks(x, labels, rotation=45, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Speed RMSE (|U,V|) — full field")
    axes[0].legend()
    fig.suptitle("Persistence vs model by export_key (full field)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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
    display_cols = [
        c
        for c in summary_df.columns
        if c
        in {
            "export_key",
            "group",
            "n_test",
            "best_val_loss",
            "model_rmse_full_15d",
            "skill_score_full_15d",
            "model_rmse_full_30d",
            "skill_score_full_30d",
            "model_rmse_full_45d",
            "skill_score_full_45d",
        }
    ]
    md = _dataframe_to_markdown(summary_df[display_cols])
    md_path.write_text(
        "# Rollout comparison (full-field speed metrics)\n\n"
        f"{len(summary_df)} export keys · horizons +15 / +30 / +45 days\n\n"
        f"{md}\n"
    )
    paths["summary_md"] = md_path

    return paths


def run_comparison(mw_dir: Path | str | None = None) -> dict[str, Any]:
    mw_dir = Path(mw_dir or Path(__file__).resolve().parent)
    paths = discover_rollout_results(mw_dir)
    if not paths:
        raise FileNotFoundError(f"No rollout_results.json under {rollout_results_dir(mw_dir)}")

    results = load_rollout_results(paths)
    long_df = metrics_to_long_df(results)
    summary_df = summary_table(long_df)
    out_dir = comparison_output_dir(mw_dir)

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
        "export_keys": [r["export_key"] for r in results],
        "output_dir": str(out_dir),
        "tables": {k: str(v) for k, v in table_paths.items()},
        "plots": {k: str(v) for k, v in plot_paths.items()},
    }
    manifest_path = save_json_manifest(out_dir / "comparison_manifest.json", manifest)
    manifest["manifest"] = str(manifest_path)
    return manifest


def save_json_manifest(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2))
    return path


if __name__ == "__main__":
    info = run_comparison()
    print(f"Compared {info['n_runs']} runs → {info['output_dir']}")
    for label, path in {**info["tables"], **info["plots"]}.items():
        print(f"  {label}: {path}")
