#!/usr/bin/env python3
"""Visualize tuning runs from an experiment output tree.

This script exports completed runs to CSV and renders a compact summary figure
for comparing baseline, NOTEARS, and DAGMA settings.

Example:
    python scripts/plot_tuning_results.py \
        --root out/chain_tuning \
        --csv out/chain_tuning_summary.csv \
        --fig out/chain_tuning_summary.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns

from export_tuning_csv import export_runs


FIT_METRICS = ["total_true_KL", "total_dat_KL", "dag_h"]
METHOD_COLORS = {
    "baseline": "#222222",
    "notears": "#d55e00",
    "dagma": "#0072b2",
}
TEXTURE_MARKERS = ["", "+", "x", "1", "2", "3", "4"]
LR_MARKERS = {
    0.001: "o",
    0.004: "X",
}


def _method_name(row: pd.Series) -> str:
    if row["family"] == "baseline":
        return "baseline"
    return str(row.get("penalty", "masked"))


def _method_label(row: pd.Series) -> str:
    lr = row.get("lr")
    if row["family"] == "baseline":
        return f"baseline\nlr={lr:g}"
    cycle_lambda = row.get("cycle_lambda")
    return f"lr={lr:g}, c={cycle_lambda:g}"


def _load_summary(root: Path, csv_path: Path, threshold: float, decimals: int) -> pd.DataFrame:
    export_runs(root, csv_path, threshold, decimals)
    df = pd.read_csv(csv_path)
    for col in FIT_METRICS + ["lr", "cycle_lambda", "dagma_s", "mask_init_value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["method"] = df.apply(_method_name, axis=1)
    df["label"] = df.apply(_method_label, axis=1)
    return df


def _aggregate_for_bars(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    grouped = (
        df.groupby(["label", "method"], as_index=False)
        .agg(mean=(metric, "mean"), std=(metric, "std"))
        .sort_values("mean", kind="stable")
        .reset_index(drop=True)
    )
    grouped["std"] = grouped["std"].fillna(0.0)
    return grouped


def _aggregate_for_scatter(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["family", "method", "lr", "cycle_lambda"], dropna=False, as_index=False)
        .agg(
            dag_h_mean=("dag_h", "mean"),
            dag_h_std=("dag_h", "std"),
            total_true_KL_mean=("total_true_KL", "mean"),
            total_true_KL_std=("total_true_KL", "std"),
        )
    )
    grouped[["dag_h_std", "total_true_KL_std"]] = grouped[["dag_h_std", "total_true_KL_std"]].fillna(0.0)
    return grouped


def _add_bar_color_legend(ax) -> None:
    handles = []
    for method in ["baseline", "notears", "dagma"]:
        color = METHOD_COLORS.get(method)
        handles.append(
            Line2D([0], [0], marker="s", color=color, markerfacecolor=color, linestyle="", markersize=10, label=method)
        )
    ax.legend(handles=handles, title="bar color = method", loc="upper right", fontsize=9, title_fontsize=10)


def _plot_metric_bars(ax, df: pd.DataFrame, metric: str, title: str, log_scale: bool = False, add_color_legend: bool = False) -> None:
    grouped = _aggregate_for_bars(df, metric)
    x = list(range(len(grouped)))
    colors = [METHOD_COLORS.get(method, "#666666") for method in grouped["method"]]
    ax.bar(x, grouped["mean"], yerr=grouped["std"], color=colors, capsize=4, ecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["label"])
    ax.set_title(title, fontsize=18)
    ax.set_xlabel("")
    ax.set_ylabel(metric, fontsize=14)
    ax.tick_params(axis="x", labelrotation=32, labelsize=11)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.tick_params(axis="y", labelsize=12)
    ax.margins(x=0.03)
    if log_scale:
        ax.set_yscale("log")
    if add_color_legend:
        _add_bar_color_legend(ax)


def _texture_map(df: pd.DataFrame) -> dict[float, str]:
    lambdas = sorted(df["cycle_lambda"].dropna().unique().tolist())
    return {
        cycle_lambda: TEXTURE_MARKERS[min(i, len(TEXTURE_MARKERS) - 1)]
        for i, cycle_lambda in enumerate(lambdas)
    }


def _add_texture_overlays(ax, df: pd.DataFrame, mapping: dict[float, str]) -> None:
    for cycle_lambda, marker in mapping.items():
        if marker == "":
            continue
        subset = df[df["cycle_lambda"] == cycle_lambda]
        if subset.empty:
            continue
        ax.scatter(
            subset["dag_h_mean"],
            subset["total_true_KL_mean"],
            marker=marker,
            s=140,
            c="black",
            linewidths=1.2,
            alpha=0.9,
            zorder=6,
        )


def _add_texture_legend(ax, mapping: dict[float, str]) -> None:
    handles = []
    for cycle_lambda, marker in mapping.items():
        label = f"c={cycle_lambda:g}"
        display_marker = "o" if marker == "" else marker
        handles.append(
            Line2D([0], [0], marker=display_marker, color="black", linestyle="", markersize=8, label=label)
        )
    if handles:
        texture_legend = ax.legend(handles=handles, title="c texture", loc="upper right", fontsize=9, title_fontsize=10)
        ax.add_artist(texture_legend)


def _add_lr_legend(ax, df: pd.DataFrame) -> None:
    lrs = sorted(df["lr"].dropna().unique().tolist())
    handles = []
    for lr in lrs:
        marker = LR_MARKERS.get(float(lr), "o")
        handles.append(
            Line2D([0], [0], marker=marker, color="gray", markerfacecolor="gray", linestyle="", markersize=8, label=f"lr={lr:g}")
        )
    if handles:
        lr_legend = ax.legend(handles=handles, title="marker = lr", loc="lower right", fontsize=9, title_fontsize=10)
        ax.add_artist(lr_legend)



def _plot_fit_tradeoff(ax, df: pd.DataFrame, title: str, method: str) -> None:
    masked = df[(df["family"] == "masked") & (df["method"] == method)].copy()
    masked = _aggregate_for_scatter(masked)

    if not masked.empty:
        texture_mapping = _texture_map(masked)
        sns.scatterplot(
            data=masked,
            x="dag_h_mean",
            y="total_true_KL_mean",
            hue="method",
            style="lr",
            markers=LR_MARKERS,
            palette=METHOD_COLORS,
            s=180,
            ax=ax,
        )
        _add_texture_overlays(ax, masked, texture_mapping)
        _add_texture_legend(ax, texture_mapping)
        _add_lr_legend(ax, masked)

    ax.set_title(title, fontsize=18)
    ax.set_xlabel("dag_h", fontsize=14)
    ax.set_ylabel("total_true_KL", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()



def make_figure(df: pd.DataFrame, fig_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk")

    fig, axes = plt.subplots(2, 3, figsize=(28, 14))

    _plot_fit_tradeoff(axes[0, 0], df, "DAGMA Fit vs Acyclicity", method="dagma")
    _plot_fit_tradeoff(axes[1, 0], df, "NOTEARS Fit vs Acyclicity", method="notears")

    _plot_metric_bars(axes[0, 1], df, "total_dat_KL", "Data Fit", log_scale=True, add_color_legend=True)
    _plot_metric_bars(axes[1, 1], df, "total_true_KL", "True Fit", log_scale=True)

    notears_df = df[df["method"] == "notears"].copy()
    dagma_df = df[df["method"] == "dagma"].copy()

    if not dagma_df.empty:
        _plot_metric_bars(axes[0, 2], dagma_df, "dag_h", "DAGMA Acyclicity")
    else:
        axes[0, 2].set_visible(False)

    if not notears_df.empty:
        _plot_metric_bars(axes[1, 2], notears_df, "dag_h", "NOTEARS Acyclicity")
    else:
        axes[1, 2].set_visible(False)

    fig.suptitle("Chain Tuning Summary", y=0.99, fontsize=24)
    fig.subplots_adjust(hspace=0.5, wspace=0.3, bottom=0.24)
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot tuning experiment results")
    parser.add_argument("--root", default="out/chain_tuning", help="root experiment directory")
    parser.add_argument("--csv", default="out/chain_tuning_summary.csv", help="output CSV path")
    parser.add_argument("--fig", default="out/chain_tuning_summary.png", help="output figure path")
    parser.add_argument("--hard-threshold", type=float, default=0.5,
                        help="threshold used for hard-graph summaries in the CSV")
    parser.add_argument("--decimals", type=int, default=4,
                        help="decimal places for mask summary strings in the CSV")
    args = parser.parse_args()

    root = Path(args.root)
    csv_path = Path(args.csv)
    fig_path = Path(args.fig)
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    df = _load_summary(root, csv_path, args.hard_threshold, args.decimals)
    if df.empty:
        raise SystemExit(f"No completed runs found under {root}")

    make_figure(df, fig_path)
    print(f"Wrote {len(df)} runs to {csv_path}")
    print(f"Saved figure to {fig_path}")


if __name__ == "__main__":
    main()
