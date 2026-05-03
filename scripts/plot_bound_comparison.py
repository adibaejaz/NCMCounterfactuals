#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_bound_results import (
    _filter_rows,
    _format_float,
    _mean,
    _std,
    _trial_sort_key,
    load_bound_results,
)


def _load_labeled_rows(inputs, include_nonstandard=False, graph_names=None):
    rows = []
    for label, root in inputs:
        for row in _filter_rows(
                load_bound_results(root),
                include_nonstandard=include_nonstandard,
                graph_names=graph_names):
            cur = dict(row)
            cur["comparison_label"] = label
            cur["comparison_root"] = root
            rows.append(cur)
    return rows


def _group_key(row):
    return (
        row["graph"],
        row["trial_index"],
        row["comparison_label"],
        row["cycle_lambda"],
        row["mask_mode"],
        row["max_lambda"],
        row["min_lambda"],
        row["theta_lr"],
        row["mask_lr"],
        row["theta_steps_per_mask"],
        row["mask_steps_per_theta"],
    )


def _group_label(group):
    graph, trial, label, cycle_lambda, mask_mode, max_lambda, min_lambda, theta_lr, mask_lr, theta_steps, mask_steps = group
    return "\n".join([
        str(label),
        str(mask_mode),
        f"c={_format_float(cycle_lambda)}",
        f"tlr={_format_float(theta_lr)} mlr={_format_float(mask_lr)}",
        f"s={theta_steps}:{mask_steps}",
    ])


def _sort_group(group):
    return (
        str(group[0]),
        _trial_sort_key(group[1]),
        str(group[2]),
        float("inf") if group[3] is None else group[3],
        str(group[4]),
        float("inf") if group[5] is None else group[5],
        float("inf") if group[6] is None else group[6],
    )


def _row_offsets(n, span=0.58):
    if n <= 1:
        return [0.0] * n
    step = span / (n - 1)
    return [-span / 2 + i * step for i in range(n)]


def _trial_bands(ax, xs, groups):
    bands = {}
    for x, group in zip(xs, groups):
        bands.setdefault((group[0], group[1]), []).append(x)
    for (graph, trial), cur_xs in bands.items():
        start = min(cur_xs) - 0.5
        end = max(cur_xs) + 0.5
        if str(trial).isdigit() and int(trial) % 2 == 0:
            ax.axvspan(start, end, color="0.97", zorder=-1)
        ax.text(
            (start + end) / 2,
            -0.78,
            f"{graph} trial {trial}",
            ha="center",
            va="top",
            fontsize=9,
            transform=ax.get_xaxis_transform(),
            clip_on=False,
        )


def plot_bounds(rows, output, title=None):
    grouped = {}
    for row in rows:
        grouped.setdefault(_group_key(row), []).append(row)
    groups = sorted(grouped, key=_sort_group)
    if not groups:
        raise ValueError("no rows to plot")

    spacing = 1.25
    xs = [i * spacing for i in range(len(groups))]
    fig, ax = plt.subplots(figsize=(max(14, 0.9 * len(groups)), 8))
    true_color = "#1f77b4"
    min_color = "#d62728"
    max_color = "#ff7f0e"

    for x, group in zip(xs, groups):
        group_rows = sorted(grouped[group], key=lambda row: (int(row["train_seed_offset"]), row["run_id"]))
        true_lower = _mean(row["true_lower"] for row in group_rows)
        true_upper = _mean(row["true_upper"] for row in group_rows)
        ax.hlines([true_lower, true_upper], x - 0.32, x + 0.32, color=true_color, linewidth=4, zorder=3)
        for offset, row in zip(_row_offsets(len(group_rows)), group_rows):
            ax.scatter(x + offset, row["ncm_min"], color=min_color, edgecolors="black", linewidths=0.3, s=34, zorder=4)
            ax.scatter(x + offset, row["ncm_max"], color=max_color, edgecolors="black", linewidths=0.3, s=34, zorder=4)

    _trial_bands(ax, xs, groups)
    ax.set_xticks(xs)
    ax.set_xticklabels([_group_label(group) for group in groups], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Query value")
    ax.set_title(title or "Bound Comparison")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(handles=[
        Line2D([0], [0], color=true_color, lw=4, label="True min/max"),
        Line2D([0], [0], marker="o", color="black", markerfacecolor=min_color, linestyle="", label="NCM min"),
        Line2D([0], [0], marker="o", color="black", markerfacecolor=max_color, linestyle="", label="NCM max"),
    ], loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.subplots_adjust(bottom=0.42, right=0.86)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_summary(rows, output, title=None):
    grouped = {}
    for row in rows:
        if not all(k in row for k in ("min_total_true_KL", "max_total_true_KL", "err_lower", "err_upper")):
            continue
        grouped.setdefault(_group_key(row), []).append(row)
    groups = sorted(grouped, key=_sort_group)
    if not groups:
        return

    metrics = [
        ("kl", "Avg True KL", lambda rs: [(r["min_total_true_KL"] + r["max_total_true_KL"]) / 2 for r in rs]),
        ("err", "Avg Bound Error", lambda rs: [(abs(r["err_lower"]) + abs(r["err_upper"])) / 2 for r in rs]),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(max(14, 0.9 * len(groups)), 9), sharex=True)
    xs = list(range(len(groups)))
    for ax, (_name, ylabel, values_fn) in zip(axes, metrics):
        means = [_mean(values_fn(grouped[group])) for group in groups]
        stds = [_std(values_fn(grouped[group])) for group in groups]
        ax.bar(xs, means, yerr=stds, color="#4c78a8", alpha=0.82, capsize=4, ecolor="black")
        if ylabel.endswith("KL") and all(value > 0 for value in means):
            ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
    axes[-1].set_xticks(xs)
    axes[-1].set_xticklabels([_group_label(group) for group in groups], rotation=45, ha="right", fontsize=8)
    axes[0].set_title(title or "Aggregate Comparison")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot multiple bound result roots side by side.")
    parser.add_argument("--input", action="append", nargs=2, metavar=("LABEL", "ROOT"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--graph", action="append", dest="graph_names")
    parser.add_argument("--include-nonstandard", action="store_true")
    parser.add_argument("--title")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_labeled_rows(args.input, include_nonstandard=args.include_nonstandard, graph_names=args.graph_names)
    plot_bounds(rows, output_dir / "bound_results_side_by_side.png", title=args.title)
    plot_summary(rows, output_dir / "bound_summary_side_by_side.png", title=args.title)
    print(f"Saved comparison plots to {output_dir}")


if __name__ == "__main__":
    main()
