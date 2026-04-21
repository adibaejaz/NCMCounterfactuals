import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _find_results_files(root_dir):
    return sorted(Path(root_dir).rglob("results.json"))


def _extract_run_label(results_path, root_dir):
    run_dir = _extract_run_dir(results_path, root_dir)

    return f"{_extract_run_id(run_dir)}:t{_extract_trial_index(run_dir)}"


def _extract_run_dir(results_path, root_dir):
    rel = results_path.relative_to(root_dir)
    parts = rel.parts[:-1]

    if len(parts) >= 2 and parts[-1].isdigit():
        return parts[-2]
    if parts:
        return parts[-1]
    return results_path.stem


def _extract_run_id(run_dir):
    for piece in run_dir.split("-"):
        if piece.startswith("run="):
            return piece[len("run="):]
    return run_dir


def _extract_trial_index(run_dir):
    for piece in run_dir.split("-"):
        if piece.startswith("trial_index="):
            return piece[len("trial_index="):]
    return "?"


def _load_hyperparams(results_path):
    run_path = results_path.parent
    if run_path.name.isdigit():
        run_path = run_path.parent

    hyperparams_path = run_path / "hyperparams.json"
    if not hyperparams_path.is_file():
        return {}

    with open(hyperparams_path) as f:
        return json.load(f)


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _as_float(value, default=None):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _standard_bound_run(hyperparams):
    return not (
        _as_bool(hyperparams.get("alt-opt"))
        or _as_bool(hyperparams.get("dag-alm"))
    )


def _format_float(value):
    if value is None:
        return "?"
    return f"{value:g}"


def _group_label(group):
    cycle_lambda, mask_mode, _trial_index = group
    return "\n".join([
        f"l={_format_float(cycle_lambda)}",
        str(mask_mode),
    ])


def _trial_sort_key(trial_index):
    return int(trial_index) if str(trial_index).isdigit() else str(trial_index)


def _row_offsets(rows, max_span=0.62):
    if len(rows) <= 1:
        return [0.0] * len(rows)

    step = max_span / (len(rows) - 1)
    return [-max_span / 2 + i * step for i in range(len(rows))]


def _standard_rows(rows):
    return [
        row for row in rows
        if row.get("standard_bound_run", True)
    ]


def _group_rows(rows):
    grouped = {}
    for row in rows:
        group = (row["cycle_lambda"], row["mask_mode"], row["trial_index"])
        grouped.setdefault(group, []).append(row)
    return grouped


def _sort_groups(groups):
    return sorted(
        groups,
        key=lambda group: (
            _trial_sort_key(group[2]),
            str(group[1]),
            float("inf") if group[0] is None else group[0],
        ),
    )


def _sorted_group_rows(group_rows):
    return sorted(
        group_rows,
        key=lambda row: (
            _as_float(row.get("train_seed_offset"), default=float("inf")),
            row["run_id"],
        ),
    )


def _add_trial_tiers(ax, xs, groups):
    trial_groups = {}
    for x, group in zip(xs, groups):
        trial_groups.setdefault(group[2], []).append(x)

    for trial_index, trial_xs in trial_groups.items():
        start = min(trial_xs) - 0.5
        end = max(trial_xs) + 0.5
        if str(trial_index).isdigit() and int(trial_index) % 2 == 0:
            ax.axvspan(start, end, color="0.98", zorder=-1)
        if start > -0.5:
            ax.axvline(start, color="0.75", linewidth=0.8, zorder=1)

    return trial_groups


def _add_trial_labels(ax, trial_groups):
    for trial_index, trial_xs in trial_groups.items():
        center = (min(trial_xs) + max(trial_xs)) / 2
        ax.text(
            center,
            -0.55,
            f"trial {trial_index}",
            ha="center",
            va="top",
            fontsize=9,
            transform=ax.get_xaxis_transform(),
            clip_on=False,
        )


def _extract_bound_series(results):
    true_lower = None
    true_upper = None
    ncm_min = None
    ncm_max = None

    for key, value in results.items():
        if key.startswith("true_lower_"):
            true_lower = value
        elif key.startswith("true_upper_"):
            true_upper = value
        elif key.startswith("min_ncm_"):
            ncm_min = value
        elif key.startswith("max_ncm_"):
            ncm_max = value

    if None in (true_lower, true_upper, ncm_min, ncm_max):
        return None

    return {
        "true_lower": float(true_lower),
        "true_upper": float(true_upper),
        "ncm_min": float(ncm_min),
        "ncm_max": float(ncm_max),
    }


def _extract_bound_error_series(results):
    err_lower = None
    err_upper = None

    for key, value in results.items():
        if key.startswith("err_min_ncm_") and key.endswith("_lower_bound"):
            err_lower = value
        elif key.startswith("err_max_ncm_") and key.endswith("_upper_bound"):
            err_upper = value

    if None in (err_lower, err_upper):
        return None

    err_lower = float(err_lower)
    err_upper = float(err_upper)
    return {
        "err_lower": err_lower,
        "err_upper": err_upper,
        "avg_abs_bound_error": (abs(err_lower) + abs(err_upper)) / 2,
        "max_abs_bound_error": max(abs(err_lower), abs(err_upper)),
    }


def _extract_kl_series(results):
    keys = [
        "min_total_true_KL",
        "max_total_true_KL",
        "min_total_dat_KL",
        "max_total_dat_KL",
    ]
    if any(key not in results for key in keys):
        return None

    return {key: float(results[key]) for key in keys}


def load_bound_results(root_dir):
    root = Path(root_dir)
    rows = []

    for results_path in _find_results_files(root):
        with open(results_path) as f:
            results = json.load(f)

        series = _extract_bound_series(results)
        if series is None:
            continue

        hyperparams = _load_hyperparams(results_path)
        run_dir = _extract_run_dir(results_path, root)
        kl_series = _extract_kl_series(results) or {}
        bound_error_series = _extract_bound_error_series(results) or {}
        rows.append({
            "label": _extract_run_label(results_path, root),
            "run_id": _extract_run_id(run_dir),
            "trial_index": _extract_trial_index(run_dir),
            "cycle_lambda": _as_float(hyperparams.get("cycle-lambda")),
            "mask_mode": hyperparams.get("mask-mode", "?"),
            "train_seed_offset": hyperparams.get("train-seed-offset", "0"),
            "cycle_penalty": hyperparams.get("cycle-penalty"),
            "alt_opt": _as_bool(hyperparams.get("alt-opt")),
            "dag_alm": _as_bool(hyperparams.get("dag-alm")),
            "standard_bound_run": _standard_bound_run(hyperparams),
            "results_path": str(results_path),
            **series,
            **kl_series,
            **bound_error_series,
        })

    rows.sort(key=lambda row: row["label"])
    return rows


def plot_bound_results(root_dir, output_path=None, title=None, figsize=None):
    rows = _standard_rows(load_bound_results(root_dir))
    if not rows:
        raise ValueError(f"No completed bound results found under {root_dir}")

    grouped = _group_rows(rows)
    groups = _sort_groups(grouped)
    n_runs = len(groups)
    if figsize is None:
        figsize = (max(12, 0.75 * n_runs), 6)

    fig, ax = plt.subplots(figsize=figsize)

    true_color = "#1f77b4"
    ncm_min_color = "#d62728"
    ncm_max_color = "#ff7f0e"
    half_width = 0.28
    xs = list(range(n_runs))

    for x in xs:
        if x % 2 == 0:
            ax.axvspan(x - 0.5, x + 0.5, color="0.96", zorder=0)

    trial_groups = _add_trial_tiers(ax, xs, groups)

    for x, group in zip(xs, groups):
        group_rows = _sorted_group_rows(grouped[group])
        true_lower = sum(row["true_lower"] for row in group_rows) / len(group_rows)
        true_upper = sum(row["true_upper"] for row in group_rows) / len(group_rows)

        ax.hlines(
            [true_lower, true_upper],
            x - half_width,
            x + half_width,
            color=true_color,
            linewidth=4.0,
            zorder=3,
        )

        offsets = _row_offsets(group_rows)

        for offset, row in zip(offsets, group_rows):
            ax.scatter(
                [x + offset],
                [row["ncm_min"]],
                color=ncm_min_color,
                marker="o",
                s=34,
                edgecolors="black",
                linewidths=0.3,
                zorder=4,
            )
            ax.scatter(
                [x + offset],
                [row["ncm_max"]],
                color=ncm_max_color,
                marker="o",
                s=34,
                edgecolors="black",
                linewidths=0.3,
                zorder=4,
            )

    ax.set_xlim(-0.8, n_runs - 0.2)
    ax.set_xticks(xs)
    ax.set_xticklabels([_group_label(group) for group in groups], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Cycle lambda / mask mode, grouped by trial", labelpad=72)
    ax.set_ylabel("Query value")
    ax.set_title(title or f"True Bounds and NCM Seed Results ({Path(root_dir)})")
    ax.grid(axis="y", alpha=0.3, zorder=1)

    _add_trial_labels(ax, trial_groups)

    ax.legend(
        handles=[
            Line2D([0], [0], color=true_color, lw=4.0, label="True min/max"),
            Line2D([0], [0], marker="o", color="black", markerfacecolor=ncm_min_color,
                   linestyle="", markersize=7, label="NCM min"),
            Line2D([0], [0], marker="o", color="black", markerfacecolor=ncm_max_color,
                   linestyle="", markersize=7, label="NCM max"),
        ],
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )

    fig.tight_layout(rect=(0, 0.22, 0.85, 1))

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_kl_results(root_dir, output_path=None, title=None, figsize=None):
    rows = [
        row for row in _standard_rows(load_bound_results(root_dir))
        if all(
            key in row
            for key in (
                "min_total_true_KL",
                "max_total_true_KL",
                "min_total_dat_KL",
                "max_total_dat_KL",
            )
        )
    ]
    if not rows:
        raise ValueError(f"No completed bound KL results found under {root_dir}")

    grouped = _group_rows(rows)
    groups = _sort_groups(grouped)
    n_runs = len(groups)
    if figsize is None:
        figsize = (max(12, 0.75 * n_runs), 10)

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    min_color = "#d62728"
    max_color = "#ff7f0e"
    xs = list(range(n_runs))

    metric_specs = [
        (axes[0], "min_total_true_KL", "max_total_true_KL", "True KL"),
        (axes[1], "min_total_dat_KL", "max_total_dat_KL", "Data KL"),
    ]

    for ax, min_key, max_key, ylabel in metric_specs:
        for x in xs:
            if x % 2 == 0:
                ax.axvspan(x - 0.5, x + 0.5, color="0.96", zorder=0)

        trial_groups = _add_trial_tiers(ax, xs, groups)

        for x, group in zip(xs, groups):
            group_rows = _sorted_group_rows(grouped[group])
            offsets = _row_offsets(group_rows)

            for offset, row in zip(offsets, group_rows):
                ax.scatter(
                    [x + offset],
                    [row[min_key]],
                    color=min_color,
                    marker="o",
                    s=34,
                    edgecolors="black",
                    linewidths=0.3,
                    zorder=4,
                )
                ax.scatter(
                    [x + offset],
                    [row[max_key]],
                    color=max_color,
                    marker="o",
                    s=34,
                    edgecolors="black",
                    linewidths=0.3,
                    zorder=4,
                )

        values = [
            row[key]
            for row in rows
            for key in (min_key, max_key)
        ]
        if values and all(value > 0 for value in values):
            ax.set_yscale("log")

        ax.set_xlim(-0.8, n_runs - 0.2)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3, zorder=1)

    axes[0].set_title(title or f"Min/Max NCM KL Divergence ({Path(root_dir)})")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([_group_label(group) for group in groups], rotation=45, ha="right", fontsize=8)
    axes[1].set_xlabel("Cycle lambda / mask mode, grouped by trial", labelpad=72)
    _add_trial_labels(axes[1], trial_groups)

    axes[0].legend(
        handles=[
            Line2D([0], [0], marker="o", color="black", markerfacecolor=min_color,
                   linestyle="", markersize=7, label="Min NCM"),
            Line2D([0], [0], marker="o", color="black", markerfacecolor=max_color,
                   linestyle="", markersize=7, label="Max NCM"),
        ],
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )

    fig.tight_layout(rect=(0, 0.22, 0.85, 1))

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, axes


def _mean(values):
    values = list(values)
    if not values:
        return math.nan
    return sum(values) / len(values)


def _std(values):
    values = list(values)
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def _fmt(value):
    if value is None or math.isnan(value):
        return "nan"
    return f"{value:.5g}"


def _print_summary_table(root_dir):
    rows = [
        row for row in _standard_rows(load_bound_results(root_dir))
        if all(
            key in row
            for key in (
                "min_total_true_KL",
                "max_total_true_KL",
                "min_total_dat_KL",
                "max_total_dat_KL",
                "avg_abs_bound_error",
            )
        )
    ]
    if not rows:
        print("No standard rows available for summary table.")
        return

    grouped = _group_rows(rows)
    groups = _sort_groups(grouped)
    headers = [
        "trial",
        "mask",
        "lambda",
        "n",
        "true_KL_mean",
        "true_KL_std",
        "dat_KL_mean",
        "dat_KL_std",
        "bound_err_mean",
        "bound_err_std",
        "max_bound_err",
    ]
    print()
    print("Summary table (standard runs only)")
    print("\t".join(headers))
    for group in groups:
        group_rows = grouped[group]
        cycle_lambda, mask_mode, trial_index = group
        true_kl = [
            (row["min_total_true_KL"] + row["max_total_true_KL"]) / 2
            for row in group_rows
        ]
        dat_kl = [
            (row["min_total_dat_KL"] + row["max_total_dat_KL"]) / 2
            for row in group_rows
        ]
        bound_error = [row["avg_abs_bound_error"] for row in group_rows]
        max_bound_error = [row.get("max_abs_bound_error", math.nan) for row in group_rows]
        print("\t".join([
            str(trial_index),
            str(mask_mode),
            _format_float(cycle_lambda),
            str(len(group_rows)),
            _fmt(_mean(true_kl)),
            _fmt(_std(true_kl)),
            _fmt(_mean(dat_kl)),
            _fmt(_std(dat_kl)),
            _fmt(_mean(bound_error)),
            _fmt(_std(bound_error)),
            _fmt(max(max_bound_error)),
        ]))


def _summary_metric_rows(root_dir):
    rows = [
        row for row in _standard_rows(load_bound_results(root_dir))
        if all(
            key in row
            for key in (
                "min_total_true_KL",
                "max_total_true_KL",
                "err_lower",
                "err_upper",
            )
        )
    ]
    metric_rows = []
    for row in rows:
        metric_rows.append({
            "group": (row["mask_mode"], row["cycle_lambda"]),
            "kl": (row["min_total_true_KL"] + row["max_total_true_KL"]) / 2,
            "min_bound_error": abs(row["err_lower"]),
            "max_bound_error": abs(row["err_upper"]),
        })
    return metric_rows


def plot_summary_stats(root_dir, output_path=None, title=None, figsize=None):
    metric_rows = _summary_metric_rows(root_dir)
    if not metric_rows:
        raise ValueError(f"No completed standard rows found for summary stats under {root_dir}")

    grouped = {}
    for row in metric_rows:
        grouped.setdefault(row["group"], []).append(row)

    groups = sorted(
        grouped,
        key=lambda group: (
            str(group[0]),
            float("inf") if group[1] is None else group[1],
        ),
    )
    labels = [
        f"{mask}\nlambda={_format_float(cycle_lambda)}"
        for mask, cycle_lambda in groups
    ]
    metrics = [
        ("kl", "KL"),
        ("min_bound_error", "Min Bound Error"),
        ("max_bound_error", "Max Bound Error"),
    ]
    colors = ["#4c78a8", "#d62728", "#ff7f0e"]

    if figsize is None:
        figsize = (max(12, 0.8 * len(groups)), 10)

    fig, axes = plt.subplots(len(metrics), 1, figsize=figsize, sharex=True)
    xs = list(range(len(groups)))

    for ax, (metric_key, ylabel), color in zip(axes, metrics, colors):
        means = [_mean(row[metric_key] for row in grouped[group]) for group in groups]
        stds = [_std(row[metric_key] for row in grouped[group]) for group in groups]
        ax.bar(xs, means, yerr=stds, color=color, alpha=0.82, capsize=4, ecolor="black")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
        if metric_key == "kl" and all(value > 0 for value in means):
            ax.set_yscale("log")

    axes[-1].set_xticks(xs)
    axes[-1].set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    axes[0].set_title(title or f"Aggregate KL and Bound Errors ({Path(root_dir)})")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, axes


def main():
    parser = argparse.ArgumentParser(description="Plot true and NCM min/max values for bound runs.")
    parser.add_argument("root_dir", help="Root directory to search recursively for results.json files")
    parser.add_argument("--output", help="Defaults to <root_dir>/bound_results.png")
    parser.add_argument("--title", help="Optional plot title")
    parser.add_argument("--kl-title", help="Optional KL plot title")
    parser.add_argument("--summary-output", help="Defaults to <root_dir>/bound_summary_stats.png")
    parser.add_argument("--summary-title", help="Optional aggregate summary plot title")
    parser.add_argument("--no-table", action="store_true", help="Do not print grouped KL/bound-error summary table")
    args = parser.parse_args()

    output = args.output or str(Path(args.root_dir) / "bound_results.png")
    kl_output = str(Path(args.root_dir) / "bound_kl_results.png")
    summary_output = args.summary_output or str(Path(args.root_dir) / "bound_summary_stats.png")
    plot_bound_results(args.root_dir, output_path=output, title=args.title)
    plot_kl_results(args.root_dir, output_path=kl_output, title=args.kl_title)
    plot_summary_stats(args.root_dir, output_path=summary_output, title=args.summary_title)
    if not args.no_table:
        _print_summary_table(args.root_dir)
    print(f"Saved plot to {output}")
    print(f"Saved KL plot to {kl_output}")
    print(f"Saved summary stats plot to {summary_output}")


if __name__ == "__main__":
    main()
