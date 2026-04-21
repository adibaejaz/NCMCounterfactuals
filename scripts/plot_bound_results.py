import argparse
import json
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
    cycle_lambda, mask_mode, trial_index = group
    return "\n".join([
        f"lambda={_format_float(cycle_lambda)}",
        str(mask_mode),
        f"t{trial_index}",
    ])


def _row_offsets(rows, max_span=0.62):
    if len(rows) <= 1:
        return [0.0] * len(rows)

    step = max_span / (len(rows) - 1)
    return [-max_span / 2 + i * step for i in range(len(rows))]


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
        })

    rows.sort(key=lambda row: row["label"])
    return rows


def plot_bound_results(root_dir, output_path=None, title=None, figsize=None):
    rows = [
        row for row in load_bound_results(root_dir)
        if row.get("standard_bound_run", True)
    ]
    if not rows:
        raise ValueError(f"No completed bound results found under {root_dir}")

    grouped = {}
    for row in rows:
        group = (row["cycle_lambda"], row["mask_mode"], row["trial_index"])
        grouped.setdefault(group, []).append(row)

    groups = sorted(
        grouped,
        key=lambda group: (
            float("inf") if group[0] is None else group[0],
            str(group[1]),
            int(group[2]) if str(group[2]).isdigit() else str(group[2]),
        ),
    )
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

    for x, group in zip(xs, groups):
        group_rows = sorted(
            grouped[group],
            key=lambda row: (
                _as_float(row.get("train_seed_offset"), default=float("inf")),
                row["run_id"],
            ),
        )
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
    ax.set_xlabel("Cycle lambda / mask mode / trial")
    ax.set_ylabel("Query value")
    ax.set_title(title or f"True Bounds and NCM Seed Results ({Path(root_dir)})")
    ax.grid(axis="y", alpha=0.3, zorder=1)

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

    fig.tight_layout(rect=(0, 0, 0.85, 1))

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_kl_results(root_dir, output_path=None, title=None, figsize=None):
    rows = [
        row for row in load_bound_results(root_dir)
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

    n_runs = len(rows)
    if figsize is None:
        figsize = (max(12, 0.8 * n_runs), 6)

    fig, ax = plt.subplots(figsize=figsize)

    true_color = "#1f77b4"
    dat_color = "#2ca02c"
    min_offset = -0.18
    max_offset = 0.18
    bar_width = 0.16
    xs = list(range(n_runs))

    for x in xs:
        if x % 2 == 0:
            ax.axvspan(x - 0.5, x + 0.5, color="0.96", zorder=0)

    min_true = [row["min_total_true_KL"] for row in rows]
    max_true = [row["max_total_true_KL"] for row in rows]
    min_dat = [row["min_total_dat_KL"] for row in rows]
    max_dat = [row["max_total_dat_KL"] for row in rows]

    ax.bar(
        [x + min_offset - bar_width / 2 for x in xs],
        min_true,
        width=bar_width,
        color=true_color,
        alpha=0.75,
        label="Min NCM true KL",
        zorder=3,
    )
    ax.bar(
        [x + max_offset - bar_width / 2 for x in xs],
        max_true,
        width=bar_width,
        color=true_color,
        alpha=1.0,
        label="Max NCM true KL",
        zorder=3,
    )
    ax.bar(
        [x + min_offset + bar_width / 2 for x in xs],
        min_dat,
        width=bar_width,
        color=dat_color,
        alpha=0.75,
        label="Min NCM data KL",
        zorder=3,
    )
    ax.bar(
        [x + max_offset + bar_width / 2 for x in xs],
        max_dat,
        width=bar_width,
        color=dat_color,
        alpha=1.0,
        label="Max NCM data KL",
        zorder=3,
    )

    all_values = min_true + max_true + min_dat + max_dat
    if all(value > 0 for value in all_values):
        ax.set_yscale("log")

    ax.set_xlim(-0.8, n_runs - 0.2)
    ax.set_xticks(xs)
    ax.set_xticklabels([row["label"] for row in rows], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Run ID")
    ax.set_ylabel("Total KL divergence")
    ax.set_title(title or f"Min/Max NCM KL Divergence ({Path(root_dir)})")
    ax.grid(axis="y", alpha=0.3, zorder=1)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )

    fig.tight_layout(rect=(0, 0, 0.85, 1))

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, ax


def main():
    parser = argparse.ArgumentParser(description="Plot true and NCM min/max values for bound runs.")
    parser.add_argument("root_dir", help="Root directory to search recursively for results.json files")
    parser.add_argument("--output", help="Defaults to <root_dir>/bound_results.png")
    parser.add_argument("--title", help="Optional plot title")
    parser.add_argument("--kl-title", help="Optional KL plot title")
    args = parser.parse_args()

    output = args.output or str(Path(args.root_dir) / "bound_results.png")
    kl_output = str(Path(args.root_dir) / "bound_kl_results.png")
    plot_bound_results(args.root_dir, output_path=output, title=args.title)
    plot_kl_results(args.root_dir, output_path=kl_output, title=args.kl_title)
    print(f"Saved plot to {output}")
    print(f"Saved KL plot to {kl_output}")


if __name__ == "__main__":
    main()
