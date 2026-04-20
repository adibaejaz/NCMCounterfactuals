import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _find_results_files(root_dir):
    return sorted(Path(root_dir).rglob("results.json"))


def _extract_run_label(results_path, root_dir):
    rel = results_path.relative_to(root_dir)
    parts = rel.parts[:-1]

    if len(parts) >= 2 and parts[-1].isdigit():
        run_dir = parts[-2]
    elif parts:
        run_dir = parts[-1]
    else:
        run_dir = results_path.stem

    run_id = run_dir
    trial_index = "?"

    for piece in run_dir.split("-"):
        if piece.startswith("run="):
            run_id = piece[len("run="):]
        elif piece.startswith("trial_index="):
            trial_index = piece[len("trial_index="):]

    return f"{run_id}:t{trial_index}"


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


def load_bound_results(root_dir):
    root = Path(root_dir)
    rows = []

    for results_path in _find_results_files(root):
        with open(results_path) as f:
            results = json.load(f)

        series = _extract_bound_series(results)
        if series is None:
            continue

        rows.append({
            "label": _extract_run_label(results_path, root),
            "results_path": str(results_path),
            **series,
        })

    rows.sort(key=lambda row: row["label"])
    return rows


def plot_bound_results(root_dir, output_path=None, title=None, figsize=None):
    rows = load_bound_results(root_dir)
    if not rows:
        raise ValueError(f"No completed bound results found under {root_dir}")

    n_runs = len(rows)
    if figsize is None:
        figsize = (max(12, 0.7 * n_runs), 6)

    fig, ax = plt.subplots(figsize=figsize)

    true_color = "#1f77b4"
    ncm_color = "#d62728"
    half_width = 0.16
    xs = list(range(n_runs))

    for x in xs:
        if x % 2 == 0:
            ax.axvspan(x - 0.5, x + 0.5, color="0.96", zorder=0)

    for x, row in zip(xs, rows):
        ax.hlines(
            [row["true_lower"], row["true_upper"]],
            x - 0.34 - half_width,
            x - 0.34 + half_width,
            color=true_color,
            linewidth=2.8,
            zorder=3,
        )
        ax.hlines(
            [row["ncm_min"], row["ncm_max"]],
            x + 0.34 - half_width,
            x + 0.34 + half_width,
            color=ncm_color,
            linewidth=2.8,
            zorder=3,
        )

    ax.set_xlim(-0.8, n_runs - 0.2)
    ax.set_xticks(xs)
    ax.set_xticklabels([row["label"] for row in rows], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Run ID")
    ax.set_ylabel("Query value")
    ax.set_title(title or f"True vs NCM Bounds ({Path(root_dir)})")
    ax.grid(axis="y", alpha=0.3, zorder=1)

    ax.legend(
        handles=[
            Line2D([0], [0], color=true_color, lw=2.8, label="True min/max"),
            Line2D([0], [0], color=ncm_color, lw=2.8, label="NCM min/max"),
        ],
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
    args = parser.parse_args()

    output = args.output or str(Path(args.root_dir) / "bound_results.png")
    plot_bound_results(args.root_dir, output_path=output, title=args.title)
    print(f"Saved plot to {output}")


if __name__ == "__main__":
    main()
