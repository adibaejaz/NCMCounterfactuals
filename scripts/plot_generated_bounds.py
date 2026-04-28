"""Plot certified ground-truth bounds from generated dataset folders."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _parse_run_dir(run_dir):
    result = {}
    for piece in run_dir.name.split("-"):
        if "=" in piece:
            key, value = piece.split("=", 1)
            result[key] = value
    return result


def _trial_sort_key(row):
    try:
        trial = int(row["trial_index"])
    except ValueError:
        trial = row["trial_index"]
    return row["graph"], trial, int(row["treatment_value"])


def _find_bound_files(root):
    return sorted(Path(root).glob("graph=*-n_samples=*-dim=*-trial_index=*/ground_truth_bounds.json"))


def load_bound_rows(root):
    rows = []
    for path in _find_bound_files(root):
        run_info = _parse_run_dir(path.parent)
        with open(path) as file:
            payload = json.load(file)

        by_value = payload.get("bounds_by_treatment_value")
        if by_value is None:
            by_value = {
                str(payload.get("acceptance_treatment_value", "?")): {
                    "query": payload.get("query", "?"),
                    "lower": payload["lower"],
                    "upper": payload["upper"],
                    "gap": payload["gap"],
                }
            }

        for treatment_value, bound in by_value.items():
            rows.append({
                "graph": run_info.get("graph", "?"),
                "trial_index": run_info.get("trial_index", "?"),
                "n_samples": run_info.get("n_samples", "?"),
                "dim": run_info.get("dim", "?"),
                "treatment_value": str(treatment_value),
                "query": bound.get("query", "?"),
                "lower": float(bound["lower"]),
                "upper": float(bound["upper"]),
                "gap": float(bound["gap"]),
                "path": str(path),
            })
    return sorted(rows, key=_trial_sort_key)


def write_csv(rows, output_path):
    if not rows:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _graph_order(rows):
    preferred = ["chain", "backdoor", "square", "four_clique"]
    present = {row["graph"] for row in rows}
    ordered = [graph for graph in preferred if graph in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def _treatment_values(rows, selected):
    values = sorted({row["treatment_value"] for row in rows}, key=lambda value: int(value))
    if selected == "all":
        return values
    selected_values = [value.strip() for value in selected.split(",") if value.strip()]
    return [value for value in values if value in selected_values]


def plot_bounds(rows, output_path, treatment_values="all", title=None):
    if not rows:
        raise ValueError("No generated bounds found")

    selected_values = _treatment_values(rows, treatment_values)
    if not selected_values:
        raise ValueError("No bounds matched treatment values: {}".format(treatment_values))

    graph_order = _graph_order(rows)
    fig, axes = plt.subplots(
        len(selected_values),
        1,
        figsize=(max(10, 1.2 * len(rows)), 3.2 * len(selected_values)),
        sharey=True,
        squeeze=False)
    axes = axes[:, 0]

    colors = {
        "chain": "#4C78A8",
        "backdoor": "#F58518",
        "square": "#54A24B",
        "four_clique": "#B279A2",
    }

    for ax, treatment_value in zip(axes, selected_values):
        value_rows = [row for row in rows if row["treatment_value"] == treatment_value]
        positions = []
        labels = []
        x = 0
        for graph in graph_order:
            graph_rows = [row for row in value_rows if row["graph"] == graph]
            graph_rows = sorted(graph_rows, key=lambda row: int(row["trial_index"]))
            for row in graph_rows:
                color = colors.get(graph, "#666666")
                center = (row["lower"] + row["upper"]) / 2
                lower_err = center - row["lower"]
                upper_err = row["upper"] - center
                ax.errorbar(
                    x,
                    center,
                    yerr=[[lower_err], [upper_err]],
                    fmt="o",
                    color=color,
                    ecolor=color,
                    capsize=4,
                    markersize=5)
                ax.text(
                    x,
                    min(1.02, row["upper"] + 0.025),
                    "{:.2f}".format(row["gap"]),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90)
                positions.append(x)
                labels.append("{}\nt{}".format(graph, row["trial_index"]))
                x += 1
            if graph_rows:
                x += 0.8

        ax.set_title("P(Y=1 | do(X={}))".format(treatment_value))
        ax.set_ylabel("Ground-truth bound")
        ax.set_ylim(-0.03, 1.08)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=colors.get(graph, "#666666"), label=graph)
        for graph in graph_order
    ]
    axes[0].legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig, axes


def main():
    parser = argparse.ArgumentParser(description="Plot generated dataset ground-truth bounds")
    parser.add_argument("--root", default="out/paper_bound_datasets")
    parser.add_argument("--out", default=None, help="output figure path")
    parser.add_argument("--csv-out", default=None, help="optional CSV summary path")
    parser.add_argument("--treatment-values", default="all", help="'all' or comma-separated values, e.g. '0' or '0,1'")
    parser.add_argument("--title", default="Generated Dataset Ground-Truth Bounds")
    args = parser.parse_args()

    rows = load_bound_rows(args.root)
    if args.out is None:
        args.out = str(Path(args.root) / "generated_bounds.png")
    if args.csv_out is None:
        args.csv_out = str(Path(args.root) / "generated_bounds.csv")

    write_csv(rows, args.csv_out)
    plot_bounds(rows, args.out, treatment_values=args.treatment_values, title=args.title)
    print("Loaded {} bound rows".format(len(rows)))
    print("Wrote {}".format(args.out))
    print("Wrote {}".format(args.csv_out))


if __name__ == "__main__":
    main()
