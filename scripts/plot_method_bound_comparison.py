import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


DEFAULT_METHODS = [
    (
        "RL NCM",
        "rl-summary",
        "out/yushu-1d-rl-ncm",
    ),
    (
        "FF NCM",
        "bound",
        "out/overnight_runs/coupling/"
        "paper_bound_masked_ff_overnight_coupled_v1_querymask_mfit1_1x1_uniform",
    ),
    (
        "Sampling baseline",
        "bound",
        "out/paper_enum_baseline",
    ),
]
DEFAULT_GRAPHS = ["chain", "backdoor", "square", "clique"]
GRAPH_ALIASES = {
    "clique": "four_clique",
}
DISPLAY_GRAPH_NAMES = {
    "four_clique": "clique",
}
POOL_NONE = "none"
POOL_ENDPOINTS = "endpoints"
POOL_ALL_OUTPUTS = "all-outputs"
TRIAL_LABEL_Y = -0.22


def _canonical_graph(graph):
    return GRAPH_ALIASES.get(str(graph), str(graph))


def _display_graph(graph):
    return DISPLAY_GRAPH_NAMES.get(str(graph), str(graph))


def _slug(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _as_float(value, default=None):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


def _trial_sort_key(value):
    return int(value) if str(value).isdigit() else str(value)


def _rerun_sort_key(value):
    return int(value) if str(value).isdigit() else str(value)


def _extract_piece(run_dir, prefix, default="?"):
    for piece in run_dir.split("-"):
        if piece.startswith(prefix):
            return piece[len(prefix):]
    return default


def _result_run_dir(results_path, root):
    rel = results_path.relative_to(root)
    parts = rel.parts[:-1]
    if len(parts) >= 2 and parts[-1].isdigit():
        return parts[-2]
    if parts:
        return parts[-1]
    return results_path.stem


def _result_run_id(run_dir):
    return _extract_piece(run_dir, "run=", default=run_dir)


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _load_hyperparams(results_path):
    run_path = results_path.parent
    if run_path.name.isdigit():
        run_path = run_path.parent
    path = run_path / "hyperparams.json"
    if not path.is_file():
        return {}
    return _load_json(path)


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
        elif key.startswith("min_ncm_") or key.startswith("enum_min_ncm_"):
            ncm_min = value
        elif key.startswith("max_ncm_") or key.startswith("enum_max_ncm_"):
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
    standard_keys = {
        "min_total_true_KL": "min_kl",
        "max_total_true_KL": "max_kl",
    }
    enum_keys = {
        "enum_min_total_true_KL": "min_kl",
        "enum_max_total_true_KL": "max_kl",
    }
    if all(key in results for key in standard_keys):
        return {alias: float(results[key]) for key, alias in standard_keys.items()}
    if all(key in results for key in enum_keys):
        return {alias: float(results[key]) for key, alias in enum_keys.items()}
    return {}


def _extract_bound_errors(results, series):
    err_lower = None
    err_upper = None

    for key, value in results.items():
        if key.startswith("enum_min_err_ncm_"):
            err_lower = value
        elif key.startswith("enum_max_err_ncm_"):
            err_upper = value
        elif key.startswith("err_min_ncm_") and key.endswith("_lower_bound"):
            err_lower = value
        elif key.startswith("err_max_ncm_") and key.endswith("_upper_bound"):
            err_upper = value

    if err_lower is None:
        err_lower = series["true_lower"] - series["ncm_min"]
    if err_upper is None:
        err_upper = series["true_upper"] - series["ncm_max"]

    min_bound_error = abs(float(err_lower))
    max_bound_error = abs(float(err_upper))
    return {
        "min_bound_error": min_bound_error,
        "max_bound_error": max_bound_error,
        "avg_abs_bound_error": (min_bound_error + max_bound_error) / 2,
    }


def _has_numeric_child_results(run_dir):
    return any((child / "results.json").is_file() for child in run_dir.iterdir() if child.is_dir() and child.name.isdigit())


def _bound_rerun(results_path, results, hyperparams):
    if "rerun_index" in results:
        return str(results["rerun_index"])
    if results_path.parent.name.isdigit() and str(hyperparams.get("id-reruns")) not in {"", "None", "1"}:
        return results_path.parent.name
    if "train-seed-offset" in hyperparams:
        return str(hyperparams["train-seed-offset"])
    if results_path.parent.name.isdigit():
        return results_path.parent.name
    return _result_run_id(_result_run_dir(results_path, results_path.parents[0]))


def _swap_inverted_bound_row(row):
    if row["ncm_max"] >= row["ncm_min"]:
        row["swapped_inverted_bounds"] = False
        return row

    row["ncm_min"], row["ncm_max"] = row["ncm_max"], row["ncm_min"]
    if "min_kl" in row and "max_kl" in row:
        row["min_kl"], row["max_kl"] = row["max_kl"], row["min_kl"]
    row.update(_extract_bound_errors({}, row))
    row["swapped_inverted_bounds"] = True
    return row


def load_bound_method(label, root_dir, dim, graphs, swap_inverted=False):
    root = Path(root_dir)
    graph_set = {_canonical_graph(graph) for graph in graphs}
    rows = []

    for results_path in sorted(root.rglob("results.json")):
        rel_parts = results_path.relative_to(root).parts
        if "dags" in rel_parts:
            continue
        if not results_path.parent.name.isdigit() and _has_numeric_child_results(results_path.parent):
            continue

        results = _load_json(results_path)
        series = _extract_bound_series(results)
        if series is None:
            continue

        run_dir = _result_run_dir(results_path, root)
        graph = _canonical_graph(_extract_piece(run_dir, "graph="))
        dimension = _extract_piece(run_dir, "dim=")
        if str(dimension) != str(dim) or graph not in graph_set:
            continue

        hyperparams = _load_hyperparams(results_path)
        row = {
            "method": label,
            "source_kind": "bound-swap-inverted" if swap_inverted else "bound",
            "graph": graph,
            "dimension": str(dimension),
            "trial": str(_extract_piece(run_dir, "trial_index=")),
            "rerun": str(_bound_rerun(results_path, results, hyperparams)),
            "run_id": _result_run_id(run_dir),
            "results_path": str(results_path),
            "swapped_inverted_bounds": False,
            "_numeric_rerun_file": results_path.parent.name.isdigit(),
            **series,
            **_extract_kl_series(results),
            **_extract_bound_errors(results, series),
        }
        if swap_inverted:
            row = _swap_inverted_bound_row(row)
        rows.append(row)

    numeric_keys = {
        (row["graph"], row["trial"])
        for row in rows
        if row["_numeric_rerun_file"]
    }
    filtered_rows = [
        row
        for row in rows
        if row["_numeric_rerun_file"] or (row["graph"], row["trial"]) not in numeric_keys
    ]
    for row in filtered_rows:
        row.pop("_numeric_rerun_file", None)
    return filtered_rows


def load_rl_summary_method(label, root_dir, dim, graphs, rl_kl_key):
    if str(dim) != "1":
        return []

    root = Path(root_dir)
    graph_set = {_canonical_graph(graph) for graph in graphs}
    rows = []

    for path in sorted(root.glob("*_run_summary.json")):
        data = _load_json(path)
        for item in data:
            graph = _canonical_graph(item.get("graph", path.name.removesuffix("_run_summary.json")))
            if graph not in graph_set:
                continue

            true_lower = float(item["ground_truth_bounds"]["lower"])
            true_upper = float(item["ground_truth_bounds"]["upper"])
            ncm_min = float(item["estimated_bounds"]["lower"])
            ncm_max = float(item["estimated_bounds"]["upper"])
            kl_values = item.get("kl", {}).get(rl_kl_key, {})
            min_bound_error = abs(ncm_min - true_lower)
            max_bound_error = abs(true_upper - ncm_max)
            rows.append({
                "method": label,
                "source_kind": "rl-summary",
                "graph": graph,
                "dimension": str(dim),
                "trial": str(item["trial"]),
                "rerun": str(item["rerun"]),
                "run_id": f"trial{item['trial']}_rerun{item['rerun']}",
                "results_path": str(path),
                "true_lower": true_lower,
                "true_upper": true_upper,
                "ncm_min": ncm_min,
                "ncm_max": ncm_max,
                "min_kl": _as_float(kl_values.get("minimize"), default=math.nan),
                "max_kl": _as_float(kl_values.get("maximize"), default=math.nan),
                "min_bound_error": min_bound_error,
                "max_bound_error": max_bound_error,
                "avg_abs_bound_error": (min_bound_error + max_bound_error) / 2,
            })

    return rows


def parse_method_spec(value):
    try:
        label, rest = value.split("=", 1)
        kind, root = rest.split(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected LABEL=KIND:ROOT, where KIND is bound, bound-swap-inverted, or rl-summary"
        ) from exc
    kind = kind.strip()
    if kind not in {"bound", "bound-swap-inverted", "rl-summary"}:
        raise argparse.ArgumentTypeError("method KIND must be bound, bound-swap-inverted, or rl-summary")
    return label.strip(), kind, root.strip()


def load_methods(methods, dim, graphs, rl_kl_key):
    rows = []
    for label, kind, root in methods:
        if kind == "bound":
            rows.extend(load_bound_method(label, root, dim, graphs))
        elif kind == "bound-swap-inverted":
            rows.extend(load_bound_method(label, root, dim, graphs, swap_inverted=True))
        elif kind == "rl-summary":
            rows.extend(load_rl_summary_method(label, root, dim, graphs, rl_kl_key))
    rows.sort(key=lambda row: (
        _canonical_graph(row["graph"]),
        _trial_sort_key(row["trial"]),
        row["method"],
        _rerun_sort_key(row["rerun"]),
        row["run_id"],
    ))
    return rows


def _method_order(rows, requested_methods):
    labels = [label for label, _kind, _root in requested_methods]
    present = {row["method"] for row in rows}
    return [label for label in labels if label in present]


def _row_offsets(rows, max_span=0.48):
    if len(rows) <= 1:
        return [0.0] * len(rows)
    step = max_span / (len(rows) - 1)
    return [-max_span / 2 + i * step for i in range(len(rows))]


def _pooled_bound_values(rows, mode):
    if mode == POOL_NONE:
        return None
    if mode == POOL_ENDPOINTS:
        return min(row["ncm_min"] for row in rows), max(row["ncm_max"] for row in rows)
    if mode == POOL_ALL_OUTPUTS:
        values = [value for row in rows for value in (row["ncm_min"], row["ncm_max"])]
        return min(values), max(values)
    raise ValueError(f"unknown pooling mode: {mode}")


def _group_rows_for_plot(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["trial"], row["method"]), []).append(row)
    return grouped


def plot_bound_comparison(rows, graph, method_labels, output_path, pool_reruns=POOL_NONE, title=None):
    graph_rows = [row for row in rows if row["graph"] == graph]
    if not graph_rows:
        raise ValueError(f"No rows available for graph={graph}")

    trials = sorted({row["trial"] for row in graph_rows}, key=_trial_sort_key)
    methods = [method for method in method_labels if any(row["method"] == method for row in graph_rows)]
    grouped = _group_rows_for_plot(graph_rows)
    groups = [(trial, method) for trial in trials for method in methods if (trial, method) in grouped]
    xs = list(range(len(groups)))

    fig_width = max(12, 0.75 * len(groups))
    fig, ax = plt.subplots(figsize=(fig_width, 6.5))
    true_color = "#1f77b4"
    min_color = "#d62728"
    max_color = "#ff7f0e"
    half_width = 0.31

    trial_xs = {}
    for x, (trial, _method) in zip(xs, groups):
        trial_xs.setdefault(trial, []).append(x)
        if x % 2 == 0:
            ax.axvspan(x - 0.5, x + 0.5, color="0.96", zorder=0)

    for trial, cur_xs in trial_xs.items():
        start = min(cur_xs) - 0.5
        end = max(cur_xs) + 0.5
        if str(trial).isdigit() and int(trial) % 2 == 0:
            ax.axvspan(start, end, color="0.98", zorder=-1)
        if start > -0.5:
            ax.axvline(start, color="0.75", linewidth=0.8, zorder=1)

    for x, group in zip(xs, groups):
        cur_rows = sorted(grouped[group], key=lambda row: (_rerun_sort_key(row["rerun"]), row["run_id"]))
        true_lower = _mean(row["true_lower"] for row in cur_rows)
        true_upper = _mean(row["true_upper"] for row in cur_rows)
        ax.hlines([true_lower, true_upper], x - half_width, x + half_width, color=true_color, linewidth=4, zorder=3)

        pooled_values = _pooled_bound_values(cur_rows, pool_reruns)
        if pooled_values is None:
            for offset, row in zip(_row_offsets(cur_rows), cur_rows):
                ax.scatter([x + offset], [row["ncm_min"]], color=min_color, marker="o", s=34,
                           edgecolors="black", linewidths=0.3, zorder=4)
                ax.scatter([x + offset], [row["ncm_max"]], color=max_color, marker="o", s=34,
                           edgecolors="black", linewidths=0.3, zorder=4)
        else:
            pooled_min, pooled_max = pooled_values
            ax.scatter([x], [pooled_min], color=min_color, marker="D", s=48,
                       edgecolors="black", linewidths=0.4, zorder=4)
            ax.scatter([x], [pooled_max], color=max_color, marker="D", s=48,
                       edgecolors="black", linewidths=0.4, zorder=4)

    ax.set_xlim(-0.8, xs[-1] + 0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([method for _trial, method in groups], rotation=35, ha="right", fontsize=9)
    ax.set_xlabel("Method, grouped by trial", labelpad=58)
    ax.set_ylabel("Query value")
    ax.set_title(title or f"True Bounds and Method Results: {_display_graph(graph)}")
    ax.grid(axis="y", alpha=0.3, zorder=1)

    for trial, cur_xs in trial_xs.items():
        center = (min(cur_xs) + max(cur_xs)) / 2
        ax.text(center, TRIAL_LABEL_Y, f"trial {trial}", ha="center", va="top",
                fontsize=9, transform=ax.get_xaxis_transform(), clip_on=False)

    marker = "D" if pool_reruns != POOL_NONE else "o"
    ax.legend(
        handles=[
            Line2D([0], [0], color=true_color, lw=4, label="True min/max"),
            Line2D([0], [0], marker=marker, color="black", markerfacecolor=min_color,
                   linestyle="", markersize=7, label="NCM min"),
            Line2D([0], [0], marker=marker, color="black", markerfacecolor=max_color,
                   linestyle="", markersize=7, label="NCM max"),
        ],
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    fig.subplots_adjust(bottom=0.28, right=0.84)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _finite(values):
    return [value for value in values if value is not None and not math.isnan(value)]


def plot_summary_stats(rows, graph, method_labels, output_path, title=None):
    graph_rows = [row for row in rows if row["graph"] == graph]
    if not graph_rows:
        raise ValueError(f"No rows available for graph={graph}")

    methods = [method for method in method_labels if any(row["method"] == method for row in graph_rows)]
    grouped = {method: [row for row in graph_rows if row["method"] == method] for method in methods}
    metrics = [
        ("min_kl", "Min KL"),
        ("max_kl", "Max KL"),
        ("min_bound_error", "Min Bound Error"),
        ("max_bound_error", "Max Bound Error"),
    ]
    colors = ["#4c78a8", "#72b7b2", "#d62728", "#ff7f0e"]
    xs = list(range(len(methods)))

    fig, axes = plt.subplots(len(metrics), 1, figsize=(max(8, 1.8 * len(methods)), 9.5), sharex=True)
    for ax, (metric, ylabel), color in zip(axes, metrics, colors):
        means = [_mean(_finite(row.get(metric) for row in grouped[method])) for method in methods]
        stds = [_std(_finite(row.get(metric) for row in grouped[method])) for method in methods]
        ax.bar(xs, means, yerr=stds, color=color, alpha=0.82, capsize=4, ecolor="black")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
        finite_means = _finite(means)
        if metric in {"min_kl", "max_kl"} and finite_means and all(value > 0 for value in finite_means):
            ax.set_yscale("log")

    axes[-1].set_xticks(xs)
    axes[-1].set_xticklabels(methods, rotation=25, ha="right", fontsize=10)
    axes[0].set_title(title or f"Aggregate KL and Bound Errors: {_display_graph(graph)}")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_rows_csv(rows, output_path):
    fieldnames = [
        "method",
        "source_kind",
        "graph",
        "dimension",
        "trial",
        "rerun",
        "run_id",
        "true_lower",
        "true_upper",
        "ncm_min",
        "ncm_max",
        "min_kl",
        "max_kl",
        "min_bound_error",
        "max_bound_error",
        "avg_abs_bound_error",
        "swapped_inverted_bounds",
        "results_path",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def summary_rows(rows, method_labels):
    output = []
    for graph in sorted({row["graph"] for row in rows}):
        for method in method_labels:
            cur_rows = [row for row in rows if row["graph"] == graph and row["method"] == method]
            if not cur_rows:
                continue
            output.append({
                "graph": _display_graph(graph),
                "method": method,
                "n": len(cur_rows),
                "trials": ",".join(str(value) for value in sorted({row["trial"] for row in cur_rows}, key=_trial_sort_key)),
                "reruns": ",".join(str(value) for value in sorted({row["rerun"] for row in cur_rows}, key=_rerun_sort_key)),
                "min_kl_mean": _mean(_finite(row.get("min_kl") for row in cur_rows)),
                "min_kl_std": _std(_finite(row.get("min_kl") for row in cur_rows)),
                "max_kl_mean": _mean(_finite(row.get("max_kl") for row in cur_rows)),
                "max_kl_std": _std(_finite(row.get("max_kl") for row in cur_rows)),
                "min_bound_error_mean": _mean(row["min_bound_error"] for row in cur_rows),
                "min_bound_error_std": _std(row["min_bound_error"] for row in cur_rows),
                "max_bound_error_mean": _mean(row["max_bound_error"] for row in cur_rows),
                "max_bound_error_std": _std(row["max_bound_error"] for row in cur_rows),
                "avg_abs_bound_error_mean": _mean(row["avg_abs_bound_error"] for row in cur_rows),
                "avg_abs_bound_error_std": _std(row["avg_abs_bound_error"] for row in cur_rows),
            })
    return output


def write_summary_csv(rows, output_path):
    fieldnames = [
        "graph",
        "method",
        "n",
        "trials",
        "reruns",
        "min_kl_mean",
        "min_kl_std",
        "max_kl_mean",
        "max_kl_std",
        "min_bound_error_mean",
        "min_bound_error_std",
        "max_bound_error_mean",
        "max_bound_error_std",
        "avg_abs_bound_error_mean",
        "avg_abs_bound_error_std",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_count_warnings(rows, method_labels, graphs, expected_trials, expected_reruns):
    if expected_trials is None and expected_reruns is None:
        return

    expected_n = None
    if expected_trials is not None and expected_reruns is not None:
        expected_n = expected_trials * expected_reruns

    for graph in graphs:
        for method in method_labels:
            cur_rows = [row for row in rows if row["graph"] == graph and row["method"] == method]
            if not cur_rows:
                print(f"Warning: graph={_display_graph(graph)} method={method} has no rows")
                continue

            trials = sorted({row["trial"] for row in cur_rows}, key=_trial_sort_key)
            reruns = sorted({row["rerun"] for row in cur_rows}, key=_rerun_sort_key)
            messages = []
            if expected_n is not None and len(cur_rows) != expected_n:
                messages.append(f"n={len(cur_rows)} expected={expected_n}")
            if expected_trials is not None and len(trials) != expected_trials:
                messages.append(f"trials={','.join(trials)}")
            if expected_reruns is not None and len(reruns) != expected_reruns:
                messages.append(f"reruns={','.join(reruns)}")
            if messages:
                print(f"Warning: graph={_display_graph(graph)} method={method} " + "; ".join(messages))


def main():
    parser = argparse.ArgumentParser(
        description="Compare bound results across methods, including Yushu RL NCM summaries."
    )
    parser.add_argument(
        "--method",
        action="append",
        type=parse_method_spec,
        dest="methods",
        help=(
            "Method source as LABEL=KIND:ROOT. KIND is bound for this repo's results.json "
            "layout or rl-summary for Yushu *_run_summary.json files. May be repeated. "
            "Use KIND=bound-swap-inverted to swap min/max values within any run where max < min. "
            "Defaults to RL NCM, FF NCM, and Sampling baseline."
        ),
    )
    parser.add_argument("--dim", default="1", help="Dimension label to include. Default: 1")
    parser.add_argument(
        "--graph",
        action="append",
        dest="graphs",
        help="Graph to include. May be repeated. 'clique' aliases to 'four_clique'.",
    )
    parser.add_argument(
        "--output-dir",
        default="out/method_bound_comparison_dim1",
        help="Directory for plots and CSV outputs.",
    )
    parser.add_argument(
        "--pool-reruns",
        choices=[POOL_NONE, POOL_ENDPOINTS, POOL_ALL_OUTPUTS],
        default=POOL_NONE,
        help="Pool rerun markers inside each trial/method group.",
    )
    parser.add_argument(
        "--rl-kl-key",
        choices=["best", "final"],
        default="best",
        help="Which KL values to use from rl-summary files.",
    )
    parser.add_argument(
        "--expected-trials",
        type=int,
        default=5,
        help="Expected number of trials per graph/method for warnings. Use 0 to disable.",
    )
    parser.add_argument(
        "--expected-reruns",
        type=int,
        default=3,
        help="Expected number of reruns per graph/method for warnings. Use 0 to disable.",
    )
    args = parser.parse_args()

    methods = args.methods or DEFAULT_METHODS
    graphs = args.graphs or DEFAULT_GRAPHS
    canonical_graphs = [_canonical_graph(graph) for graph in graphs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_methods(methods, args.dim, canonical_graphs, args.rl_kl_key)
    if not rows:
        raise ValueError("No matching rows found for the requested methods, graphs, and dimension.")

    method_labels = _method_order(rows, methods)
    print_count_warnings(
        rows,
        method_labels,
        canonical_graphs,
        args.expected_trials or None,
        args.expected_reruns or None,
    )
    normalized_csv = output_dir / "normalized_rows.csv"
    summary_csv = output_dir / "summary_stats.csv"
    write_rows_csv(rows, normalized_csv)
    write_summary_csv(summary_rows(rows, method_labels), summary_csv)

    saved_paths = [normalized_csv, summary_csv]
    for graph in canonical_graphs:
        graph_rows = [row for row in rows if row["graph"] == graph]
        if not graph_rows:
            print(f"Skipping graph={_display_graph(graph)}: no rows")
            continue
        graph_slug = _slug(_display_graph(graph))
        bound_path = output_dir / f"bound_results_{graph_slug}.png"
        summary_path = output_dir / f"bound_summary_stats_{graph_slug}.png"
        plot_bound_comparison(
            rows,
            graph,
            method_labels,
            bound_path,
            pool_reruns=args.pool_reruns,
        )
        plot_summary_stats(rows, graph, method_labels, summary_path)
        saved_paths.extend([bound_path, summary_path])

    for path in saved_paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
