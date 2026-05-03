#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_bound_results import (
    _as_float,
    _extract_bound_series,
    _extract_kl_series,
    _parse_schedule_filter,
    _standard_bound_run,
)


def _find_mask_pairs(root_dir):
    return sorted(Path(root_dir).rglob("mask_min.json"))


def _run_dir_from_mask_path(mask_path):
    run_dir = mask_path.parent
    if run_dir.name.isdigit():
        run_dir = run_dir.parent
    return run_dir


def _extract_run_id(run_dir):
    for piece in run_dir.name.split("-"):
        if piece.startswith("run="):
            return piece[len("run="):]
    return run_dir.name


def _extract_trial_index(run_dir):
    for piece in run_dir.name.split("-"):
        if piece.startswith("trial_index="):
            return piece[len("trial_index="):]
    return "?"


def _extract_graph(run_dir):
    for piece in run_dir.name.split("-"):
        if piece.startswith("graph="):
            return piece[len("graph="):]
    return "?"


def _extract_dimension_label(run_dir):
    for piece in run_dir.name.split("-"):
        if piece.startswith("dim="):
            return piece[len("dim="):]
    return "?"


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _load_mask(path):
    obj = _load_json(path)
    return list(obj["variables"]), np.asarray(obj["mask"], dtype=float)


def _load_hyperparams(run_dir):
    path = run_dir / "hyperparams.json"
    if not path.is_file():
        return {}
    return _load_json(path)


def _load_result_metrics(mask_dir):
    path = mask_dir / "results.json"
    if not path.is_file():
        return {}

    results = _load_json(path)
    metrics = {}
    metrics.update(_extract_bound_series(results) or {})
    metrics.update(_extract_kl_series(results) or {})
    return metrics


def _row_label(row):
    offset = row["train_seed_offset"]
    if offset is None:
        offset = "0"
    lines = [
        f"seed {offset}",
        row["run_id"][:6],
        f"tlr={_format_optional_float(row['theta_lr'])}",
        f"mlr={_format_optional_float(row['mask_lr'])}",
        f"s={row['theta_steps_per_mask']}:{row['mask_steps_per_theta']}",
    ]
    if all(key in row for key in ("true_lower", "true_upper", "ncm_min", "ncm_max")):
        lines.extend([
            f"true=[{_format_optional_float(row['true_lower'])}, {_format_optional_float(row['true_upper'])}]",
            f"ncm=[{_format_optional_float(row['ncm_min'])}, {_format_optional_float(row['ncm_max'])}]",
        ])
    if all(key in row for key in ("min_total_true_KL", "max_total_true_KL")):
        lines.append(
            f"KL true={_format_optional_float(row['min_total_true_KL'])}/"
            f"{_format_optional_float(row['max_total_true_KL'])}"
        )
    if all(key in row for key in ("min_total_dat_KL", "max_total_dat_KL")):
        lines.append(
            f"KL dat={_format_optional_float(row['min_total_dat_KL'])}/"
            f"{_format_optional_float(row['max_total_dat_KL'])}"
        )
    return "\n".join(lines)


def _trial_sort_key(trial_index):
    return int(trial_index) if str(trial_index).isdigit() else str(trial_index)


def _graph_sort_key(graph):
    return str(graph)


def _sanitize_output_label(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _filter_values(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value}
    return {str(value)}


def _format_optional_float(value):
    if value is None:
        return "?"
    return f"{value:g}"


def load_mask_rows(root_dir, include_nonstandard=False):
    rows = []
    for mask_min_path in _find_mask_pairs(root_dir):
        mask_dir = mask_min_path.parent
        mask_max_path = mask_dir / "mask_max.json"
        if not mask_max_path.is_file():
            continue

        run_dir = _run_dir_from_mask_path(mask_min_path)
        hyperparams = _load_hyperparams(run_dir)
        if not include_nonstandard and not _standard_bound_run(hyperparams):
            continue

        min_vars, mask_min = _load_mask(mask_min_path)
        max_vars, mask_max = _load_mask(mask_max_path)
        if min_vars != max_vars:
            raise ValueError(f"mask variable order mismatch in {mask_dir}")

        rows.append({
            "run_id": _extract_run_id(run_dir),
            "trial_index": _extract_trial_index(run_dir),
            "graph": _extract_graph(run_dir),
            "dimension": _extract_dimension_label(run_dir),
            "cycle_lambda": _as_float(hyperparams.get("cycle-lambda")),
            "mask_mode": hyperparams.get("mask-mode", "?"),
            "train_seed_offset": hyperparams.get("train-seed-offset", "0"),
            "cycle_penalty": hyperparams.get("cycle-penalty"),
            "alt_opt": str(hyperparams.get("alt-opt", "False")).lower() == "true",
            "dag_alm": str(hyperparams.get("dag-alm", "False")).lower() == "true",
            "theta_lr": _as_float(hyperparams.get("theta-lr")),
            "mask_lr": _as_float(hyperparams.get("mask-lr")),
            "theta_steps_per_mask": hyperparams.get("theta-steps-per-mask", "?"),
            "mask_steps_per_theta": hyperparams.get("mask-steps-per-theta", "?"),
            "variables": min_vars,
            "mask_min": mask_min,
            "mask_max": mask_max,
            "mask_diff": mask_max - mask_min,
            "mask_dir": str(mask_dir),
            **_load_result_metrics(mask_dir),
        })

    rows.sort(key=lambda row: (
        row["cycle_lambda"] if row["cycle_lambda"] is not None else float("inf"),
        str(row["mask_mode"]),
        _trial_sort_key(row["trial_index"]),
        row["theta_lr"] if row["theta_lr"] is not None else float("inf"),
        row["mask_lr"] if row["mask_lr"] is not None else float("inf"),
        str(row["theta_steps_per_mask"]),
        str(row["mask_steps_per_theta"]),
        _as_float(row.get("train_seed_offset"), default=float("inf")),
        row["run_id"],
    ))
    return rows


def _filter_rows(
        rows,
        graph_names=None,
        dimensions=None,
        mask_mode=None,
        cycle_lambda=None,
        trial=None,
        run_id=None,
        train_seed_offset=None,
        theta_lr=None,
        mask_lr=None,
        theta_steps_per_mask=None,
        mask_steps_per_theta=None,
        schedule_filters=None):
    filtered = []
    accepted_graphs = _filter_values(graph_names)
    accepted_dimensions = _filter_values(dimensions)
    for row in rows:
        if accepted_graphs is not None and str(row["graph"]) not in accepted_graphs:
            continue
        if accepted_dimensions is not None and str(row["dimension"]) not in accepted_dimensions:
            continue
        if mask_mode is not None and row["mask_mode"] != mask_mode:
            continue
        if cycle_lambda is not None and row["cycle_lambda"] != cycle_lambda:
            continue
        if trial is not None and str(row["trial_index"]) != str(trial):
            continue
        if run_id is not None and row["run_id"] != run_id:
            continue
        if train_seed_offset is not None and str(row["train_seed_offset"]) not in _filter_values(train_seed_offset):
            continue
        if theta_lr is not None and row["theta_lr"] != theta_lr:
            continue
        if mask_lr is not None and row["mask_lr"] != mask_lr:
            continue
        if theta_steps_per_mask is not None and str(row["theta_steps_per_mask"]) != str(theta_steps_per_mask):
            continue
        if mask_steps_per_theta is not None and str(row["mask_steps_per_theta"]) != str(mask_steps_per_theta):
            continue
        if schedule_filters and not any(
                str(row["theta_steps_per_mask"]) == str(theta_steps)
                and str(row["mask_steps_per_theta"]) == str(mask_steps)
                and row["theta_lr"] == schedule_theta_lr
                and row["mask_lr"] == schedule_mask_lr
                for theta_steps, mask_steps, schedule_theta_lr, schedule_mask_lr in schedule_filters):
            continue
        filtered.append(row)
    return filtered


def _select_mask_rows(
        root_dir,
        graph_names=None,
        dimensions=None,
        mask_mode=None,
        cycle_lambda=None,
        trial=None,
        run_id=None,
        train_seed_offset=None,
        theta_lr=None,
        mask_lr=None,
        theta_steps_per_mask=None,
        mask_steps_per_theta=None,
        schedule_filters=None,
        include_nonstandard=False):
    rows = load_mask_rows(root_dir, include_nonstandard=include_nonstandard)
    return _filter_rows(
        rows,
        graph_names=graph_names,
        dimensions=dimensions,
        mask_mode=mask_mode,
        cycle_lambda=cycle_lambda,
        trial=trial,
        run_id=run_id,
        train_seed_offset=train_seed_offset,
        theta_lr=theta_lr,
        mask_lr=mask_lr,
        theta_steps_per_mask=theta_steps_per_mask,
        mask_steps_per_theta=mask_steps_per_theta,
        schedule_filters=schedule_filters,
    )


def _add_heatmap(ax, matrix, variables, title, vmin, vmax, cmap):
    im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(len(variables)))
    ax.set_xticklabels(variables, fontsize=8)
    ax.set_yticks(range(len(variables)))
    ax.set_yticklabels(variables, fontsize=8)
    ax.set_xlabel("target")
    ax.set_ylabel("source")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)
    return im


def plot_mask_heatmaps(
        root_dir,
        output_path=None,
        rows=None,
        graph_names=None,
        dimensions=None,
        mask_mode=None,
        cycle_lambda=None,
        trial=None,
        run_id=None,
        train_seed_offset=None,
        theta_lr=None,
        mask_lr=None,
        theta_steps_per_mask=None,
        mask_steps_per_theta=None,
        schedule_filters=None,
        include_nonstandard=False,
        max_rows=None,
        title=None):
    if rows is None:
        rows = _select_mask_rows(
            root_dir,
            graph_names=graph_names,
            dimensions=dimensions,
            mask_mode=mask_mode,
            cycle_lambda=cycle_lambda,
            trial=trial,
            run_id=run_id,
            train_seed_offset=train_seed_offset,
            theta_lr=theta_lr,
            mask_lr=mask_lr,
            theta_steps_per_mask=theta_steps_per_mask,
            mask_steps_per_theta=mask_steps_per_theta,
            schedule_filters=schedule_filters,
            include_nonstandard=include_nonstandard,
        )
    else:
        rows = list(rows)
    if max_rows is not None:
        rows = rows[:max_rows]
    if not rows:
        raise ValueError("No mask rows matched the requested filters")

    variables = rows[0]["variables"]
    for row in rows:
        if row["variables"] != variables:
            raise ValueError("Cannot plot masks with different variable orders in one figure")

    n_rows = len(rows)
    fig_width = 13
    fig_height = max(3.5, 2.6 * n_rows)
    fig, axes = plt.subplots(n_rows, 3, figsize=(fig_width, fig_height), squeeze=False)

    mask_images = []
    diff_images = []
    for i, row in enumerate(rows):
        row_title = _row_label(row)
        axes[i, 0].text(
            -0.72,
            0.5,
            row_title,
            transform=axes[i, 0].transAxes,
            rotation=0,
            va="center",
            ha="right",
            fontsize=9,
        )
        mask_images.append(_add_heatmap(
            axes[i, 0],
            row["mask_min"],
            variables,
            "mask_min",
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
        ))
        mask_images.append(_add_heatmap(
            axes[i, 1],
            row["mask_max"],
            variables,
            "mask_max",
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
        ))
        diff_abs = max(0.1, float(np.abs(row["mask_diff"]).max()))
        diff_images.append(_add_heatmap(
            axes[i, 2],
            row["mask_diff"],
            variables,
            "max - min",
            vmin=-diff_abs,
            vmax=diff_abs,
            cmap="coolwarm",
        ))

    fig.colorbar(mask_images[-1], ax=axes[:, :2], fraction=0.025, pad=0.02, label="mask value")
    fig.colorbar(diff_images[-1], ax=axes[:, 2], fraction=0.04, pad=0.02, label="mask_max - mask_min")

    if title is None:
        filters = []
        if mask_mode is not None:
            filters.append(f"mask={mask_mode}")
        if dimensions is not None:
            filters.append(f"dim={','.join(sorted(_filter_values(dimensions)))}")
        if cycle_lambda is not None:
            filters.append(f"lambda={cycle_lambda:g}")
        if trial is not None:
            filters.append(f"trial={trial}")
        if run_id is not None:
            filters.append(f"run={run_id}")
        if train_seed_offset is not None:
            filters.append(f"seed={','.join(sorted(_filter_values(train_seed_offset)))}")
        if theta_lr is not None:
            filters.append(f"theta_lr={theta_lr:g}")
        if mask_lr is not None:
            filters.append(f"mask_lr={mask_lr:g}")
        if theta_steps_per_mask is not None or mask_steps_per_theta is not None:
            filters.append(f"steps={theta_steps_per_mask or '?'}:{mask_steps_per_theta or '?'}")
        if schedule_filters:
            labels = [
                f"{theta_steps}:{mask_steps} tlr={schedule_theta_lr:g} mlr={schedule_mask_lr:g}"
                for theta_steps, mask_steps, schedule_theta_lr, schedule_mask_lr in schedule_filters
            ]
            filters.append("schedule=" + " | ".join(labels))
        title = "Mask Heatmaps"
        if filters:
            title += " (" + ", ".join(filters) + ")"
    fig.suptitle(title, y=0.995, fontsize=14)
    fig.subplots_adjust(left=0.24, right=0.88, top=0.93, bottom=0.06, wspace=0.35, hspace=0.45)

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, axes


def _trial_output_path(output_path, root_dir, trial):
    trial_label = str(trial).replace("/", "_")
    if output_path is None:
        output_path = Path(root_dir) / "mask_heatmaps.png"
    else:
        output_path = Path(output_path)
    return str(output_path.with_name(f"{output_path.stem}_trial{trial_label}{output_path.suffix}"))


def _graph_trial_output_path(output_path, root_dir, graph, trial):
    graph_label = _sanitize_output_label(graph)
    trial_label = _sanitize_output_label(trial)
    if output_path is None:
        output_path = Path(root_dir) / "mask_heatmaps.png"
    else:
        output_path = Path(output_path)
    return str(output_path.with_name(
        f"{output_path.stem}_graph-{graph_label}_trial-{trial_label}{output_path.suffix}"))


def plot_mask_heatmaps_by_trial(
        root_dir,
        output_path=None,
        graph_names=None,
        dimensions=None,
        mask_mode=None,
        cycle_lambda=None,
        run_id=None,
        train_seed_offset=None,
        theta_lr=None,
        mask_lr=None,
        theta_steps_per_mask=None,
        mask_steps_per_theta=None,
        schedule_filters=None,
        include_nonstandard=False,
        max_rows=None,
        title=None):
    rows = _select_mask_rows(
        root_dir,
        graph_names=graph_names,
        dimensions=dimensions,
        mask_mode=mask_mode,
        cycle_lambda=cycle_lambda,
        run_id=run_id,
        train_seed_offset=train_seed_offset,
        theta_lr=theta_lr,
        mask_lr=mask_lr,
        theta_steps_per_mask=theta_steps_per_mask,
        mask_steps_per_theta=mask_steps_per_theta,
        schedule_filters=schedule_filters,
        include_nonstandard=include_nonstandard,
    )
    if not rows:
        raise ValueError("No mask rows matched the requested filters")

    trials = sorted({row["trial_index"] for row in rows}, key=_trial_sort_key)
    outputs = []
    for trial in trials:
        trial_rows = [row for row in rows if row["trial_index"] == trial]
        trial_output = _trial_output_path(output_path, root_dir, trial)
        trial_title = title
        if trial_title is None:
            trial_title = f"Mask Heatmaps (trial={trial})"
        fig, _axes = plot_mask_heatmaps(
            root_dir,
            output_path=trial_output,
            rows=trial_rows,
            graph_names=graph_names,
            dimensions=dimensions,
            mask_mode=mask_mode,
            cycle_lambda=cycle_lambda,
            trial=trial,
            run_id=run_id,
            train_seed_offset=train_seed_offset,
            theta_lr=theta_lr,
            mask_lr=mask_lr,
            theta_steps_per_mask=theta_steps_per_mask,
            mask_steps_per_theta=mask_steps_per_theta,
            schedule_filters=schedule_filters,
            include_nonstandard=include_nonstandard,
            max_rows=max_rows,
            title=trial_title,
        )
        plt.close(fig)
        outputs.append(trial_output)
    return outputs


def plot_mask_heatmaps_by_graph_trial(
        root_dir,
        output_path=None,
        graph_names=None,
        dimensions=None,
        mask_mode=None,
        cycle_lambda=None,
        run_id=None,
        train_seed_offset=None,
        theta_lr=None,
        mask_lr=None,
        theta_steps_per_mask=None,
        mask_steps_per_theta=None,
        schedule_filters=None,
        include_nonstandard=False,
        max_rows=None,
        title=None):
    rows = _select_mask_rows(
        root_dir,
        graph_names=graph_names,
        dimensions=dimensions,
        mask_mode=mask_mode,
        cycle_lambda=cycle_lambda,
        run_id=run_id,
        train_seed_offset=train_seed_offset,
        theta_lr=theta_lr,
        mask_lr=mask_lr,
        theta_steps_per_mask=theta_steps_per_mask,
        mask_steps_per_theta=mask_steps_per_theta,
        schedule_filters=schedule_filters,
        include_nonstandard=include_nonstandard,
    )
    if not rows:
        raise ValueError("No mask rows matched the requested filters")

    groups = {}
    for row in rows:
        groups.setdefault((row["graph"], row["trial_index"]), []).append(row)

    outputs = []
    for graph, trial in sorted(groups, key=lambda key: (_graph_sort_key(key[0]), _trial_sort_key(key[1]))):
        group_rows = groups[(graph, trial)]
        group_output = _graph_trial_output_path(output_path, root_dir, graph, trial)
        group_title = title
        if group_title is None:
            group_title = f"Mask Heatmaps (graph={graph}, trial={trial})"
        fig, _axes = plot_mask_heatmaps(
            root_dir,
            output_path=group_output,
            rows=group_rows,
            graph_names=[graph],
            dimensions=dimensions,
            mask_mode=mask_mode,
            cycle_lambda=cycle_lambda,
            trial=trial,
            run_id=run_id,
            train_seed_offset=train_seed_offset,
            theta_lr=theta_lr,
            mask_lr=mask_lr,
            theta_steps_per_mask=theta_steps_per_mask,
            mask_steps_per_theta=mask_steps_per_theta,
            schedule_filters=schedule_filters,
            include_nonstandard=include_nonstandard,
            max_rows=max_rows,
            title=group_title,
        )
        plt.close(fig)
        outputs.append(group_output)
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Plot learned min/max mask heatmaps for bound runs.")
    parser.add_argument("root_dir", help="Root directory to search recursively for mask_min.json files")
    parser.add_argument("--output", help="Defaults to <root_dir>/mask_heatmaps.png")
    parser.add_argument("--graph", action="append", dest="graph_names",
                        help="Filter by graph name. May be repeated.")
    parser.add_argument("--dimension", "--dim", action="append", dest="dimensions",
                        help="Filter by dimension parsed from dim=<value> in the run directory. May be repeated.")
    parser.add_argument("--mask-mode", help="Filter by mask mode, e.g. gate, multiply, st-gate")
    parser.add_argument("--cycle-lambda", type=float, help="Filter by cycle lambda")
    parser.add_argument("--trial", help="Filter by trial index")
    parser.add_argument("--run-id", help="Filter by full run hash")
    parser.add_argument("--train-seed-offset", action="append", help="Filter by train seed offset. May be repeated.")
    parser.add_argument("--theta-lr", type=float, help="Filter by theta learning rate")
    parser.add_argument("--mask-lr", type=float, help="Filter by mask learning rate")
    parser.add_argument("--theta-steps-per-mask", help="Filter by theta steps per mask update")
    parser.add_argument("--mask-steps-per-theta", help="Filter by mask steps per theta update")
    parser.add_argument(
        "--schedule-filter",
        action="append",
        type=_parse_schedule_filter,
        dest="schedule_filters",
        help=(
            "Filter by one theta/mask schedule as THETA_STEPS:MASK_STEPS:THETA_LR:MASK_LR. "
            "May be repeated to keep multiple schedules."
        ),
    )
    parser.add_argument("--include-nonstandard", action="store_true",
                        help="Deprecated compatibility flag; nonstandard runs are included by default")
    parser.add_argument("--standard-only", action="store_true",
                        help="Only include runs without alt-opt or dag-alm")
    parser.add_argument("--by-graph-trial", action="store_true",
                        help="Save one plot per graph/trial group. This is automatic when multiple graphs match.")
    parser.add_argument("--max-rows", type=int, help="Maximum number of matching runs to plot")
    parser.add_argument("--title", help="Optional figure title")
    args = parser.parse_args()

    output = args.output or str(Path(args.root_dir) / "mask_heatmaps.png")
    include_nonstandard = args.include_nonstandard or not args.standard_only
    rows = _select_mask_rows(
        args.root_dir,
        graph_names=args.graph_names,
        dimensions=args.dimensions,
        mask_mode=args.mask_mode,
        cycle_lambda=args.cycle_lambda,
        trial=args.trial,
        run_id=args.run_id,
        train_seed_offset=args.train_seed_offset,
        theta_lr=args.theta_lr,
        mask_lr=args.mask_lr,
        theta_steps_per_mask=args.theta_steps_per_mask,
        mask_steps_per_theta=args.mask_steps_per_theta,
        schedule_filters=args.schedule_filters,
        include_nonstandard=include_nonstandard,
    )
    if not rows:
        raise ValueError("No mask rows matched the requested filters")

    graphs = sorted({row["graph"] for row in rows}, key=_graph_sort_key)
    trials = sorted({row["trial_index"] for row in rows}, key=_trial_sort_key)
    if args.by_graph_trial or args.trial is None or len(graphs) > 1:
        outputs = []
        groups = {}
        for row in rows:
            groups.setdefault((row["graph"], row["trial_index"]), []).append(row)
        for graph, trial in sorted(groups, key=lambda key: (_graph_sort_key(key[0]), _trial_sort_key(key[1]))):
            group_rows = groups[(graph, trial)]
            group_output = _graph_trial_output_path(output, args.root_dir, graph, trial)
            group_title = args.title or f"Mask Heatmaps (graph={graph}, trial={trial})"
            fig, _axes = plot_mask_heatmaps(
                args.root_dir,
                output_path=group_output,
                rows=group_rows,
                graph_names=[graph],
                dimensions=args.dimensions,
                mask_mode=args.mask_mode,
                cycle_lambda=args.cycle_lambda,
                trial=trial,
                run_id=args.run_id,
                train_seed_offset=args.train_seed_offset,
                theta_lr=args.theta_lr,
                mask_lr=args.mask_lr,
                theta_steps_per_mask=args.theta_steps_per_mask,
                mask_steps_per_theta=args.mask_steps_per_theta,
                schedule_filters=args.schedule_filters,
                include_nonstandard=include_nonstandard,
                max_rows=args.max_rows,
                title=group_title,
            )
            plt.close(fig)
            outputs.append(group_output)
        print("Saved mask heatmaps to:")
        for path in outputs:
            print(path)
    elif args.trial is None and len(trials) > 1:
        outputs = []
        for trial in trials:
            trial_rows = [row for row in rows if row["trial_index"] == trial]
            trial_output = _trial_output_path(output, args.root_dir, trial)
            trial_title = args.title or f"Mask Heatmaps (trial={trial})"
            fig, _axes = plot_mask_heatmaps(
                args.root_dir,
                output_path=trial_output,
                rows=trial_rows,
                graph_names=args.graph_names,
                dimensions=args.dimensions,
                mask_mode=args.mask_mode,
                cycle_lambda=args.cycle_lambda,
                trial=trial,
                run_id=args.run_id,
                train_seed_offset=args.train_seed_offset,
                theta_lr=args.theta_lr,
                mask_lr=args.mask_lr,
                theta_steps_per_mask=args.theta_steps_per_mask,
                mask_steps_per_theta=args.mask_steps_per_theta,
                schedule_filters=args.schedule_filters,
                include_nonstandard=include_nonstandard,
                max_rows=args.max_rows,
                title=trial_title,
            )
            plt.close(fig)
            outputs.append(trial_output)
        print("Saved mask heatmaps to:")
        for path in outputs:
            print(path)
    else:
        plot_mask_heatmaps(
            args.root_dir,
            output_path=output,
            rows=rows,
            graph_names=args.graph_names,
            dimensions=args.dimensions,
            mask_mode=args.mask_mode,
            cycle_lambda=args.cycle_lambda,
            trial=args.trial,
            run_id=args.run_id,
            train_seed_offset=args.train_seed_offset,
            theta_lr=args.theta_lr,
            mask_lr=args.mask_lr,
            theta_steps_per_mask=args.theta_steps_per_mask,
            mask_steps_per_theta=args.mask_steps_per_theta,
            schedule_filters=args.schedule_filters,
            include_nonstandard=include_nonstandard,
            max_rows=args.max_rows,
            title=args.title,
        )
        print(f"Saved mask heatmaps to {output}")


if __name__ == "__main__":
    main()
