import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

TRIAL_LABEL_Y = -0.82
PLOT_BOTTOM = 0.42
PLOT_RIGHT = 0.85
X_LABEL_PAD = 132
STANDARD_GROUP_SPACING = 1.0
NONSTANDARD_GROUP_SPACING = 1.45
POOL_NONE = "none"
POOL_ENDPOINTS = "endpoints"
POOL_ALL_OUTPUTS = "all-outputs"


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


def _extract_graph(run_dir):
    for piece in run_dir.split("-"):
        if piece.startswith("graph="):
            return piece[len("graph="):]
    return "?"


def _extract_dimension_label(run_dir):
    for piece in run_dir.split("-"):
        if piece.startswith("dim="):
            return piece[len("dim="):]
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


def _format_optional(value):
    if value in (None, "", "None"):
        return "?"
    return str(value)


def _format_init_label(mode, value, init_range):
    mode = _format_optional(mode)
    value = _format_optional(value)
    init_range = _format_optional(init_range)
    if value != "?":
        return f"{mode}:{value}"
    if init_range != "?":
        return f"{mode}:{init_range}"
    return mode


def _format_coupling_label(coupled_edges):
    if coupled_edges in (None, "", "[]", "None"):
        return "off"
    return "on"


def _cycle_query_lambda_label(cycle_lambda, min_lambda, max_lambda):
    return " ".join([
        f"c={_format_float(cycle_lambda)}",
        f"qmin={_format_float(min_lambda)}",
        f"qmax={_format_float(max_lambda)}",
    ])


def _group_label(group):
    cycle_lambda, mask_mode, _trial_index, max_lambda, min_lambda = group[:5]
    lines = [
        str(mask_mode),
        _cycle_query_lambda_label(cycle_lambda, min_lambda, max_lambda),
    ]
    if len(group) > 5:
        (
            alt_opt,
            dag_alm,
            theta_lr,
            mask_lr,
            theta_steps,
            mask_steps,
            mask_init_mode,
            mask_init_value,
            mask_init_range,
            mask_fit_loss_weight,
            mask_coupled_edges,
            query_update_target,
            selection_query_lambda,
        ) = group[5:]
        parts = []
        if alt_opt:
            parts.append("alt")
        if dag_alm:
            parts.append("alm")
        parts.extend([
            f"tlr={_format_float(theta_lr)}",
            f"mlr={_format_float(mask_lr)}",
            f"s={theta_steps}:{mask_steps}",
        ])
        lines.append(" ".join(parts))
        lines.append(
            " ".join([
                f"init={_format_init_label(mask_init_mode, mask_init_value, mask_init_range)}",
                f"mfit={_format_float(mask_fit_loss_weight)}",
            ])
        )
        lines.append(
            " ".join([
                f"coup={_format_coupling_label(mask_coupled_edges)}",
                f"q={_format_optional(query_update_target)}",
                f"sel={_format_float(selection_query_lambda)}",
            ])
        )
    return "\n".join(lines)


def _trial_sort_key(trial_index):
    return int(trial_index) if str(trial_index).isdigit() else str(trial_index)


def _row_offsets(rows, max_span=0.62):
    if len(rows) <= 1:
        return [0.0] * len(rows)

    step = max_span / (len(rows) - 1)
    return [-max_span / 2 + i * step for i in range(len(rows))]


def _group_spacing(include_nonstandard=False):
    return NONSTANDARD_GROUP_SPACING if include_nonstandard else STANDARD_GROUP_SPACING


def _pooled_bound_values(group_rows, pool_train_seeds):
    if pool_train_seeds == POOL_NONE:
        return None

    if pool_train_seeds == POOL_ENDPOINTS:
        return (
            min(row["ncm_min"] for row in group_rows),
            max(row["ncm_max"] for row in group_rows),
        )

    if pool_train_seeds == POOL_ALL_OUTPUTS:
        values = [
            value
            for row in group_rows
            for value in (row["ncm_min"], row["ncm_max"])
        ]
        return min(values), max(values)

    raise ValueError(f"unknown pooling mode: {pool_train_seeds}")


def _standard_rows(rows):
    return [
        row for row in rows
        if row.get("standard_bound_run", True)
    ]


def _matches_any(value, accepted):
    if accepted is None:
        return True
    return str(value) in {str(item) for item in accepted}


def _matches_schedule_filter(row, schedule_filter):
    theta_steps, mask_steps, theta_lr, mask_lr = schedule_filter
    return (
        str(row["theta_steps_per_mask"]) == str(theta_steps)
        and str(row["mask_steps_per_theta"]) == str(mask_steps)
        and row["theta_lr"] == theta_lr
        and row["mask_lr"] == mask_lr
    )


def _filter_rows(
        rows,
        include_nonstandard=False,
        graph_names=None,
        dimension_labels=None,
        train_seed_offsets=None,
        theta_lr=None,
        mask_lr=None,
        theta_steps_per_mask=None,
        mask_steps_per_theta=None,
        schedule_filters=None):
    if include_nonstandard:
        filtered = list(rows)
    else:
        filtered = _standard_rows(rows)

    narrowed = []
    for row in filtered:
        if not _matches_any(row["graph"], graph_names):
            continue
        if not _matches_any(row["dimension_label"], dimension_labels):
            continue
        if not _matches_any(row["train_seed_offset"], train_seed_offsets):
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
                _matches_schedule_filter(row, schedule_filter)
                for schedule_filter in schedule_filters):
            continue
        narrowed.append(row)
    return narrowed


def _filter_label(
        graph_names=None,
        dimension_labels=None,
        train_seed_offsets=None,
        theta_lr=None,
        mask_lr=None,
        theta_steps_per_mask=None,
        mask_steps_per_theta=None,
        schedule_filters=None):
    filters = []
    if graph_names:
        filters.append(f"graph={','.join(str(graph) for graph in graph_names)}")
    if dimension_labels:
        filters.append(f"dim={','.join(str(dim) for dim in dimension_labels)}")
    if train_seed_offsets:
        filters.append(f"seed={','.join(str(offset) for offset in train_seed_offsets)}")
    if theta_lr is not None:
        filters.append(f"theta_lr={theta_lr:g}")
    if mask_lr is not None:
        filters.append(f"mask_lr={mask_lr:g}")
    if theta_steps_per_mask is not None or mask_steps_per_theta is not None:
        filters.append(f"steps={theta_steps_per_mask or '?'}:{mask_steps_per_theta or '?'}")
    if schedule_filters:
        labels = [
            f"{theta_steps}:{mask_steps} tlr={theta_lr:g} mlr={mask_lr:g}"
            for theta_steps, mask_steps, theta_lr, mask_lr in schedule_filters
        ]
        filters.append("schedule=" + " | ".join(labels))
    if not filters:
        return ""
    return " (" + ", ".join(filters) + ")"


def _group_rows(rows, include_nonstandard=False):
    grouped = {}
    for row in rows:
        group = (
            row["cycle_lambda"],
            row["mask_mode"],
            row["trial_index"],
            row["max_lambda"],
            row["min_lambda"],
        )
        if include_nonstandard:
            group = group + (
                row["alt_opt"],
                row["dag_alm"],
                row["theta_lr"],
                row["mask_lr"],
                row["theta_steps_per_mask"],
                row["mask_steps_per_theta"],
                row["mask_init_mode"],
                row["mask_init_value"],
                row["mask_init_range"],
                row["mask_fit_loss_weight"],
                row["mask_coupled_edges"],
                row["query_update_target"],
                row["selection_query_lambda"],
            )
        grouped.setdefault(group, []).append(row)
    return grouped


def _available_graphs(rows, graph_names=None):
    return sorted({
        row["graph"] for row in rows
        if _matches_any(row["graph"], graph_names)
    })


def _sort_groups(groups):
    return sorted(
        groups,
        key=lambda group: (
            _trial_sort_key(group[2]),
            str(group[1]),
            float("inf") if group[0] is None else group[0],
            float("inf") if group[3] is None else group[3],
            float("inf") if group[4] is None else group[4],
            tuple(group[5:]),
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
            TRIAL_LABEL_Y,
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


def _extract_bound_error_series(results):
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
    standard_keys = {
        "min_total_true_KL": "min_total_true_KL",
        "max_total_true_KL": "max_total_true_KL",
        "min_total_dat_KL": "min_total_dat_KL",
        "max_total_dat_KL": "max_total_dat_KL",
    }
    enum_keys = {
        "enum_min_total_true_KL": "min_total_true_KL",
        "enum_max_total_true_KL": "max_total_true_KL",
        "enum_min_total_dat_KL": "min_total_dat_KL",
        "enum_max_total_dat_KL": "max_total_dat_KL",
    }

    if all(key in results for key in standard_keys):
        return {alias: float(results[key]) for key, alias in standard_keys.items()}
    if all(key in results for key in enum_keys):
        return {alias: float(results[key]) for key, alias in enum_keys.items()}
    return None


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
            "graph": _extract_graph(run_dir),
            "dimension_label": _extract_dimension_label(run_dir),
            "trial_index": _extract_trial_index(run_dir),
            "cycle_lambda": _as_float(hyperparams.get("cycle-lambda")),
            "max_lambda": _as_float(hyperparams.get("max-lambda")),
            "min_lambda": _as_float(hyperparams.get("min-lambda")),
            "mask_mode": hyperparams.get("mask-mode", "?"),
            "train_seed_offset": hyperparams.get("train-seed-offset", "0"),
            "cycle_penalty": hyperparams.get("cycle-penalty"),
            "alt_opt": _as_bool(hyperparams.get("alt-opt")),
            "dag_alm": _as_bool(hyperparams.get("dag-alm")),
            "theta_lr": _as_float(hyperparams.get("theta-lr")),
            "mask_lr": _as_float(hyperparams.get("mask-lr")),
            "theta_steps_per_mask": hyperparams.get("theta-steps-per-mask", "?"),
            "mask_steps_per_theta": hyperparams.get("mask-steps-per-theta", "?"),
            "mask_init_mode": hyperparams.get("mask-init-mode"),
            "mask_init_value": _as_float(hyperparams.get("mask-init-value")),
            "mask_init_range": hyperparams.get("mask-init-range"),
            "mask_fit_loss_weight": _as_float(hyperparams.get("mask-fit-loss-weight")),
            "mask_coupled_edges": hyperparams.get("mask-coupled-edges"),
            "query_update_target": hyperparams.get("query-update-target"),
            "selection_query_lambda": _as_float(hyperparams.get("selection-query-lambda")),
            "standard_bound_run": _standard_bound_run(hyperparams),
            "results_path": str(results_path),
            **series,
            **kl_series,
            **bound_error_series,
        })

    rows.sort(key=lambda row: row["label"])
    return rows


def _plot_bound_rows_on_ax(
        ax,
        rows,
        root_dir,
        include_nonstandard=False,
        graph_names=None,
        dimension_labels=None,
        train_seed_offsets=None,
        theta_lr=None,
        mask_lr=None,
        theta_steps_per_mask=None,
        mask_steps_per_theta=None,
        schedule_filters=None,
        pool_train_seeds=POOL_NONE,
        title=None,
        show_legend=True,
        show_title=True):
    grouped = _group_rows(rows, include_nonstandard=include_nonstandard)
    groups = _sort_groups(grouped)
    n_runs = len(groups)

    true_color = "#1f77b4"
    ncm_min_color = "#d62728"
    ncm_max_color = "#ff7f0e"
    half_width = 0.28
    spacing = _group_spacing(include_nonstandard=include_nonstandard)
    xs = [i * spacing for i in range(n_runs)]

    for x in xs:
        group_index = round(x / spacing)
        if group_index % 2 == 0:
            ax.axvspan(x - spacing / 2, x + spacing / 2, color="0.96", zorder=0)

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

        pooled_values = _pooled_bound_values(group_rows, pool_train_seeds)
        if pooled_values is None:
            for offset, row in zip(offsets, group_rows):
                ax.scatter([x + offset], [row["ncm_min"]], color=ncm_min_color, marker="o", s=34,
                           edgecolors="black", linewidths=0.3, zorder=4)
                ax.scatter([x + offset], [row["ncm_max"]], color=ncm_max_color, marker="o", s=34,
                           edgecolors="black", linewidths=0.3, zorder=4)
        else:
            pooled_min, pooled_max = pooled_values
            ax.scatter([x], [pooled_min], color=ncm_min_color, marker="D", s=46,
                       edgecolors="black", linewidths=0.4, zorder=4)
            ax.scatter([x], [pooled_max], color=ncm_max_color, marker="D", s=46,
                       edgecolors="black", linewidths=0.4, zorder=4)

    ax.set_xlim(-0.8 * spacing, xs[-1] + 0.8 * spacing)
    ax.set_xticks(xs)
    ax.set_xticklabels([_group_label(group) for group in groups], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Cycle lambda / query min-max lambda / mask mode, grouped by trial", labelpad=X_LABEL_PAD)
    ax.set_ylabel("Query value")
    if show_title:
        filter_label = _filter_label(
            graph_names=graph_names,
            dimension_labels=dimension_labels,
            train_seed_offsets=train_seed_offsets,
            theta_lr=theta_lr,
            mask_lr=mask_lr,
            theta_steps_per_mask=theta_steps_per_mask,
            mask_steps_per_theta=mask_steps_per_theta,
            schedule_filters=schedule_filters,
        )
        ax.set_title(title or f"True Bounds and NCM Seed Results ({Path(root_dir)}){filter_label}")
    ax.grid(axis="y", alpha=0.3, zorder=1)
    _add_trial_labels(ax, trial_groups)

    legend_handles = [
        Line2D([0], [0], color=true_color, lw=4.0, label="True min/max"),
        Line2D([0], [0], marker="D" if pool_train_seeds != POOL_NONE else "o",
               color="black", markerfacecolor=ncm_min_color,
               linestyle="", markersize=7, label="Pooled NCM min" if pool_train_seeds != POOL_NONE else "NCM min"),
        Line2D([0], [0], marker="D" if pool_train_seeds != POOL_NONE else "o",
               color="black", markerfacecolor=ncm_max_color,
               linestyle="", markersize=7, label="Pooled NCM max" if pool_train_seeds != POOL_NONE else "NCM max"),
    ]
    if show_legend:
        ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    return legend_handles


def plot_bound_results(
        root_dir,
        output_path=None,
        title=None,
        figsize=None,
        include_nonstandard=False,
        graph_names=None,
        dimension_labels=None,
        train_seed_offsets=None,
        theta_lr=None,
        mask_lr=None,
        theta_steps_per_mask=None,
        mask_steps_per_theta=None,
        schedule_filters=None,
        subplot_graphs=False,
        pool_train_seeds=POOL_NONE):
    rows = _filter_rows(
        load_bound_results(root_dir),
        include_nonstandard=include_nonstandard,
        graph_names=graph_names,
        dimension_labels=dimension_labels,
        train_seed_offsets=train_seed_offsets,
        theta_lr=theta_lr,
        mask_lr=mask_lr,
        theta_steps_per_mask=theta_steps_per_mask,
        mask_steps_per_theta=mask_steps_per_theta,
        schedule_filters=schedule_filters,
    )
    if not rows:
        raise ValueError(f"No completed bound results found under {root_dir}")

    if subplot_graphs:
        graphs = _available_graphs(rows, graph_names=graph_names)
        graph_rows = [(graph, [row for row in rows if row["graph"] == graph]) for graph in graphs]
        max_groups = max(
            len(_group_rows(cur_rows, include_nonstandard=include_nonstandard))
            for _graph, cur_rows in graph_rows
        )
        if figsize is None:
            width_per_run = 1.05 if include_nonstandard else 0.75
            height_per_graph = 6 if include_nonstandard else 5
            figsize = (max(12, width_per_run * max_groups), max(6, height_per_graph * len(graph_rows)))

        fig, axes = plt.subplots(len(graph_rows), 1, figsize=figsize, squeeze=False)
        legend_handles = None
        for ax, (graph, cur_rows) in zip(axes[:, 0], graph_rows):
            legend_handles = _plot_bound_rows_on_ax(
                ax,
                cur_rows,
                root_dir,
                include_nonstandard=include_nonstandard,
                graph_names=[graph],
                dimension_labels=dimension_labels,
                train_seed_offsets=train_seed_offsets,
                theta_lr=theta_lr,
                mask_lr=mask_lr,
                theta_steps_per_mask=theta_steps_per_mask,
                mask_steps_per_theta=mask_steps_per_theta,
                schedule_filters=schedule_filters,
                pool_train_seeds=pool_train_seeds,
                title=f"Graph: {graph}",
                show_legend=False,
            )

        filter_label = _filter_label(
            graph_names=graph_names,
            dimension_labels=dimension_labels,
            train_seed_offsets=train_seed_offsets,
            theta_lr=theta_lr,
            mask_lr=mask_lr,
            theta_steps_per_mask=theta_steps_per_mask,
            mask_steps_per_theta=mask_steps_per_theta,
            schedule_filters=schedule_filters,
        )
        fig.suptitle(title or f"True Bounds and NCM Seed Results ({Path(root_dir)}){filter_label}")
        fig.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(PLOT_RIGHT + 0.02, 0.96), borderaxespad=0.0)
        fig.subplots_adjust(bottom=PLOT_BOTTOM / len(graph_rows), right=PLOT_RIGHT, hspace=0.9)

        if output_path is not None:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
        return fig, axes[:, 0]

    grouped = _group_rows(rows, include_nonstandard=include_nonstandard)
    groups = _sort_groups(grouped)
    n_runs = len(groups)
    if figsize is None:
        width_per_run = 1.05 if include_nonstandard else 0.75
        height = 8 if include_nonstandard else 6
        figsize = (max(12, width_per_run * n_runs), height)

    fig, ax = plt.subplots(figsize=figsize)
    _plot_bound_rows_on_ax(
        ax,
        rows,
        root_dir,
        include_nonstandard=include_nonstandard,
        graph_names=graph_names,
        dimension_labels=dimension_labels,
        train_seed_offsets=train_seed_offsets,
        theta_lr=theta_lr,
        mask_lr=mask_lr,
        theta_steps_per_mask=theta_steps_per_mask,
        mask_steps_per_theta=mask_steps_per_theta,
        schedule_filters=schedule_filters,
        pool_train_seeds=pool_train_seeds,
        title=title,
    )
    fig.subplots_adjust(bottom=PLOT_BOTTOM, right=PLOT_RIGHT)

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_kl_results(
        root_dir,
        output_path=None,
        title=None,
        figsize=None,
        include_nonstandard=False,
        graph_names=None,
        dimension_labels=None,
        train_seed_offsets=None,
        theta_lr=None,
        mask_lr=None,
        theta_steps_per_mask=None,
        mask_steps_per_theta=None,
        schedule_filters=None):
    rows = [
        row for row in _filter_rows(
            load_bound_results(root_dir),
            include_nonstandard=include_nonstandard,
            graph_names=graph_names,
            dimension_labels=dimension_labels,
            train_seed_offsets=train_seed_offsets,
            theta_lr=theta_lr,
            mask_lr=mask_lr,
            theta_steps_per_mask=theta_steps_per_mask,
            mask_steps_per_theta=mask_steps_per_theta,
            schedule_filters=schedule_filters,
        )
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

    grouped = _group_rows(rows, include_nonstandard=include_nonstandard)
    groups = _sort_groups(grouped)
    n_runs = len(groups)
    if figsize is None:
        width_per_run = 1.05 if include_nonstandard else 0.75
        height = 13 if include_nonstandard else 10
        figsize = (max(12, width_per_run * n_runs), height)

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    min_color = "#d62728"
    max_color = "#ff7f0e"
    spacing = _group_spacing(include_nonstandard=include_nonstandard)
    xs = [i * spacing for i in range(n_runs)]

    metric_specs = [
        (axes[0], "min_total_true_KL", "max_total_true_KL", "True KL"),
        (axes[1], "min_total_dat_KL", "max_total_dat_KL", "Data KL"),
    ]

    for ax, min_key, max_key, ylabel in metric_specs:
        for x in xs:
            group_index = round(x / spacing)
            if group_index % 2 == 0:
                ax.axvspan(x - spacing / 2, x + spacing / 2, color="0.96", zorder=0)

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

        ax.set_xlim(-0.8 * spacing, xs[-1] + 0.8 * spacing)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3, zorder=1)

    filter_label = _filter_label(
        graph_names=graph_names,
        dimension_labels=dimension_labels,
        train_seed_offsets=train_seed_offsets,
        theta_lr=theta_lr,
        mask_lr=mask_lr,
        theta_steps_per_mask=theta_steps_per_mask,
        mask_steps_per_theta=mask_steps_per_theta,
        schedule_filters=schedule_filters,
    )
    axes[0].set_title(title or f"Min/Max NCM KL Divergence ({Path(root_dir)}){filter_label}")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([_group_label(group) for group in groups], rotation=45, ha="right", fontsize=8)
    axes[1].set_xlabel("Cycle lambda / query min-max lambda / mask mode, grouped by trial", labelpad=X_LABEL_PAD)
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

    fig.subplots_adjust(bottom=PLOT_BOTTOM, right=PLOT_RIGHT, hspace=0.12)

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


def _parse_schedule_filter(value):
    pieces = value.replace(",", ":").split(":")
    if len(pieces) != 4:
        raise argparse.ArgumentTypeError(
            "expected THETA_STEPS:MASK_STEPS:THETA_LR:MASK_LR, "
            "for example 5:1:4e-3:4e-3"
        )
    theta_steps, mask_steps, theta_lr, mask_lr = pieces
    try:
        return theta_steps, mask_steps, float(theta_lr), float(mask_lr)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "theta/mask learning rates in schedule filters must be numeric"
        ) from exc


def _path_with_graph(path, graph):
    path = Path(path)
    return str(path.with_name(f"{path.stem}_{graph}{path.suffix}"))


def _print_summary_table(
        root_dir,
        include_nonstandard=False,
        graph_names=None,
        dimension_labels=None,
        train_seed_offsets=None,
        theta_lr=None,
        mask_lr=None,
        theta_steps_per_mask=None,
        mask_steps_per_theta=None,
        schedule_filters=None):
    rows = [
        row for row in _filter_rows(
            load_bound_results(root_dir),
            include_nonstandard=include_nonstandard,
            graph_names=graph_names,
            dimension_labels=dimension_labels,
            train_seed_offsets=train_seed_offsets,
            theta_lr=theta_lr,
            mask_lr=mask_lr,
            theta_steps_per_mask=theta_steps_per_mask,
            mask_steps_per_theta=mask_steps_per_theta,
            schedule_filters=schedule_filters,
        )
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
        row_kind = "rows" if include_nonstandard else "standard rows"
        print(f"No {row_kind} available for summary table.")
        return

    grouped = _group_rows(rows, include_nonstandard=include_nonstandard)
    groups = _sort_groups(grouped)
    headers = [
        "trial",
        "mask",
        "cycle_lambda",
        "max_lambda",
        "min_lambda",
        "alt_opt",
        "dag_alm",
        "theta_lr",
        "mask_lr",
        "theta_steps",
        "mask_steps",
        "mask_init",
        "mask_fit_loss_weight",
        "coupled",
        "query_update_target",
        "selection_query_lambda",
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
    table_label = "all runs" if include_nonstandard else "standard runs only"
    print(f"Summary table ({table_label})")
    print("\t".join(headers))
    for group in groups:
        group_rows = grouped[group]
        cycle_lambda, mask_mode, trial_index, max_lambda, min_lambda = group[:5]
        if len(group) > 5:
            (
                alt_opt,
                dag_alm,
                theta_lr,
                mask_lr,
                theta_steps,
                mask_steps,
                mask_init_mode,
                mask_init_value,
                mask_init_range,
                mask_fit_loss_weight,
                mask_coupled_edges,
                query_update_target,
                selection_query_lambda,
            ) = group[5:]
        else:
            (
                alt_opt,
                dag_alm,
                theta_lr,
                mask_lr,
                theta_steps,
                mask_steps,
                mask_init_mode,
                mask_init_value,
                mask_init_range,
                mask_fit_loss_weight,
                mask_coupled_edges,
                query_update_target,
                selection_query_lambda,
            ) = ("", "", None, None, "", "", None, None, None, None, None, None, None)
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
            _format_float(max_lambda),
            _format_float(min_lambda),
            str(alt_opt),
            str(dag_alm),
            _format_float(theta_lr),
            _format_float(mask_lr),
            str(theta_steps),
            str(mask_steps),
            _format_init_label(mask_init_mode, mask_init_value, mask_init_range),
            _format_float(mask_fit_loss_weight),
            _format_coupling_label(mask_coupled_edges),
            _format_optional(query_update_target),
            _format_float(selection_query_lambda),
            str(len(group_rows)),
            _fmt(_mean(true_kl)),
            _fmt(_std(true_kl)),
            _fmt(_mean(dat_kl)),
            _fmt(_std(dat_kl)),
            _fmt(_mean(bound_error)),
            _fmt(_std(bound_error)),
            _fmt(max(max_bound_error)),
        ]))


def _summary_metric_rows(
        root_dir,
        include_nonstandard=False,
        graph_names=None,
        dimension_labels=None,
        train_seed_offsets=None,
        theta_lr=None,
        mask_lr=None,
        theta_steps_per_mask=None,
        mask_steps_per_theta=None,
        schedule_filters=None):
    rows = [
        row for row in _filter_rows(
            load_bound_results(root_dir),
            include_nonstandard=include_nonstandard,
            graph_names=graph_names,
            dimension_labels=dimension_labels,
            train_seed_offsets=train_seed_offsets,
            theta_lr=theta_lr,
            mask_lr=mask_lr,
            theta_steps_per_mask=theta_steps_per_mask,
            mask_steps_per_theta=mask_steps_per_theta,
            schedule_filters=schedule_filters,
        )
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
        group = (
            row["mask_mode"],
            row["cycle_lambda"],
            row["max_lambda"],
            row["min_lambda"],
        )
        if include_nonstandard:
            group = group + (
                row["alt_opt"],
                row["dag_alm"],
                row["theta_lr"],
                row["mask_lr"],
                row["theta_steps_per_mask"],
                row["mask_steps_per_theta"],
                row["mask_init_mode"],
                row["mask_init_value"],
                row["mask_init_range"],
                row["mask_fit_loss_weight"],
                row["mask_coupled_edges"],
                row["query_update_target"],
                row["selection_query_lambda"],
            )
        metric_rows.append({
            "group": group,
            "min_kl": row["min_total_true_KL"],
            "max_kl": row["max_total_true_KL"],
            "min_bound_error": abs(row["err_lower"]),
            "max_bound_error": abs(row["err_upper"]),
        })
    return metric_rows


def plot_summary_stats(
        root_dir,
        output_path=None,
        title=None,
        figsize=None,
        include_nonstandard=False,
        graph_names=None,
        dimension_labels=None,
        train_seed_offsets=None,
        theta_lr=None,
        mask_lr=None,
        theta_steps_per_mask=None,
        mask_steps_per_theta=None,
        schedule_filters=None):
    metric_rows = _summary_metric_rows(
        root_dir,
        include_nonstandard=include_nonstandard,
        graph_names=graph_names,
        dimension_labels=dimension_labels,
        train_seed_offsets=train_seed_offsets,
        theta_lr=theta_lr,
        mask_lr=mask_lr,
        theta_steps_per_mask=theta_steps_per_mask,
        mask_steps_per_theta=mask_steps_per_theta,
        schedule_filters=schedule_filters,
    )
    if not metric_rows:
        row_kind = "rows" if include_nonstandard else "standard rows"
        raise ValueError(f"No completed {row_kind} found for summary stats under {root_dir}")

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
    labels = []
    for group in groups:
        mask, cycle_lambda, max_lambda, min_lambda = group[:4]
        lines = [
            str(mask),
            _cycle_query_lambda_label(cycle_lambda, min_lambda, max_lambda),
        ]
        if len(group) > 4:
            (
                alt_opt,
                dag_alm,
                theta_lr,
                mask_lr,
                theta_steps,
                mask_steps,
                mask_init_mode,
                mask_init_value,
                mask_init_range,
                mask_fit_loss_weight,
                mask_coupled_edges,
                query_update_target,
                selection_query_lambda,
            ) = group[4:]
            parts = []
            if alt_opt:
                parts.append("alt")
            if dag_alm:
                parts.append("alm")
            parts.extend([
                f"tlr={_format_float(theta_lr)}",
                f"mlr={_format_float(mask_lr)}",
                f"s={theta_steps}:{mask_steps}",
            ])
            lines.append(" ".join(parts))
            lines.append(
                " ".join([
                    f"init={_format_init_label(mask_init_mode, mask_init_value, mask_init_range)}",
                    f"mfit={_format_float(mask_fit_loss_weight)}",
                ])
            )
            lines.append(
                " ".join([
                    f"coup={_format_coupling_label(mask_coupled_edges)}",
                    f"q={_format_optional(query_update_target)}",
                    f"sel={_format_float(selection_query_lambda)}",
                ])
            )
        labels.append("\n".join(lines))
    metrics = [
        ("min_kl", "Min KL"),
        ("max_kl", "Max KL"),
        ("min_bound_error", "Min Bound Error"),
        ("max_bound_error", "Max Bound Error"),
    ]
    colors = ["#4c78a8", "#72b7b2", "#d62728", "#ff7f0e"]

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
        if metric_key in {"min_kl", "max_kl"} and all(value > 0 for value in means):
            ax.set_yscale("log")

    axes[-1].set_xticks(xs)
    axes[-1].set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    filter_label = _filter_label(
        graph_names=graph_names,
        dimension_labels=dimension_labels,
        train_seed_offsets=train_seed_offsets,
        theta_lr=theta_lr,
        mask_lr=mask_lr,
        theta_steps_per_mask=theta_steps_per_mask,
        mask_steps_per_theta=mask_steps_per_theta,
        schedule_filters=schedule_filters,
    )
    axes[0].set_title(title or f"Aggregate KL and Bound Errors ({Path(root_dir)}){filter_label}")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, axes


def main():
    parser = argparse.ArgumentParser(description="Plot true and NCM min/max values for bound runs.")
    parser.add_argument("root_dir", help="Root directory to search recursively for results.json files")
    parser.add_argument("--output", help="Defaults to <root_dir>/bound_results.png")
    parser.add_argument("--title", help="Optional plot title")
    parser.add_argument("--kl-output", help="Defaults to <root_dir>/bound_kl_results.png")
    parser.add_argument("--kl-title", help="Optional KL plot title")
    parser.add_argument("--summary-output", help="Defaults to <root_dir>/bound_summary_stats.png")
    parser.add_argument("--summary-title", help="Optional aggregate summary plot title")
    parser.add_argument("--no-table", action="store_true", help="Do not print grouped KL/bound-error summary table")
    parser.add_argument(
        "--graph",
        action="append",
        dest="graph_names",
        help="Filter by graph name. May be repeated.",
    )
    parser.add_argument(
        "--dim",
        action="append",
        dest="dimension_labels",
        help=(
            "Filter by dimensionality label from the run directory, such as 1. "
            "May be repeated."
        ),
    )
    parser.add_argument(
        "--per-graph",
        action="store_true",
        help="Write separate bound, KL, and summary plots for each graph under the root.",
    )
    parser.add_argument(
        "--subplot-graphs",
        action="store_true",
        help="Put each graph in its own subplot in the bound-results plot.",
    )
    parser.add_argument(
        "--pool-train-seeds",
        choices=[POOL_NONE, POOL_ENDPOINTS, POOL_ALL_OUTPUTS],
        default=POOL_NONE,
        help=(
            "Pool NCM min/max markers across train seeds within each plotted group. "
            "'endpoints' uses min over min outputs and max over max outputs; "
            "'all-outputs' uses min/max over both min and max outputs."
        ),
    )
    parser.add_argument(
        "--include-nonstandard",
        action="store_true",
        help="Include runs normally skipped as nonstandard, such as alt-opt or dag-alm runs",
    )
    parser.add_argument(
        "--train-seed-offset",
        action="append",
        dest="train_seed_offsets",
        help="Filter by train seed offset. May be repeated.",
    )
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
    args = parser.parse_args()

    output = args.output or str(Path(args.root_dir) / "bound_results.png")
    kl_output = args.kl_output or str(Path(args.root_dir) / "bound_kl_results.png")
    summary_output = args.summary_output or str(Path(args.root_dir) / "bound_summary_stats.png")

    graph_groups = [args.graph_names]
    if args.per_graph:
        rows = load_bound_results(args.root_dir)
        available_graphs = sorted({
            row["graph"] for row in rows
            if _matches_any(row["graph"], args.graph_names)
            and _matches_any(row["dimension_label"], args.dimension_labels)
        })
        if not available_graphs:
            raise ValueError(f"No completed bound results found under {args.root_dir}")
        graph_groups = [[graph] for graph in available_graphs]

    saved_paths = []
    for graph_names in graph_groups:
        graph_suffix = graph_names[0] if args.per_graph and graph_names else None
        cur_output = _path_with_graph(output, graph_suffix) if graph_suffix else output
        cur_kl_output = _path_with_graph(kl_output, graph_suffix) if graph_suffix else kl_output
        cur_summary_output = _path_with_graph(summary_output, graph_suffix) if graph_suffix else summary_output

        plot_bound_results(
            args.root_dir,
            output_path=cur_output,
            title=args.title,
            include_nonstandard=args.include_nonstandard,
            graph_names=graph_names,
            dimension_labels=args.dimension_labels,
            train_seed_offsets=args.train_seed_offsets,
            theta_lr=args.theta_lr,
            mask_lr=args.mask_lr,
            theta_steps_per_mask=args.theta_steps_per_mask,
            mask_steps_per_theta=args.mask_steps_per_theta,
            schedule_filters=args.schedule_filters,
            subplot_graphs=args.subplot_graphs and not args.per_graph,
            pool_train_seeds=args.pool_train_seeds,
        )
        plot_kl_results(
            args.root_dir,
            output_path=cur_kl_output,
            title=args.kl_title,
            include_nonstandard=args.include_nonstandard,
            graph_names=graph_names,
            dimension_labels=args.dimension_labels,
            train_seed_offsets=args.train_seed_offsets,
            theta_lr=args.theta_lr,
            mask_lr=args.mask_lr,
            theta_steps_per_mask=args.theta_steps_per_mask,
            mask_steps_per_theta=args.mask_steps_per_theta,
            schedule_filters=args.schedule_filters,
        )
        plot_summary_stats(
            args.root_dir,
            output_path=cur_summary_output,
            title=args.summary_title,
            include_nonstandard=args.include_nonstandard,
            graph_names=graph_names,
            dimension_labels=args.dimension_labels,
            train_seed_offsets=args.train_seed_offsets,
            theta_lr=args.theta_lr,
            mask_lr=args.mask_lr,
            theta_steps_per_mask=args.theta_steps_per_mask,
            mask_steps_per_theta=args.mask_steps_per_theta,
            schedule_filters=args.schedule_filters,
        )
        if not args.no_table:
            _print_summary_table(
                args.root_dir,
                include_nonstandard=args.include_nonstandard,
                graph_names=graph_names,
                dimension_labels=args.dimension_labels,
                train_seed_offsets=args.train_seed_offsets,
                theta_lr=args.theta_lr,
                mask_lr=args.mask_lr,
                theta_steps_per_mask=args.theta_steps_per_mask,
                mask_steps_per_theta=args.mask_steps_per_theta,
                schedule_filters=args.schedule_filters,
            )
        saved_paths.extend([cur_output, cur_kl_output, cur_summary_output])

    for path in saved_paths:
        print(f"Saved plot to {path}")


if __name__ == "__main__":
    main()
