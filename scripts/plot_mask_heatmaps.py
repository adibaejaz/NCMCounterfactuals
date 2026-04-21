#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_bound_results import _as_float, _standard_bound_run


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


def _row_label(row):
    offset = row["train_seed_offset"]
    if offset is None:
        offset = "0"
    return f"seed {offset}\n{row['run_id'][:6]}"


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
            "cycle_lambda": _as_float(hyperparams.get("cycle-lambda")),
            "mask_mode": hyperparams.get("mask-mode", "?"),
            "train_seed_offset": hyperparams.get("train-seed-offset", "0"),
            "cycle_penalty": hyperparams.get("cycle-penalty"),
            "variables": min_vars,
            "mask_min": mask_min,
            "mask_max": mask_max,
            "mask_diff": mask_max - mask_min,
            "mask_dir": str(mask_dir),
        })

    rows.sort(key=lambda row: (
        row["cycle_lambda"] if row["cycle_lambda"] is not None else float("inf"),
        str(row["mask_mode"]),
        int(row["trial_index"]) if str(row["trial_index"]).isdigit() else str(row["trial_index"]),
        _as_float(row.get("train_seed_offset"), default=float("inf")),
        row["run_id"],
    ))
    return rows


def _filter_rows(rows, mask_mode=None, cycle_lambda=None, trial=None, run_id=None):
    filtered = []
    for row in rows:
        if mask_mode is not None and row["mask_mode"] != mask_mode:
            continue
        if cycle_lambda is not None and row["cycle_lambda"] != cycle_lambda:
            continue
        if trial is not None and str(row["trial_index"]) != str(trial):
            continue
        if run_id is not None and row["run_id"] != run_id:
            continue
        filtered.append(row)
    return filtered


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
        mask_mode=None,
        cycle_lambda=None,
        trial=None,
        run_id=None,
        include_nonstandard=False,
        max_rows=None,
        title=None):
    rows = load_mask_rows(root_dir, include_nonstandard=include_nonstandard)
    rows = _filter_rows(
        rows,
        mask_mode=mask_mode,
        cycle_lambda=cycle_lambda,
        trial=trial,
        run_id=run_id,
    )
    if max_rows is not None:
        rows = rows[:max_rows]
    if not rows:
        raise ValueError("No mask rows matched the requested filters")

    variables = rows[0]["variables"]
    for row in rows:
        if row["variables"] != variables:
            raise ValueError("Cannot plot masks with different variable orders in one figure")

    n_rows = len(rows)
    fig_width = 11
    fig_height = max(3.5, 2.6 * n_rows)
    fig, axes = plt.subplots(n_rows, 3, figsize=(fig_width, fig_height), squeeze=False)

    mask_images = []
    diff_images = []
    for i, row in enumerate(rows):
        row_title = _row_label(row)
        axes[i, 0].text(
            -0.55,
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
        if cycle_lambda is not None:
            filters.append(f"lambda={cycle_lambda:g}")
        if trial is not None:
            filters.append(f"trial={trial}")
        if run_id is not None:
            filters.append(f"run={run_id}")
        title = "Mask Heatmaps"
        if filters:
            title += " (" + ", ".join(filters) + ")"
    fig.suptitle(title, y=0.995, fontsize=14)
    fig.subplots_adjust(left=0.16, right=0.88, top=0.93, bottom=0.06, wspace=0.35, hspace=0.45)

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, axes


def main():
    parser = argparse.ArgumentParser(description="Plot learned min/max mask heatmaps for bound runs.")
    parser.add_argument("root_dir", help="Root directory to search recursively for mask_min.json files")
    parser.add_argument("--output", help="Defaults to <root_dir>/mask_heatmaps.png")
    parser.add_argument("--mask-mode", help="Filter by mask mode, e.g. gate, multiply, st-gate")
    parser.add_argument("--cycle-lambda", type=float, help="Filter by cycle lambda")
    parser.add_argument("--trial", help="Filter by trial index")
    parser.add_argument("--run-id", help="Filter by full run hash")
    parser.add_argument("--include-nonstandard", action="store_true",
                        help="Include runs using alt-opt or dag-alm")
    parser.add_argument("--max-rows", type=int, help="Maximum number of matching runs to plot")
    parser.add_argument("--title", help="Optional figure title")
    args = parser.parse_args()

    output = args.output or str(Path(args.root_dir) / "mask_heatmaps.png")
    plot_mask_heatmaps(
        args.root_dir,
        output_path=output,
        mask_mode=args.mask_mode,
        cycle_lambda=args.cycle_lambda,
        trial=args.trial,
        run_id=args.run_id,
        include_nonstandard=args.include_nonstandard,
        max_rows=args.max_rows,
        title=args.title,
    )
    print(f"Saved mask heatmaps to {output}")


if __name__ == "__main__":
    main()
