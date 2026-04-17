#!/usr/bin/env python3
"""Export tuning run summaries to CSV.

This scans an experiment output tree such as ``out/chain_tuning`` and writes
one CSV row per completed run. The export includes:

- training/config hyperparameters
- final fit metrics from ``results.json``
- learned graph as per-edge mask-weight columns when ``mask.json`` exists
- compact mask summary strings for quick inspection

Example:
    python scripts/export_tuning_csv.py \
        --root out/chain_tuning \
        --out out/chain_tuning_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def _load_json(path: Path) -> Dict:
    with path.open() as f:
        return json.load(f)


def _coerce_scalar(value):
    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return value
    if not isinstance(value, str):
        return value

    lowered = value.lower()
    if lowered == "none":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _mask_edges(variables: Sequence[str], mask: Sequence[Sequence[float]]) -> List[Tuple[str, float]]:
    edges: List[Tuple[str, float]] = []
    for i, src in enumerate(variables):
        for j, dst in enumerate(variables):
            if i == j:
                continue
            edges.append((f"{src}->{dst}", float(mask[i][j])))
    return edges


def _mask_summary(edges: Sequence[Tuple[str, float]], decimals: int) -> str:
    ordered = sorted(edges, key=lambda item: (-item[1], item[0]))
    fmt = "{:." + str(decimals) + "f}"
    return "; ".join(f"{edge}={fmt.format(weight)}" for edge, weight in ordered)


def _hard_graph(edges: Sequence[Tuple[str, float]], threshold: float, decimals: int) -> str:
    kept = [(edge, weight) for edge, weight in edges if weight > threshold]
    kept.sort(key=lambda item: (-item[1], item[0]))
    if not kept:
        return ""
    fmt = "{:." + str(decimals) + "f}"
    return "; ".join(f"{edge}={fmt.format(weight)}" for edge, weight in kept)


def _discover_edge_columns(run_dirs: Sequence[Path]) -> List[str]:
    edge_names = set()
    for run_dir in run_dirs:
        mask_path = run_dir / "mask.json"
        if not mask_path.exists():
            continue
        mask_payload = _load_json(mask_path)
        for edge, _ in _mask_edges(mask_payload["variables"], mask_payload["mask"]):
            edge_names.add(edge)
    return sorted(edge_names)


def _collect_run_dirs(root: Path) -> List[Path]:
    run_dirs = []
    for hp_path in sorted(root.rglob("hyperparams.json")):
        run_dir = hp_path.parent
        if (run_dir / "results.json").exists():
            run_dirs.append(run_dir)
    return run_dirs


def export_runs(root: Path, out_path: Path, threshold: float, decimals: int) -> int:
    run_dirs = _collect_run_dirs(root)
    edge_columns = _discover_edge_columns(run_dirs)

    base_columns = [
        "run_dir",
        "family",
        "penalty",
        "dagma_s",
        "mask_init_value",
        "cycle_lambda",
        "lr",
        "total_true_KL",
        "total_dat_KL",
        "err_ncm_ATE",
        "dag_h",
        "variables",
        "mask_summary",
        f"hard_graph_gt_{threshold:g}",
    ]
    edge_weight_columns = [f"edge_weight:{edge}" for edge in edge_columns]
    fieldnames = base_columns + edge_weight_columns

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for run_dir in run_dirs:
            hp = {k: _coerce_scalar(v) for k, v in _load_json(run_dir / "hyperparams.json").items()}
            results = _load_json(run_dir / "results.json")

            row = {
                "run_dir": str(run_dir),
                "family": "masked" if (run_dir / "mask.json").exists() else "baseline",
                "penalty": hp.get("cycle-penalty"),
                "dagma_s": hp.get("dagma-s"),
                "mask_init_value": hp.get("mask-init-value"),
                "cycle_lambda": hp.get("cycle-lambda"),
                "lr": hp.get("lr"),
                "total_true_KL": results.get("total_true_KL"),
                "total_dat_KL": results.get("total_dat_KL"),
                "err_ncm_ATE": results.get("err_ncm_ATE"),
                "dag_h": results.get("dag_h"),
                "variables": "",
                "mask_summary": "",
                f"hard_graph_gt_{threshold:g}": "",
            }

            for col in edge_weight_columns:
                row[col] = ""

            mask_path = run_dir / "mask.json"
            if mask_path.exists():
                mask_payload = _load_json(mask_path)
                variables = mask_payload["variables"]
                edges = _mask_edges(variables, mask_payload["mask"])
                row["variables"] = ",".join(variables)
                row["mask_summary"] = _mask_summary(edges, decimals)
                row[f"hard_graph_gt_{threshold:g}"] = _hard_graph(edges, threshold, decimals)
                edge_map = dict(edges)
                for edge in edge_columns:
                    row[f"edge_weight:{edge}"] = edge_map.get(edge, "")

            writer.writerow(row)

    return len(run_dirs)


def main():
    parser = argparse.ArgumentParser(description="Export tuning runs to CSV")
    parser.add_argument("--root", default="out/chain_tuning", help="root experiment directory")
    parser.add_argument("--out", default="out/chain_tuning_summary.csv", help="output CSV path")
    parser.add_argument("--hard-threshold", type=float, default=0.5,
                        help="threshold used for the hard-graph summary column")
    parser.add_argument("--decimals", type=int, default=4,
                        help="decimal places for mask summary strings")
    args = parser.parse_args()

    count = export_runs(Path(args.root), Path(args.out), args.hard_threshold, args.decimals)
    print(f"Wrote {count} runs to {args.out}")


if __name__ == "__main__":
    main()
