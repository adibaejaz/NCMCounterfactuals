#!/usr/bin/env python3
import argparse
import csv
import json
import re
import shutil
import statistics
from pathlib import Path


def _find_results_files(root_dir):
    return sorted(Path(root_dir).rglob("results.json"))


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _extract_run_dir(results_path):
    run_dir = results_path.parent
    if run_dir.name.isdigit():
        run_dir = run_dir.parent
    return run_dir


def _extract_piece(run_dir, prefix, default="?"):
    for piece in run_dir.name.split("-"):
        if piece.startswith(prefix):
            return piece[len(prefix):]
    return default


def _extract_graph(run_dir):
    return _extract_piece(run_dir, "graph=")


def _extract_trial_index(run_dir):
    return _extract_piece(run_dir, "trial_index=")


def _extract_run_id(run_dir):
    return _extract_piece(run_dir, "run=", run_dir.name)


def _query_name(results):
    names = sorted(key[len("min_ncm_"):] for key in results if key.startswith("min_ncm_"))
    if not names:
        raise ValueError("results file does not contain a min_ncm_* query")
    if len(names) > 1:
        raise ValueError("multiple min_ncm_* queries found; pass a single-query result root")
    return names[0]


def _safe_label(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _threshold_label(value):
    return str(value).replace(".", "p").replace("-", "m")


def _load_hyperparams(run_dir):
    path = run_dir / "hyperparams.json"
    if not path.is_file():
        return {}
    return _load_json(path)


def _candidate_from_results(results_path, side):
    results = _load_json(results_path)
    query_name = _query_name(results)
    run_dir = _extract_run_dir(results_path)
    result_dir = results_path.parent
    prefix = f"{side}_"
    candidate = {
        "side": side,
        "graph": _extract_graph(run_dir),
        "trial_index": _extract_trial_index(run_dir),
        "run_id": _extract_run_id(run_dir),
        "results_path": str(results_path),
        "run_dir": str(run_dir),
        "mask_path": str(result_dir / f"mask_{side}.json"),
        "query_name": query_name,
        "query_value": float(results[f"{side}_ncm_{query_name}"]),
        "dat_kl": float(results[f"{side}_total_dat_KL"]),
        "true_kl": float(results[f"{side}_total_true_KL"]),
        "results": results,
        "hyperparams": _load_hyperparams(run_dir),
    }
    for key, value in results.items():
        if key.startswith(prefix):
            candidate[key] = value
    return candidate


def _collect_candidates(root_dir):
    groups = {}
    for results_path in _find_results_files(root_dir):
        for side in ("min", "max"):
            candidate = _candidate_from_results(results_path, side)
            groups.setdefault((candidate["graph"], candidate["trial_index"]), []).append(candidate)
    return groups


def _copy_side_metrics(target, candidate, target_side):
    source_side = candidate["side"]
    source_prefix = f"{source_side}_"
    target_prefix = f"{target_side}_"
    for key, value in candidate["results"].items():
        if key.startswith(source_prefix):
            target[target_prefix + key[len(source_prefix):]] = value


def _copy_selected_mask(candidate, out_dir, target_side):
    mask_path = Path(candidate["mask_path"])
    if mask_path.is_file():
        shutil.copyfile(mask_path, out_dir / f"mask_{target_side}.json")


def _select_joint(candidates, kl_metric, kl_threshold):
    passing = [c for c in candidates if c[kl_metric] <= kl_threshold]
    used = passing if passing else candidates
    return min(used, key=lambda c: c["query_value"]), max(used, key=lambda c: c["query_value"]), passing


def _select_separate(candidates, kl_metric, kl_threshold):
    passing = [c for c in candidates if c[kl_metric] <= kl_threshold]
    selected = {}
    for side in ("min", "max"):
        side_pool = [c for c in passing if c["side"] == side]
        if not side_pool:
            side_pool = [c for c in candidates if c["side"] == side]
        selected[side] = side_pool
    return (
        min(selected["min"], key=lambda c: c["query_value"]),
        max(selected["max"], key=lambda c: c["query_value"]),
        passing,
    )


def _summarize(values):
    values = [float(v) for v in values]
    if not values:
        return {}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _summary(rows):
    by_graph = {}
    for row in rows:
        by_graph.setdefault(row["graph"], []).append(row)
    return {
        "overall": _summary_block(rows),
        "by_graph": {graph: _summary_block(graph_rows) for graph, graph_rows in sorted(by_graph.items())},
    }


def _summary_block(rows):
    return {
        "n_groups": len(rows),
        "avg_abs_bound_error": _summarize(row["avg_abs_bound_error"] for row in rows),
        "max_abs_bound_error": _summarize(row["max_abs_bound_error"] for row in rows),
        "selected_lower_dat_kl": _summarize(row["selected_lower_dat_kl"] for row in rows),
        "selected_upper_dat_kl": _summarize(row["selected_upper_dat_kl"] for row in rows),
        "selected_lower_true_kl": _summarize(row["selected_lower_true_kl"] for row in rows),
        "selected_upper_true_kl": _summarize(row["selected_upper_true_kl"] for row in rows),
        "n_relaxed_filter": sum(1 for row in rows if row["filter_relaxed"]),
    }


def pool_results(root_dir, output_root, kl_threshold=0.02, kl_metric="dat_kl", separate_label_pools=False):
    groups = _collect_candidates(root_dir)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for (graph, trial_index), candidates in sorted(groups.items()):
        if separate_label_pools:
            lower, upper, passing = _select_separate(candidates, kl_metric, kl_threshold)
            strategy = "separate_label_pools"
        else:
            lower, upper, passing = _select_joint(candidates, kl_metric, kl_threshold)
            strategy = "joint_query_extrema"

        query_name = lower["query_name"]
        source_results = lower["results"]
        true_lower_key = f"true_lower_{query_name}"
        true_upper_key = f"true_upper_{query_name}"
        if true_lower_key not in source_results or true_upper_key not in source_results:
            raise ValueError(f"missing true bound keys for {query_name} in {lower['results_path']}")

        true_lower = float(source_results[true_lower_key])
        true_upper = float(source_results[true_upper_key])
        lower_value = float(lower["query_value"])
        upper_value = float(upper["query_value"])

        pooled = {
            key: value
            for key, value in source_results.items()
            if key.startswith("true_")
        }
        _copy_side_metrics(pooled, lower, "min")
        _copy_side_metrics(pooled, upper, "max")
        pooled[f"min_ncm_{query_name}"] = lower_value
        pooled[f"max_ncm_{query_name}"] = upper_value
        pooled[f"min_err_ncm_{query_name}"] = pooled[f"true_{query_name}"] - lower_value
        pooled[f"max_err_ncm_{query_name}"] = pooled[f"true_{query_name}"] - upper_value
        pooled[f"minmax_{query_name}_gap"] = upper_value - lower_value
        pooled[f"err_min_ncm_{query_name}_lower_bound"] = true_lower - lower_value
        pooled[f"err_max_ncm_{query_name}_upper_bound"] = true_upper - upper_value
        pooled["pooling_strategy"] = strategy
        pooled["pooling_kl_metric"] = kl_metric
        pooled["pooling_kl_threshold"] = kl_threshold
        pooled["pooling_candidates_total"] = len(candidates)
        pooled["pooling_candidates_passing"] = len(passing)
        pooled["pooling_filter_relaxed"] = len(passing) == 0
        pooled["pooling_selected_lower"] = {
            key: lower[key]
            for key in ("side", "run_id", "results_path", "mask_path", "query_value", "dat_kl", "true_kl")
        }
        pooled["pooling_selected_upper"] = {
            key: upper[key]
            for key in ("side", "run_id", "results_path", "mask_path", "query_value", "dat_kl", "true_kl")
        }

        run_dir = output_root / (
            f"gen=CTM-graph={graph}-n_samples=10000-dim=1-"
            f"trial_index={trial_index}-run=pooled_{_threshold_label(kl_threshold)}"
        )
        result_dir = run_dir / "0"
        result_dir.mkdir(parents=True, exist_ok=True)
        _write_json(result_dir / "results.json", pooled)
        _copy_selected_mask(lower, result_dir, "min")
        _copy_selected_mask(upper, result_dir, "max")

        hyperparams = dict(lower["hyperparams"])
        hyperparams.update({
            "pooling-strategy": strategy,
            "pooling-kl-metric": kl_metric,
            "pooling-kl-threshold": kl_threshold,
            "train-seed-offset": "pooled",
            "alt-opt": False,
            "dag-alm": False,
        })
        _write_json(run_dir / "hyperparams.json", {key: str(value) for key, value in hyperparams.items()})

        avg_abs_error = (abs(lower_value - true_lower) + abs(true_upper - upper_value)) / 2
        max_abs_error = max(abs(lower_value - true_lower), abs(true_upper - upper_value))
        summary_rows.append({
            "graph": graph,
            "trial_index": trial_index,
            "query_name": query_name,
            "true_lower": true_lower,
            "true_upper": true_upper,
            "selected_lower": lower_value,
            "selected_upper": upper_value,
            "selected_gap": upper_value - lower_value,
            "avg_abs_bound_error": avg_abs_error,
            "max_abs_bound_error": max_abs_error,
            "candidates_total": len(candidates),
            "candidates_passing": len(passing),
            "filter_relaxed": len(passing) == 0,
            "selected_lower_side": lower["side"],
            "selected_upper_side": upper["side"],
            "selected_lower_run_id": lower["run_id"],
            "selected_upper_run_id": upper["run_id"],
            "selected_lower_dat_kl": lower["dat_kl"],
            "selected_upper_dat_kl": upper["dat_kl"],
            "selected_lower_true_kl": lower["true_kl"],
            "selected_upper_true_kl": upper["true_kl"],
            "selected_lower_results_path": lower["results_path"],
            "selected_upper_results_path": upper["results_path"],
        })

    csv_path = output_root / "pooled_selections.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    summary = _summary(summary_rows)
    summary.update({
        "source_root": str(root_dir),
        "output_root": str(output_root),
        "pooling_strategy": "separate_label_pools" if separate_label_pools else "joint_query_extrema",
        "kl_metric": kl_metric,
        "kl_threshold": kl_threshold,
        "selections_csv": str(csv_path),
    })
    _write_json(output_root / "summary_stats.json", summary)
    return output_root, summary


def main():
    parser = argparse.ArgumentParser(description="Pool min/max bound runs by final query value after KL filtering.")
    parser.add_argument("root_dir", help="Root directory containing original paired min/max results")
    parser.add_argument("--output-root", help="Derived output root. Defaults to <root>_pooled_query_kl<THRESHOLD>")
    parser.add_argument("--kl-threshold", type=float, default=0.02)
    parser.add_argument("--kl-metric", choices=("dat_kl", "true_kl"), default="dat_kl")
    parser.add_argument(
        "--separate-label-pools",
        action="store_true",
        help="Select lower only from original min models and upper only from original max models. Default pools both labels jointly.",
    )
    args = parser.parse_args()

    output_root = args.output_root
    if output_root is None:
        strategy_label = "separate_label_pools" if args.separate_label_pools else "joint"
        output_root = "{}_pooled_query_{}_kl{}".format(
            args.root_dir.rstrip("/"), strategy_label, _threshold_label(args.kl_threshold))
    output_root, summary = pool_results(
        args.root_dir,
        output_root,
        kl_threshold=args.kl_threshold,
        kl_metric=args.kl_metric,
        separate_label_pools=args.separate_label_pools,
    )
    overall = summary["overall"]
    print(f"Wrote pooled results to {output_root}")
    print(f"Groups: {overall['n_groups']}")
    print("Avg abs bound error:", overall["avg_abs_bound_error"])
    print("Max abs bound error:", overall["max_abs_bound_error"])
    print(f"Selection CSV: {summary['selections_csv']}")


if __name__ == "__main__":
    main()
