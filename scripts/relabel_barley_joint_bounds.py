"""Reuse accepted observational barley datasets for a joint-intervention query.

This script reconstructs the accepted CTM structural models from an existing
generated-data root, computes new barley joint-intervention bound diagnostics,
and writes a new generated-data root that reuses the original observational
datasets and stored metrics.
"""

import argparse
import ast
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

from src.ds.causal_graph import CausalGraph
from src.generate_bound_datasets import (
    _adjustment_separation_diagnostics,
    _bound_diagnostics,
    _bound_do_label,
    _dimension_label,
    _write_json,
)
from src.run.data_setup import _build_dat_model, _build_v_sizes
from src.scm.ctm import CTM


def _parse_literal(value):
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return ast.literal_eval(value)


def _coerce_hyperparams(raw):
    hyperparams = {}
    for key, value in raw.items():
        if key in {"regions", "gen-bs", "min-empirical-cell-count", "bound-outcome-value",
                   "bound-treatment-value"}:
            hyperparams[key] = int(value)
        elif key in {"c2-scale", "bound-gap-min", "adjustment-gap-min", "positivity-epsilon"}:
            hyperparams[key] = float(value)
        elif key in {"positivity", "adjustment-gap-enabled"}:
            hyperparams[key] = str(value).lower() == "true"
        elif key in {"do-var-list", "bound-do", "v-sizes"}:
            hyperparams[key] = _parse_literal(value)
        else:
            hyperparams[key] = value
    return hyperparams


def _load_json(path):
    with open(path) as file:
        return json.load(file)


def _source_trial_dirs(source_root, graph):
    pattern = "graph={}-n_samples=*-dim=*-trial_index=*".format(graph)
    return sorted(path for path in Path(source_root).glob(pattern) if path.is_dir())


def _trial_index(metadata):
    return int(metadata["trial_index"])


def _target_trial_dir(output_root, graph, n, dim, trial_index, bound_do, v_sizes=None):
    label = _bound_do_label(bound_do)
    return Path(output_root) / "graph={}-n_samples={}-dim={}-bound_do={}-trial_index={}".format(
        graph, n, _dimension_label(dim, v_sizes), label, trial_index)


def _link_or_copy_file(source_path, target_path, overwrite=False, copy=False):
    source_path = Path(source_path)
    target_path = Path(target_path)
    if target_path.exists():
        if not overwrite:
            raise FileExistsError("{} already exists; pass --overwrite to replace".format(target_path))
        target_path.unlink()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if copy:
        shutil.copy2(source_path, target_path)
        return "copied"

    try:
        os.link(source_path, target_path)
        return "hardlinked"
    except OSError:
        shutil.copy2(source_path, target_path)
        return "copied"


def _reuse_large_artifacts(source_dir, target_dir, graph, overwrite=False, copy=False):
    artifacts = [
        "dat.th",
        "{}_dat.th".format(graph),
        "stored_metrics.th",
        "{}_stored_metrics.th".format(graph),
    ]
    actions = {}
    for name in artifacts:
        source_path = Path(source_dir) / name
        if not source_path.is_file():
            raise FileNotFoundError("required source artifact missing: {}".format(source_path))
        actions[name] = _link_or_copy_file(
            source_path, Path(target_dir) / name, overwrite=overwrite, copy=copy)
    return actions


def _copy_positivity_diagnostics(source_dir, target_dir, graph):
    for name in [
            "positivity_diagnostics.json",
            "{}_positivity_diagnostics.json".format(graph)]:
        source_path = Path(source_dir) / name
        if source_path.is_file():
            shutil.copy2(source_path, Path(target_dir) / name)


def _diagnostic_args(args):
    return SimpleNamespace(
        graph="barley",
        bound_outcome="protein",
        bound_outcome_value=int(args.bound_outcome_value),
        bound_treatment="sort",
        bound_treatment_value=0,
        bound_do=dict(args.bound_do),
        bound_do_json=json.dumps(args.bound_do, sort_keys=True),
        bound_mc_samples=int(args.bound_mc_samples),
        bound_gap_min=float(args.bound_gap_min),
        adjustment_gap_min=float(args.adjustment_gap_min),
    )


def _updated_hyperparams(raw_hyperparams, args):
    updated = dict(raw_hyperparams)
    updated["bound-outcome"] = "protein"
    updated["bound-outcome-value"] = str(args.bound_outcome_value)
    updated["bound-do"] = str(dict(args.bound_do))
    updated["bound-gap-min"] = str(args.bound_gap_min)
    updated["adjustment-gap-min"] = str(args.adjustment_gap_min)
    updated["adjustment-gap-enabled"] = str(not args.no_adjustment_gap)
    return updated


def _updated_metadata(metadata, source_dir, target_dir, bound, adjustment, artifact_actions, args):
    updated = dict(metadata)
    updated["bound_do"] = dict(args.bound_do)
    updated["bound_do_label"] = _bound_do_label(args.bound_do)
    updated["bound_outcome"] = "protein"
    updated["bound_outcome_value"] = int(args.bound_outcome_value)
    updated["bound_gap_min"] = float(args.bound_gap_min)
    updated["adjustment_gap_min"] = float(args.adjustment_gap_min)
    updated["adjustment_gap_enabled"] = not args.no_adjustment_gap
    updated["accepted"] = bool(
        bound["passed"] and (args.no_adjustment_gap or adjustment["passed"]))
    updated["reject_reasons"] = []
    if not bound["passed"]:
        updated["reject_reasons"].append("bound_gap")
    if not args.no_adjustment_gap and not adjustment["passed"]:
        updated["reject_reasons"].append("adjustment_separation")
    updated["reused_from"] = str(Path(source_dir))
    updated["reused_artifacts"] = artifact_actions
    updated["relabel_output_dir"] = str(Path(target_dir))
    updated["relabel_bound_mc_samples"] = int(args.bound_mc_samples)
    return updated


def _process_trial(source_dir, args):
    metadata = _load_json(Path(source_dir) / "data_metadata.json")
    if metadata.get("graph") != "barley":
        raise ValueError("expected barley metadata in {}, got {}".format(
            source_dir, metadata.get("graph")))
    if metadata.get("do_var_list") != [{}]:
        raise ValueError("expected observational do_var_list [{{}}] in {}".format(source_dir))

    raw_hyperparams = _load_json(Path(source_dir) / "hyperparams.json")
    hyperparams = _coerce_hyperparams(raw_hyperparams)
    if "v_sizes" in metadata:
        hyperparams["v-sizes"] = metadata["v_sizes"]

    graph_file = metadata.get("graph_file", "dat/cg/barley.cg")
    cg = CausalGraph.read(graph_file)
    dat_m = _build_dat_model(
        CTM,
        cg,
        int(metadata["dim"]),
        hyperparams,
        int(metadata["candidate_seed"]))

    diag_args = _diagnostic_args(args)
    bound = _bound_diagnostics(dat_m, diag_args)
    if args.no_adjustment_gap:
        adjustment = {
            "enabled": False,
            "passed": True,
            "reason": "disabled_by_no_adjustment_gap",
        }
    else:
        adjustment = _adjustment_separation_diagnostics(bound, diag_args)

    v_sizes = _build_v_sizes(cg, int(metadata["dim"]), hyperparams)
    explicit_v_sizes = v_sizes if "v-sizes" in hyperparams else None
    target_dir = _target_trial_dir(
        args.output_root,
        "barley",
        int(metadata["n_samples"]),
        int(metadata["dim"]),
        _trial_index(metadata),
        args.bound_do,
        explicit_v_sizes)
    target_dir.mkdir(parents=True, exist_ok=True)

    artifact_actions = _reuse_large_artifacts(
        source_dir, target_dir, "barley", overwrite=args.overwrite, copy=args.copy)
    _copy_positivity_diagnostics(source_dir, target_dir, "barley")

    updated_metadata = _updated_metadata(
        metadata, source_dir, target_dir, bound, adjustment, artifact_actions, args)
    if updated_metadata["reject_reasons"] and not args.allow_failed:
        raise RuntimeError(
            "joint query diagnostics failed for {}: {}".format(
                source_dir, updated_metadata["reject_reasons"]))

    updated_hyperparams = _updated_hyperparams(raw_hyperparams, args)
    _write_json(target_dir / "hyperparams.json", updated_hyperparams)
    _write_json(target_dir / "data_metadata.json", updated_metadata)
    _write_json(target_dir / "barley_data_metadata.json", updated_metadata)
    _write_json(target_dir / "ground_truth_bounds.json", bound)
    _write_json(target_dir / "barley_ground_truth_bounds.json", bound)
    _write_json(target_dir / "adjustment_separation_diagnostics.json", adjustment)
    _write_json(target_dir / "barley_adjustment_separation_diagnostics.json", adjustment)
    return target_dir, bound, adjustment


def build_parser():
    parser = argparse.ArgumentParser(
        description="Reuse accepted observational barley CTM datasets for joint barley bounds.")
    parser.add_argument(
        "--source-root",
        default="out/paper_bound_datasets_adjustment_gap",
        help="root containing accepted single-intervention barley dataset directories")
    parser.add_argument(
        "--output-root",
        default="out/paper_bound_datasets_barley_joint_obs",
        help="root for reused joint-bound dataset directories")
    parser.add_argument("--trial-index", action="append", type=int, default=[],
                        help="specific trial index to relabel; may be repeated")
    parser.add_argument("--bound-mc-samples", type=int, default=1000000)
    parser.add_argument("--bound-gap-min", type=float, default=0.1)
    parser.add_argument("--adjustment-gap-min", type=float, default=0.1)
    parser.add_argument("--no-adjustment-gap", action="store_true")
    parser.add_argument("--bound-outcome-value", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--copy", action="store_true",
                        help="copy large artifacts instead of hardlinking them")
    parser.add_argument("--allow-failed", action="store_true",
                        help="write outputs even if bound or adjustment diagnostics fail")
    return parser


def main():
    args = build_parser().parse_args()
    args.bound_do = {"sort": 0, "saatid": 0}
    if args.bound_mc_samples <= 0:
        raise ValueError("--bound-mc-samples must be positive")

    source_dirs = _source_trial_dirs(args.source_root, "barley")
    if args.trial_index:
        selected = set(args.trial_index)
        source_dirs = [
            path for path in source_dirs
            if int(_load_json(path / "data_metadata.json")["trial_index"]) in selected
        ]
    if not source_dirs:
        raise FileNotFoundError("no barley trial directories found in {}".format(args.source_root))

    for source_dir in source_dirs:
        target_dir, bound, adjustment = _process_trial(source_dir, args)
        print("[reused] {} -> {} gap={:.6f} bound_pass={} adjustment_pass={}".format(
            source_dir,
            target_dir,
            bound["gap"],
            bound["passed"],
            adjustment["passed"]))


if __name__ == "__main__":
    main()
