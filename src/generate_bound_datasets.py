"""Generate certified CTM datasets for bound experiments.

This script is intentionally method-free: it samples CTM data-generating
models, generates datasets, certifies bound width and joint positivity, and
saves the accepted datasets for downstream experiments.
"""

import argparse
import hashlib
import json
import os
import random

import numpy as np
import torch as T

from src.ds.causal_graph import CausalGraph
from src.metric.evaluation import scm_query_bound_metrics, serialize_probability
from src.metric.queries import get_experimental_variables
from src.run.data_setup import (
    _build_dat_model,
    _build_stored_metrics,
    _build_v_sizes,
    parse_var_dim_overrides,
)
from src.scm.ctm import CTM
from src.scm.scm import expand_do


def _stable_payload(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _stable_seed(payload):
    return int(hashlib.sha512(_stable_payload(payload).encode()).hexdigest(), 16) & 0xffffffff


def _parse_do_var_list(args):
    if args.do_var_list_json is not None:
        raw = json.loads(args.do_var_list_json)
        if not isinstance(raw, list):
            raise ValueError("--do-var-list-json must decode to a list of dictionaries")
        for do_set in raw:
            if not isinstance(do_set, dict):
                raise ValueError("--do-var-list-json must decode to a list of dictionaries")
        return raw

    if args.do_regime == "obs":
        return [{}]
    if args.do_regime == "obs-x":
        if len(args.bound_do) > 1:
            return [{}, dict(args.bound_do)]
        return [{}, {args.bound_treatment: 0}, {args.bound_treatment: 1}]
    if args.do_regime == "graph-default":
        return get_experimental_variables(args.graph)
    raise ValueError("unknown do regime: {}".format(args.do_regime))


def _parse_bound_do(args):
    if args.bound_do_json is None:
        return {args.bound_treatment: args.bound_treatment_value}

    raw = json.loads(args.bound_do_json)
    if not isinstance(raw, dict) or not raw:
        raise ValueError("--bound-do-json must decode to a non-empty dictionary")

    bound_do = {}
    for var, value in raw.items():
        if not isinstance(var, str):
            raise ValueError("--bound-do-json keys must be variable names")
        if not isinstance(value, int):
            raise ValueError("--bound-do-json values must be integers")
        bound_do[var] = value
    return bound_do


def _bit_columns(data, variables):
    cols = []
    for var in variables:
        value = data[var].detach().cpu().long()
        if value.ndim == 1:
            value = value.view(-1, 1)
        for j in range(value.shape[1]):
            cols.append(value[:, j])
    if not cols:
        n = len(data[next(iter(data))])
        return T.empty((n, 0), dtype=T.long)
    return T.stack(cols, dim=1)


def _encoded_counts(data, variables, num_bits):
    bits = _bit_columns(data, variables)
    if bits.shape[1] != num_bits:
        raise ValueError("expected {} bits, got {}".format(num_bits, bits.shape[1]))
    if num_bits == 0:
        return T.tensor([bits.shape[0]], dtype=T.long)
    powers = (2 ** T.arange(num_bits - 1, -1, -1, dtype=T.long))
    indices = (bits * powers).sum(dim=1)
    return T.bincount(indices, minlength=2 ** num_bits)


def _joint_positivity_for_regime(
        dat_m,
        dat_set,
        do_set,
        v_sizes,
        true_n,
        epsilon,
        min_count):
    variables = [v for v in v_sizes if v not in do_set]
    num_bits = sum(v_sizes[v] for v in variables)
    if num_bits > 24:
        raise ValueError(
            "joint positivity over {} bits is too large for exhaustive checking".format(num_bits))

    expanded_do_true = {k: expand_do(v, n=true_n) for (k, v) in do_set.items()}
    true_dat = dat_m(n=true_n, do=expanded_do_true)
    true_counts = _encoded_counts(true_dat, variables, num_bits)
    empirical_counts = _encoded_counts(dat_set, variables, num_bits)

    true_probs = true_counts.float() / float(true_n)
    min_true_prob = float(true_probs.min().item())
    min_empirical_count = int(empirical_counts.min().item())
    return {
        "do_set": dict(do_set),
        "variables": variables,
        "num_joint_cells": int(2 ** num_bits),
        "true_n": int(true_n),
        "min_true_joint_probability": min_true_prob,
        "min_empirical_joint_count": min_empirical_count,
        "passed": bool(min_true_prob >= epsilon and min_empirical_count >= min_count),
    }


def _joint_positivity_diagnostics(
        dat_m,
        dat_sets,
        do_var_list,
        v_sizes,
        true_n,
        epsilon,
        min_count):
    regimes = []
    for do_set, dat_set in zip(do_var_list, dat_sets):
        regimes.append(_joint_positivity_for_regime(
            dat_m=dat_m,
            dat_set=dat_set,
            do_set=do_set,
            v_sizes=v_sizes,
            true_n=true_n,
            epsilon=epsilon,
            min_count=min_count))
    return {
        "positivity_epsilon": float(epsilon),
        "min_empirical_cell_count": int(min_count),
        "passed": all(row["passed"] for row in regimes),
        "min_true_joint_probability": min(row["min_true_joint_probability"] for row in regimes),
        "min_empirical_joint_count": min(row["min_empirical_joint_count"] for row in regimes),
        "regimes": regimes,
    }


def _generate_candidate(cg, n, dim, hyperparams, do_var_list, seed):
    T.manual_seed(seed)
    T.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    dat_m = _build_dat_model(CTM, cg, dim, hyperparams, seed)
    dat_sets = []
    for do_set in do_var_list:
        expanded_do = {k: expand_do(v, n=n) for (k, v) in do_set.items()}
        dat_sets.append(dat_m(n=n, do=expanded_do))
    return dat_m, dat_sets


def _bound_diagnostics(dat_m, args):
    outcome_event = {args.bound_outcome: args.bound_outcome_value}
    if args.bound_do_json is None and len(args.bound_do) == 1:
        treatment_values = tuple(sorted({0, 1, args.bound_treatment_value}))
        metrics = scm_query_bound_metrics(
            dat_m,
            graph_name=args.graph,
            outcome_var=args.bound_outcome,
            outcome_value=args.bound_outcome_value,
            treatment_var=args.bound_treatment,
            treatment_values=treatment_values,
            n=args.bound_mc_samples,
        )

        bounds_by_treatment_value = {}
        for treatment_value in treatment_values:
            do_name = serialize_probability(
                outcome_event,
                do_vals={args.bound_treatment: treatment_value})
            lower_key = "true_lower_{}".format(do_name)
            upper_key = "true_upper_{}".format(do_name)
            lower = float(metrics[lower_key])
            upper = float(metrics[upper_key])
            bounds_by_treatment_value[str(treatment_value)] = {
                "query": do_name,
                "lower_key": lower_key,
                "upper_key": upper_key,
                "lower": lower,
                "upper": upper,
                "gap": upper - lower,
            }

        selected = bounds_by_treatment_value[str(args.bound_treatment_value)]
        return {
            "query": selected["query"],
            "bound_mc_samples": int(args.bound_mc_samples),
            "acceptance_treatment_value": int(args.bound_treatment_value),
            "bound_do": dict(args.bound_do),
            "lower_key": selected["lower_key"],
            "upper_key": selected["upper_key"],
            "lower": selected["lower"],
            "upper": selected["upper"],
            "gap": selected["gap"],
            "passed": bool(selected["gap"] >= args.bound_gap_min),
            "bounds_by_treatment_value": bounds_by_treatment_value,
            "metrics": {k: float(v) for (k, v) in metrics.items()},
        }

    metrics = scm_query_bound_metrics(
        dat_m,
        graph_name=args.graph,
        outcome_var=args.bound_outcome,
        outcome_value=args.bound_outcome_value,
        treatment_events=[args.bound_do],
        n=args.bound_mc_samples,
    )
    do_name = serialize_probability(outcome_event, do_vals=args.bound_do)
    lower_key = "true_lower_{}".format(do_name)
    upper_key = "true_upper_{}".format(do_name)
    lower = float(metrics[lower_key])
    upper = float(metrics[upper_key])
    selected = {
        "query": do_name,
        "lower_key": lower_key,
        "upper_key": upper_key,
        "lower": lower,
        "upper": upper,
        "gap": upper - lower,
    }
    return {
        "query": selected["query"],
        "bound_mc_samples": int(args.bound_mc_samples),
        "acceptance_treatment_event": dict(args.bound_do),
        "bound_do": dict(args.bound_do),
        "lower_key": selected["lower_key"],
        "upper_key": selected["upper_key"],
        "lower": selected["lower"],
        "upper": selected["upper"],
        "gap": selected["gap"],
        "passed": bool(selected["gap"] >= args.bound_gap_min),
        "bounds_by_treatment_event": {
            _stable_payload(args.bound_do): selected,
        },
        "metrics": {k: float(v) for (k, v) in metrics.items()},
    }


def _barley_joint_adjustment_query_sets():
    return (
        (),
        ("dg25",),
        ("frspdag",),
        ("srtprot",),
        ("srtprot", "dg25"),
        ("srtprot", "frspdag"),
        ("sorttkv",),
        ("sorttkv", "dg25"),
        ("sorttkv", "frspdag"),
        ("srtsize",),
        ("srtsize", "dg25"),
        ("srtsize", "frspdag"),
    )


def _sachs_joint_adjustment_query_sets():
    return (
        (),
        ("Jnk",),
        ("P38",),
        ("Erk", "Mek"),
        ("Erk", "Mek", "Raf"),
        ("Mek",),
        ("Mek", "Raf"),
        ("Raf",),
    )


def _adjustment_query_sets(graph_name, bound_do=None):
    if graph_name == "backdoor":
        return (("Z",),)
    if graph_name == "square":
        return (("Z",), ("W",))
    if graph_name == "four_clique":
        return (("Z", "W"), ("Z",), ("W",))
    if graph_name == "sachs":
        if bound_do is not None and set(bound_do) == {"PKA", "PKC"}:
            return _sachs_joint_adjustment_query_sets()
        return (
            ("Raf",),
            ("PKA",),
            ("Mek",),
            ("Raf", "PKA"),
            ("Raf", "Mek"),
            ("PKA", "Mek"),
            ("Raf", "PKA", "Mek"),
            ("Jnk",),
            ("PKA", "Jnk"),
            ("P38",),
            ("PKA", "P38"),
        )
    if graph_name == "barley":
        if bound_do is not None and set(bound_do) == {"sort", "saatid"}:
            return _barley_joint_adjustment_query_sets()
        return (("srtprot",), ("sorttkv",), ("srtsize",))
    return ()


def _adjustment_key_part(adjustment_vars):
    if not adjustment_vars:
        return "none"
    return "_".join(adjustment_vars)


def _adjusted_metric_key(graph_name, adjustment_vars, do_name):
    if graph_name == "backdoor":
        return "true_adjusted_{}".format(do_name)
    return "true_adjusted_{}_{}".format(_adjustment_key_part(adjustment_vars), do_name)


def _adjustment_separation_diagnostics(bound_diagnostics, args):
    adjustment_sets = _adjustment_query_sets(args.graph, bound_do=args.bound_do)
    if not adjustment_sets:
        return {
            "enabled": False,
            "passed": True,
            "reason": "graph_has_no_adjustment_queries",
        }

    outcome_event = {args.bound_outcome: args.bound_outcome_value}
    do_name = serialize_probability(
        outcome_event,
        do_vals=args.bound_do)
    conditional_key = "true_{}".format(serialize_probability(
        outcome_event,
        cond_vals=args.bound_do))

    metrics = bound_diagnostics["metrics"]
    conditional = float(metrics[conditional_key])
    comparisons = []
    for adjustment_vars in adjustment_sets:
        adjusted_key = _adjusted_metric_key(args.graph, adjustment_vars, do_name)
        adjusted = float(metrics[adjusted_key])
        comparisons.append({
            "adjustment_vars": list(adjustment_vars),
            "adjusted_key": adjusted_key,
            "adjusted": adjusted,
            "abs_diff_from_conditional": abs(adjusted - conditional),
        })

    max_diff = max(row["abs_diff_from_conditional"] for row in comparisons)
    return {
        "enabled": True,
        "epsilon": float(args.adjustment_gap_min),
        "query": do_name,
        "conditional_key": conditional_key,
        "conditional": conditional,
        "max_abs_diff_from_conditional": max_diff,
        "passed": bool(max_diff >= args.adjustment_gap_min),
        "comparisons": comparisons,
    }


def _dimension_label(dim, v_sizes=None):
    if not v_sizes:
        return str(dim)
    parts = ["{}{}".format(var, v_sizes[var]) for var in sorted(v_sizes)]
    return "{}-v{}".format(dim, "_".join(parts))


def _bound_do_label(bound_do):
    return "_".join("{}{}".format(var, bound_do[var]) for var in sorted(bound_do))


def _trial_directory(root, graph, n, dim, trial_index, v_sizes=None, bound_do_label=None):
    query_part = ""
    if bound_do_label:
        query_part = "-bound_do={}".format(bound_do_label)
    return os.path.join(
        root,
        "graph={}-n_samples={}-dim={}{}-trial_index={}".format(
            graph, n, _dimension_label(dim, v_sizes), query_part, trial_index))


def _write_json(path, obj):
    with open(path, "w") as file:
        json.dump(obj, file, indent=2, sort_keys=True)


def _append_jsonl(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as file:
        file.write(json.dumps(obj, sort_keys=True) + "\n")


def _read_rejected_retry_indices(path):
    retry_indices = set()
    if not os.path.isfile(path):
        return retry_indices

    with open(path) as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "invalid JSON in rejected candidate log {} at line {}".format(
                        path, line_number)) from exc
            retry_index = row.get("retry_index")
            if retry_index is not None:
                retry_indices.add(int(retry_index))
    return retry_indices


def _save_accepted(
        out_dir,
        graph,
        dat_sets,
        stored_metrics,
        hyperparams,
        metadata,
        bound_diagnostics,
        positivity_diagnostics,
        adjustment_diagnostics):
    os.makedirs(out_dir, exist_ok=True)
    T.save(dat_sets, os.path.join(out_dir, "dat.th"))
    T.save(dat_sets, os.path.join(out_dir, "{}_dat.th".format(graph)))
    T.save(stored_metrics, os.path.join(out_dir, "stored_metrics.th"))
    T.save(stored_metrics, os.path.join(out_dir, "{}_stored_metrics.th".format(graph)))
    _write_json(os.path.join(out_dir, "hyperparams.json"), {k: str(v) for (k, v) in hyperparams.items()})
    _write_json(os.path.join(out_dir, "data_metadata.json"), metadata)
    _write_json(os.path.join(out_dir, "{}_data_metadata.json".format(graph)), metadata)
    _write_json(os.path.join(out_dir, "ground_truth_bounds.json"), bound_diagnostics)
    _write_json(os.path.join(out_dir, "{}_ground_truth_bounds.json".format(graph)), bound_diagnostics)
    _write_json(os.path.join(out_dir, "positivity_diagnostics.json"), positivity_diagnostics)
    _write_json(os.path.join(out_dir, "{}_positivity_diagnostics.json".format(graph)), positivity_diagnostics)
    _write_json(os.path.join(out_dir, "adjustment_separation_diagnostics.json"), adjustment_diagnostics)
    _write_json(
        os.path.join(out_dir, "{}_adjustment_separation_diagnostics.json".format(graph)),
        adjustment_diagnostics)


def _candidate_seed(args, trial_index, retry_index, hyperparams, do_var_list):
    bound_payload = {
        "outcome": args.bound_outcome,
        "outcome_value": args.bound_outcome_value,
        "treatment": args.bound_treatment,
        "treatment_value": args.bound_treatment_value,
    }
    if args.bound_do_json is not None or len(args.bound_do) > 1:
        bound_payload["do"] = dict(args.bound_do)

    return _stable_seed({
        "purpose": "paper_bound_dataset_generation",
        "graph": args.graph,
        "trial_index": trial_index,
        "retry_index": retry_index,
        "n_samples": args.n_samples,
        "dim": args.dim,
        "v_sizes": hyperparams.get("v-sizes"),
        "hyperparams": hyperparams,
        "do_var_list": do_var_list,
        "bound": bound_payload,
    })


def _default_positivity_epsilon(cg):
    if len(cg.v) <= 3:
        return 0.01
    return 0.005


def _disabled_adjustment_diagnostics():
    return {
        "enabled": False,
        "passed": True,
        "reason": "disabled_by_no_adjustment_gap",
    }


def _disabled_positivity_diagnostics(args):
    return {
        "enabled": False,
        "passed": True,
        "reason": "disabled_by_no_positivity",
        "positivity_epsilon": float(args.positivity_epsilon),
        "positivity_mc_samples": int(args.positivity_mc_samples),
        "min_empirical_cell_count": int(args.min_empirical_cell_count),
        "min_true_joint_probability": None,
        "min_empirical_joint_count": None,
        "regimes": [],
    }


def _run_trial(args, trial_index, cg, do_var_list, hyperparams):
    v_sizes = _build_v_sizes(cg, args.dim, hyperparams)
    explicit_v_sizes = v_sizes if hyperparams.get("v-sizes") else None
    bound_do_label = _bound_do_label(args.bound_do) if len(args.bound_do) > 1 else None
    out_dir = _trial_directory(
        args.name, args.graph, args.n_samples, args.dim, trial_index, explicit_v_sizes,
        bound_do_label=bound_do_label)
    if os.path.isfile(os.path.join(out_dir, "dat.th")) and not args.overwrite:
        print("[done]", out_dir)
        return

    reject_log = os.path.join(out_dir, "rejected_candidates.jsonl")
    if args.overwrite and os.path.isfile(reject_log):
        os.remove(reject_log)
    rejected_retry_indices = set() if args.overwrite else _read_rejected_retry_indices(reject_log)
    if rejected_retry_indices:
        print("[resume] graph={} trial={} skipping {} rejected retries from {}".format(
            args.graph, trial_index, len(rejected_retry_indices), reject_log))

    for retry_index in range(args.max_retries + 1):
        if retry_index in rejected_retry_indices:
            continue

        seed = _candidate_seed(args, trial_index, retry_index, hyperparams, do_var_list)
        print("[candidate] graph={} trial={} retry={} seed={}".format(
            args.graph, trial_index, retry_index, seed))

        dat_m, dat_sets = _generate_candidate(
            cg=cg,
            n=args.n_samples,
            dim=args.dim,
            hyperparams=hyperparams,
            do_var_list=do_var_list,
            seed=seed)
        bound = _bound_diagnostics(dat_m, args)
        if args.no_adjustment_gap:
            adjustment = _disabled_adjustment_diagnostics()
        else:
            adjustment = _adjustment_separation_diagnostics(bound, args)

        if args.no_positivity:
            positivity = _disabled_positivity_diagnostics(args)
        else:
            positivity = _joint_positivity_diagnostics(
                dat_m=dat_m,
                dat_sets=dat_sets,
                do_var_list=do_var_list,
                v_sizes=v_sizes,
                true_n=args.positivity_mc_samples,
                epsilon=args.positivity_epsilon,
                min_count=args.min_empirical_cell_count)

        reject_reasons = []
        if not bound["passed"]:
            reject_reasons.append("bound_gap")
        if not adjustment["passed"]:
            reject_reasons.append("adjustment_separation")
        if not positivity["passed"]:
            reject_reasons.append("positivity")

        metadata = {
            "graph": args.graph,
            "graph_file": "dat/cg/{}.cg".format(args.graph),
            "trial_index": int(trial_index),
            "retry_index": int(retry_index),
            "candidate_seed": int(seed),
            "n_samples": int(args.n_samples),
            "dim": int(args.dim),
            "v_sizes": {k: int(v) for (k, v) in v_sizes.items()},
            "dimension_label": _dimension_label(args.dim, explicit_v_sizes),
            "generator": "CTM",
            "do_var_list": do_var_list,
            "bound_do": dict(args.bound_do),
            "bound_do_label": _bound_do_label(args.bound_do),
            "accepted": len(reject_reasons) == 0,
            "reject_reasons": reject_reasons,
            "bound_gap_min": float(args.bound_gap_min),
            "adjustment_gap_min": float(args.adjustment_gap_min),
            "adjustment_gap_enabled": not args.no_adjustment_gap,
            "positivity_epsilon": float(args.positivity_epsilon),
            "positivity_enabled": not args.no_positivity,
            "min_empirical_cell_count": int(args.min_empirical_cell_count),
        }

        if not reject_reasons:
            stored_metrics = _build_stored_metrics(dat_m, dat_sets, do_var_list)
            _save_accepted(
                out_dir=out_dir,
                graph=args.graph,
                dat_sets=dat_sets,
                stored_metrics=stored_metrics,
                hyperparams=hyperparams,
                metadata=metadata,
                bound_diagnostics=bound,
                positivity_diagnostics=positivity,
                adjustment_diagnostics=adjustment)
            print("[accepted]", out_dir)
            return

        _append_jsonl(reject_log, {
            **metadata,
            "bound_gap": bound["gap"],
            "bound_lower": bound["lower"],
            "bound_upper": bound["upper"],
            "adjustment_max_abs_diff_from_conditional": adjustment.get(
                "max_abs_diff_from_conditional"),
            "min_true_joint_probability": positivity["min_true_joint_probability"],
            "min_empirical_joint_count": positivity["min_empirical_joint_count"],
        })

    raise RuntimeError(
        "failed to generate accepted dataset for graph={} trial={} after {} retries".format(
            args.graph, trial_index, args.max_retries))


def build_parser():
    parser = argparse.ArgumentParser(description="Generate certified CTM datasets for bound experiments")
    parser.add_argument("name", help="output root directory")
    parser.add_argument("--graph", required=True, help="graph name, e.g. chain, backdoor, square, four_clique")
    parser.add_argument("--n-trials", type=int, default=1)
    parser.add_argument("--trial-index", action="append", type=int, default=[],
                        help="specific trial index to generate; may be repeated")
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--dim", type=int, default=1)
    parser.add_argument("--var-dim", action="append", default=[],
                        help="per-variable dimension override formatted as VAR=DIM; may be repeated")
    parser.add_argument("--regions", type=int, default=20)
    parser.add_argument("--c2-scale", type=float, default=2.0)
    parser.add_argument("--gen-bs", type=int, default=10000)

    parser.add_argument("--do-regime", default="obs", choices=["obs", "obs-x", "graph-default"],
                        help="data collection regimes to generate")
    parser.add_argument("--do-var-list-json",
                        help="explicit JSON list of do dictionaries, e.g. '[{}, {\"X\": 0}, {\"X\": 1}]'")

    parser.add_argument("--bound-treatment", default="X")
    parser.add_argument("--bound-treatment-value", type=int, default=0)
    parser.add_argument("--bound-do-json",
                        help="explicit bound intervention dictionary, e.g. '{\"sort\": 0, \"saatid\": 0}'")
    parser.add_argument("--bound-outcome", default="Y")
    parser.add_argument("--bound-outcome-value", type=int, default=1)
    parser.add_argument("--bound-gap-min", type=float, default=0.1)
    parser.add_argument("--bound-mc-samples", type=int, default=1000000)
    parser.add_argument("--adjustment-gap-min", type=float, default=0.1,
                        help="minimum absolute gap between P(y|x) and at least one adjustment query")
    parser.add_argument("--no-adjustment-gap", action="store_true",
                        help="do not require adjustment queries to differ from the conditional query")

    parser.add_argument("--positivity-epsilon", type=float, default=None,
                        help="minimum true joint cell probability; defaults to 0.01 for <=3 nodes and 0.005 otherwise")
    parser.add_argument("--positivity-mc-samples", type=int, default=1000000)
    parser.add_argument("--min-empirical-cell-count", type=int, default=10)
    parser.add_argument("--no-positivity", action="store_true",
                        help="do not require true/empirical joint positivity")
    parser.add_argument("--max-retries", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.graph == "barley":
        if args.bound_treatment == "X":
            args.bound_treatment = "sort"
        if args.bound_outcome == "Y":
            args.bound_outcome = "protein"
    args.bound_do = _parse_bound_do(args)
    if args.n_trials < 0:
        raise ValueError("--n-trials must be nonnegative")
    if args.max_retries < 0:
        raise ValueError("--max-retries must be nonnegative")

    cg_file = "dat/cg/{}.cg".format(args.graph)
    cg = CausalGraph.read(cg_file)
    if args.positivity_epsilon is None:
        args.positivity_epsilon = _default_positivity_epsilon(cg)
    required_vars = set(args.bound_do) | {args.bound_outcome}
    if not required_vars.issubset(set(cg.v)):
        raise ValueError("graph {} does not contain {}".format(args.graph, sorted(required_vars)))

    do_var_list = _parse_do_var_list(args)
    v_size_overrides = parse_var_dim_overrides(args.var_dim, cg, strict=False)

    hyperparams = {
        "regions": args.regions,
        "c2-scale": args.c2_scale,
        "gen-bs": args.gen_bs,
        "do-var-list": do_var_list,
        "positivity": not args.no_positivity,
        "bound-gap-min": args.bound_gap_min,
        "adjustment-gap-min": args.adjustment_gap_min,
        "adjustment-gap-enabled": not args.no_adjustment_gap,
        "positivity-epsilon": args.positivity_epsilon,
        "min-empirical-cell-count": args.min_empirical_cell_count,
        "bound-treatment": args.bound_treatment,
        "bound-treatment-value": args.bound_treatment_value,
        "bound-outcome": args.bound_outcome,
        "bound-outcome-value": args.bound_outcome_value,
    }
    if args.bound_do_json is not None or len(args.bound_do) > 1:
        hyperparams["bound-do"] = dict(args.bound_do)
    if v_size_overrides:
        hyperparams["v-sizes"] = _build_v_sizes(cg, args.dim, {"v-sizes": v_size_overrides})

    trial_indices = args.trial_index if args.trial_index else range(args.n_trials)
    for trial_index in trial_indices:
        if trial_index < 0:
            raise ValueError("--trial-index values must be nonnegative")
        _run_trial(args, trial_index, cg, do_var_list, hyperparams)


if __name__ == "__main__":
    main()
