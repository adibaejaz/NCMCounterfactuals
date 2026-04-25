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
from src.run.data_setup import _build_dat_model, _build_stored_metrics
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
        return [{}, {args.bound_treatment: 0}, {args.bound_treatment: 1}]
    if args.do_regime == "graph-default":
        return get_experimental_variables(args.graph)
    raise ValueError("unknown do regime: {}".format(args.do_regime))


def _v_sizes(cg, dim):
    return {k: 1 if k in {"X", "Y", "M", "W"} else dim for k in cg}


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
    stored_metrics = _build_stored_metrics(dat_m, dat_sets, do_var_list)
    return dat_m, dat_sets, stored_metrics


def _bound_diagnostics(dat_m, args):
    metrics = scm_query_bound_metrics(
        dat_m,
        graph_name=args.graph,
        outcome_var=args.bound_outcome,
        outcome_value=args.bound_outcome_value,
        treatment_var=args.bound_treatment,
        treatment_values=(args.bound_treatment_value,),
        n=args.bound_mc_samples,
    )
    do_name = serialize_probability(
        {args.bound_outcome: args.bound_outcome_value},
        do_vals={args.bound_treatment: args.bound_treatment_value})
    lower_key = "true_lower_{}".format(do_name)
    upper_key = "true_upper_{}".format(do_name)
    lower = float(metrics[lower_key])
    upper = float(metrics[upper_key])
    return {
        "query": do_name,
        "bound_mc_samples": int(args.bound_mc_samples),
        "lower_key": lower_key,
        "upper_key": upper_key,
        "lower": lower,
        "upper": upper,
        "gap": upper - lower,
        "passed": bool((upper - lower) >= args.bound_gap_min),
        "metrics": {k: float(v) for (k, v) in metrics.items()},
    }


def _trial_directory(root, graph, n, dim, trial_index):
    return os.path.join(
        root,
        "graph={}-n_samples={}-dim={}-trial_index={}".format(
            graph, n, dim, trial_index))


def _write_json(path, obj):
    with open(path, "w") as file:
        json.dump(obj, file, indent=2, sort_keys=True)


def _append_jsonl(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as file:
        file.write(json.dumps(obj, sort_keys=True) + "\n")


def _save_accepted(
        out_dir,
        dat_sets,
        stored_metrics,
        hyperparams,
        metadata,
        bound_diagnostics,
        positivity_diagnostics):
    os.makedirs(out_dir, exist_ok=True)
    T.save(dat_sets, os.path.join(out_dir, "dat.th"))
    T.save(stored_metrics, os.path.join(out_dir, "stored_metrics.th"))
    _write_json(os.path.join(out_dir, "hyperparams.json"), {k: str(v) for (k, v) in hyperparams.items()})
    _write_json(os.path.join(out_dir, "data_metadata.json"), metadata)
    _write_json(os.path.join(out_dir, "ground_truth_bounds.json"), bound_diagnostics)
    _write_json(os.path.join(out_dir, "positivity_diagnostics.json"), positivity_diagnostics)


def _candidate_seed(args, trial_index, retry_index, hyperparams, do_var_list):
    return _stable_seed({
        "purpose": "paper_bound_dataset_generation",
        "graph": args.graph,
        "trial_index": trial_index,
        "retry_index": retry_index,
        "n_samples": args.n_samples,
        "dim": args.dim,
        "hyperparams": hyperparams,
        "do_var_list": do_var_list,
        "bound": {
            "outcome": args.bound_outcome,
            "outcome_value": args.bound_outcome_value,
            "treatment": args.bound_treatment,
            "treatment_value": args.bound_treatment_value,
        },
    })


def _run_trial(args, trial_index, cg, do_var_list, hyperparams):
    out_dir = _trial_directory(args.name, args.graph, args.n_samples, args.dim, trial_index)
    if os.path.isfile(os.path.join(out_dir, "dat.th")) and not args.overwrite:
        print("[done]", out_dir)
        return

    reject_log = os.path.join(out_dir, "rejected_candidates.jsonl")
    if args.overwrite and os.path.isfile(reject_log):
        os.remove(reject_log)

    v_sizes = _v_sizes(cg, args.dim)
    for retry_index in range(args.max_retries + 1):
        seed = _candidate_seed(args, trial_index, retry_index, hyperparams, do_var_list)
        print("[candidate] graph={} trial={} retry={} seed={}".format(
            args.graph, trial_index, retry_index, seed))

        dat_m, dat_sets, stored_metrics = _generate_candidate(
            cg=cg,
            n=args.n_samples,
            dim=args.dim,
            hyperparams=hyperparams,
            do_var_list=do_var_list,
            seed=seed)
        bound = _bound_diagnostics(dat_m, args)
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
            "generator": "CTM",
            "do_var_list": do_var_list,
            "accepted": len(reject_reasons) == 0,
            "reject_reasons": reject_reasons,
            "bound_gap_min": float(args.bound_gap_min),
            "positivity_epsilon": float(args.positivity_epsilon),
            "min_empirical_cell_count": int(args.min_empirical_cell_count),
        }

        if not reject_reasons:
            _save_accepted(
                out_dir=out_dir,
                dat_sets=dat_sets,
                stored_metrics=stored_metrics,
                hyperparams=hyperparams,
                metadata=metadata,
                bound_diagnostics=bound,
                positivity_diagnostics=positivity)
            print("[accepted]", out_dir)
            return

        _append_jsonl(reject_log, {
            **metadata,
            "bound_gap": bound["gap"],
            "bound_lower": bound["lower"],
            "bound_upper": bound["upper"],
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
    parser.add_argument("--regions", type=int, default=20)
    parser.add_argument("--c2-scale", type=float, default=1.0)
    parser.add_argument("--gen-bs", type=int, default=10000)

    parser.add_argument("--do-regime", default="obs", choices=["obs", "obs-x", "graph-default"],
                        help="data collection regimes to generate")
    parser.add_argument("--do-var-list-json",
                        help="explicit JSON list of do dictionaries, e.g. '[{}, {\"X\": 0}, {\"X\": 1}]'")

    parser.add_argument("--bound-treatment", default="X")
    parser.add_argument("--bound-treatment-value", type=int, default=0)
    parser.add_argument("--bound-outcome", default="Y")
    parser.add_argument("--bound-outcome-value", type=int, default=1)
    parser.add_argument("--bound-gap-min", type=float, default=0.1)
    parser.add_argument("--bound-mc-samples", type=int, default=1000000)

    parser.add_argument("--positivity-epsilon", type=float, default=0.01)
    parser.add_argument("--positivity-mc-samples", type=int, default=1000000)
    parser.add_argument("--min-empirical-cell-count", type=int, default=10)
    parser.add_argument("--max-retries", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.n_trials < 0:
        raise ValueError("--n-trials must be nonnegative")
    if args.max_retries < 0:
        raise ValueError("--max-retries must be nonnegative")

    cg_file = "dat/cg/{}.cg".format(args.graph)
    cg = CausalGraph.read(cg_file)
    required_vars = {args.bound_treatment, args.bound_outcome}
    if not required_vars.issubset(set(cg.v)):
        raise ValueError("graph {} does not contain {}".format(args.graph, sorted(required_vars)))

    do_var_list = _parse_do_var_list(args)
    hyperparams = {
        "regions": args.regions,
        "c2-scale": args.c2_scale,
        "gen-bs": args.gen_bs,
        "do-var-list": do_var_list,
        "positivity": True,
        "bound-gap-min": args.bound_gap_min,
        "positivity-epsilon": args.positivity_epsilon,
        "min-empirical-cell-count": args.min_empirical_cell_count,
        "bound-treatment": args.bound_treatment,
        "bound-treatment-value": args.bound_treatment_value,
        "bound-outcome": args.bound_outcome,
        "bound-outcome-value": args.bound_outcome_value,
    }

    trial_indices = args.trial_index if args.trial_index else range(args.n_trials)
    for trial_index in trial_indices:
        if trial_index < 0:
            raise ValueError("--trial-index values must be nonnegative")
        _run_trial(args, trial_index, cg, do_var_list, hyperparams)


if __name__ == "__main__":
    main()
