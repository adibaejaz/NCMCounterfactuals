"""Launcher for a shared-data chain-graph fit experiment.

This experiment compares:
- a standard FF-NCM baseline trained on the true chain graph
- masked FF-NCM variants across masking rules and cycle-penalty weights

For each trial, a single ``DataBundle`` is generated once and then reused
across every model configuration so that all comparisons are made on exactly
the same dataset.
"""

import json
import os
import argparse

from src.metric.queries import get_experimental_variables, get_query
from src.pipeline import DivergencePipeline, MaskedDivergencePipeline
from src.run import NCMRunner, MaskedNCMRunner
from src.run.data_setup import build_data_bundle
from src.scm.ctm import CTM
from src.scm.model_classes import XORModel, RoundModel
from src.scm.ncm import FF_NCM
from src.scm.ncm.masked_feedforward_ncm import MaskedFF_NCM


VALID_GENERATORS = {
    "ctm": CTM,
    "xor": XORModel,
    "round": RoundModel,
}


def _shared_hyperparams(args):
    do_var_list = get_experimental_variables("chain")
    eval_query, _ = get_query("chain", args.query_track)
    return {
        "lr": args.lr,
        "data-bs": args.data_bs,
        "ncm-bs": args.ncm_bs,
        "h-layers": args.h_layers,
        "h-size": args.h_size,
        "u-size": args.u_size,
        "neural-pu": args.neural_pu,
        "layer-norm": args.layer_norm,
        "regions": args.regions,
        "c2-scale": 2.0 if args.scale_regions else 1.0,
        "gen-bs": args.gen_bs,
        "query-track": args.query_track,
        "do-var-list": do_var_list,
        "eval-query": eval_query,
        "full-batch": args.full_batch,
        "positivity": not args.no_positivity,
    }


def _masked_hyperparams(shared_hp, args, mask_mode, cycle_lambda):
    hp = dict(shared_hp)
    hp.update({
        "mask-mode": mask_mode,
        "mask-threshold": args.mask_threshold,
        "gate-sharpness": args.gate_sharpness,
        "max-iters": args.max_iters,
        "tol": args.tol,
        "use-dag-updates": args.use_dag_updates,
        "learn-mask": True,
        "mask-init-mode": args.mask_init_mode,
        "mask-init-value": args.mask_init_value,
        "mask-init-range": (args.mask_init_low, args.mask_init_high),
        "mask-fixed-zero-edges": _parse_fixed_zero_edges(args.mask_fixed_zero),
        "cycle-lambda": cycle_lambda,
        "cycle-penalty": args.cycle_penalty,
        "dagma-s": args.dagma_s,
    })
    return hp


def _parse_fixed_zero_edges(specs):
    edges = []
    for spec in specs:
        if "->" not in spec:
            raise ValueError("fixed-zero edge '{}' must have form SRC->DST".format(spec))
        src, dst = spec.split("->", 1)
        src = src.strip()
        dst = dst.strip()
        if not src or not dst:
            raise ValueError("fixed-zero edge '{}' must have non-empty SRC and DST".format(spec))
        edges.append((src, dst))
    return edges


def main():
    parser = argparse.ArgumentParser(description="Shared-data chain fit experiment")
    parser.add_argument("name", help="experiment root name")
    parser.add_argument("--gen", default="ctm", choices=sorted(VALID_GENERATORS.keys()))
    parser.add_argument("--query-track", default="ate")
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--dim", type=int, default=1)
    parser.add_argument("--gpu")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="index of this worker shard")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="total number of worker shards")

    parser.add_argument("--lr", type=float, default=4e-3)
    parser.add_argument("--data-bs", type=int, default=1000)
    parser.add_argument("--ncm-bs", type=int, default=1000)
    parser.add_argument("--h-layers", type=int, default=2)
    parser.add_argument("--h-size", type=int, default=128)
    parser.add_argument("--u-size", type=int, default=1)
    parser.add_argument("--neural-pu", action="store_true")
    parser.add_argument("--layer-norm", action="store_true")
    parser.add_argument("--full-batch", action="store_true")

    parser.add_argument("--regions", type=int, default=20)
    parser.add_argument("--scale-regions", action="store_true")
    parser.add_argument("--gen-bs", type=int, default=10000)
    parser.add_argument("--no-positivity", action="store_true")

    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--gate-sharpness", type=float, default=10.0)
    parser.add_argument("--max-iters", type=int, default=5)
    parser.add_argument("--tol", type=float, default=None)
    parser.add_argument("--use-dag-updates", action="store_true")
    parser.add_argument("--mask-init-mode", default="constant",
                        choices=["constant", "uniform", "zeros", "ones"])
    parser.add_argument("--mask-init-value", type=float, default=0.5)
    parser.add_argument("--mask-init-low", type=float, default=0.25)
    parser.add_argument("--mask-init-high", type=float, default=0.75)
    parser.add_argument("--mask-fixed-zero", action="append", default=[],
                        help="edge to constrain to 0, formatted as SRC->DST; may be repeated")
    parser.add_argument("--cycle-penalty", default="notears", choices=["notears", "dagma"])
    parser.add_argument("--dagma-s", type=float, default=1.0)

    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("shard-index must satisfy 0 <= shard-index < num-shards")

    dat_model = VALID_GENERATORS[args.gen]
    gpu_used = 0 if args.gpu is None else [int(args.gpu)]
    cg_file = "dat/cg/chain.cg"

    shared_hp = _shared_hyperparams(args)
    baseline_runner = NCMRunner(DivergencePipeline, dat_model, FF_NCM)
    masked_runner = MaskedNCMRunner(MaskedDivergencePipeline, dat_model, MaskedFF_NCM)

    summaries = []
    mask_modes = ["threshold", "gate", "multiply"]
    cycle_lambdas = [0.0, 0.1]
    jobs = []

    for trial_index in range(args.n_trials):
        jobs.append({
            "family": "baseline",
            "mask_mode": None,
            "cycle_lambda": None,
            "trial_index": trial_index,
        })
        for mask_mode in mask_modes:
            for cycle_lambda in cycle_lambdas:
                jobs.append({
                    "family": "masked",
                    "mask_mode": mask_mode,
                    "cycle_lambda": cycle_lambda,
                    "trial_index": trial_index,
                })

    print("total_jobs:", len(jobs))
    print("shard_index:", args.shard_index)
    print("num_shards:", args.num_shards)

    for job_idx, job in enumerate(jobs):
        if job_idx % args.num_shards != args.shard_index:
            continue

        trial_index = job["trial_index"]
        key = baseline_runner.get_key(cg_file, args.n_samples, args.dim, trial_index)
        data_seed = baseline_runner.get_data_seed(key)
        data_bundle = build_data_bundle(
            dat_model,
            cg_file,
            args.n_samples,
            args.dim,
            shared_hp,
            data_seed,
        )

        if job["family"] == "baseline":
            baseline_exp = "{}/baseline".format(args.name)
            _, baseline_results = baseline_runner.run(
                baseline_exp,
                cg_file,
                args.n_samples,
                args.dim,
                trial_index,
                hyperparams=dict(shared_hp),
                gpu=gpu_used,
                data_bundle=data_bundle,
                verbose=args.verbose,
            )
            summaries.append({
                "family": "baseline",
                "mask_mode": None,
                "cycle_lambda": None,
                "trial_index": trial_index,
                "results": baseline_results,
            })
        else:
            mask_mode = job["mask_mode"]
            cycle_lambda = job["cycle_lambda"]
            masked_hp = _masked_hyperparams(shared_hp, args, mask_mode, cycle_lambda)
            exp_name = "{}/masked_{}_cycle{}".format(
                args.name,
                mask_mode,
                str(cycle_lambda).replace(".", "p"),
            )
            _, masked_results = masked_runner.run(
                exp_name,
                cg_file,
                args.n_samples,
                args.dim,
                trial_index,
                hyperparams=masked_hp,
                gpu=gpu_used,
                data_bundle=data_bundle,
                verbose=args.verbose,
            )
            summaries.append({
                "family": "masked",
                "mask_mode": mask_mode,
                "cycle_lambda": cycle_lambda,
                "trial_index": trial_index,
                "results": masked_results,
            })

    out_dir = os.path.join("out", args.name)
    os.makedirs(out_dir, exist_ok=True)
    summary_name = "summary.json" if args.num_shards == 1 else "summary_shard{}.json".format(args.shard_index)
    with open(os.path.join(out_dir, summary_name), "w") as f:
        json.dump(summaries, f)


if __name__ == "__main__":
    main()
