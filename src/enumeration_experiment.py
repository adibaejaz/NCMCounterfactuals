import argparse
import os
import glob
import json

import numpy as np
import torch as T

from src.ds import CTF, CTFTerm
from src.ds.causal_graph import CausalGraph
from src.metric.queries import get_experimental_variables, get_query
from src.pipeline import DivergencePipeline
from src.run import EnumerationNCMRunner
from src.run.data_setup import build_reused_data_bundle
from src.scm.ctm import CTM
from src.scm.model_classes import RoundModel, XORModel
from src.scm.ncm import FF_NCM

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

VALID_GENERATORS = {
    "ctm": CTM,
    "xor": XORModel,
    "round": RoundModel,
}

VALID_GRAPHS = {
    "backdoor", "bow", "frontdoor", "napkin", "simple", "chain", "bdm", "med", "expl", "double_bow", "iv", "bad_fd",
    "extended_bow", "bad_m", "m", "square", "four_clique",
    "zid_a", "zid_b", "zid_c",
    "gid_a", "gid_b", "gid_c", "gid_d",
    "med_c1", "med_c2",
    "expl_xm", "expl_xm_dox", "expl_xy", "expl_dox", "expl_xy_dox", "expl_my", "expl_my_dox",
}


def _build_bound_queries(graph_name, treatment_var, treatment_value, outcome_var, outcome_value):
    eval_query = CTF(
        {CTFTerm({outcome_var}, {treatment_var: treatment_value}, {outcome_var: outcome_value})},
        set(),
        name="P({}={} | do({}={}))".format(outcome_var, outcome_value, treatment_var, treatment_value),
    )
    query_bounds = {
        "graph_name": graph_name,
        "outcome_var": outcome_var,
        "outcome_value": outcome_value,
        "treatment_var": treatment_var,
        "treatment_values": (treatment_value,),
    }
    return eval_query, query_bounds


def _graph_has_vars(graph_name, required_vars):
    cg = CausalGraph.read("dat/cg/{}.cg".format(graph_name))
    return set(required_vars).issubset(set(cg.v))


def _find_reuse_source(root_dir, graph, n, dim, trial_index, train_seed_offset):
    pattern = os.path.join(
        root_dir,
        "gen=CTM-graph={}-n_samples={}-dim={}-trial_index={}-run=*".format(graph, n, dim, trial_index),
        "hyperparams.json",
    )
    matches = []
    for hp_path in glob.glob(pattern):
        with open(hp_path) as file:
            hyperparams = json.load(file)
        if int(hyperparams.get("train-seed-offset", 0)) == int(train_seed_offset):
            matches.append(os.path.dirname(hp_path))
    if not matches:
        raise ValueError(
            "no reuse-data source found under {} for graph={}, n={}, dim={}, trial_index={}, train-seed-offset={}".format(
                root_dir, graph, n, dim, trial_index, train_seed_offset))
    if len(matches) > 1:
        raise ValueError(
            "multiple reuse-data sources found under {} for graph={}, n={}, dim={}, trial_index={}, train-seed-offset={}: {}".format(
                root_dir, graph, n, dim, trial_index, train_seed_offset, matches))
    return matches[0]


def build_parser():
    parser = argparse.ArgumentParser(description="Enumerate a DAG equivalence class and train one DAG NCM per member")
    parser.add_argument("name", help="name of the experiment")
    parser.add_argument("--equiv-class-file", required=True,
                        help="path to a partially directed graph file with -- for undirected edges")
    parser.add_argument("--max-enum-dags", type=int, default=None,
                        help="optional cap on the number of enumerated DAGs")
    parser.add_argument("--gen", default="ctm", choices=sorted(VALID_GENERATORS.keys()))
    parser.add_argument("--graph", default="backdoor", choices=sorted(VALID_GRAPHS))
    parser.add_argument("--query-track", default="ate")
    parser.add_argument("--bound-query", action="store_true")
    parser.add_argument("--bound-treatment", default="X")
    parser.add_argument("--bound-outcome", default="Y")
    parser.add_argument("--bound-outcome-value", type=int, default=1)
    parser.add_argument("--bound-treatment-value", action="append", type=int, default=[])
    parser.add_argument("--n-trials", type=int, default=1)
    parser.add_argument("--trial-index", action="append", type=int, default=[])
    parser.add_argument("--id-reruns", type=int, default=1,
                        help="number of training-seed reruns per dataset; each rerun takes min/max over the DAG class")
    parser.add_argument("--train-seed-offset", type=int, default=0,
                        help="offset added to the per-DAG training seed without changing the data seed")
    parser.add_argument("--reuse-data-from",
                        help="path to a saved run directory or dat.th file whose exact dataset should be reused")
    parser.add_argument("--reuse-data-root",
                        help="root directory of saved runs to search by trial_index and train-seed-offset")
    parser.add_argument("--data-seed-search-limit", type=int, default=1000,
                        help="max increments past the base data seed when recovering a reused dataset")
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--dim", type=int, default=1)
    parser.add_argument("--gpu")
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
    parser.add_argument("--verbose", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.query_track is not None:
        args.query_track = args.query_track.lower()

    if args.bound_query and not _graph_has_vars(args.graph, {args.bound_treatment, args.bound_outcome}):
        raise ValueError("graph {} does not contain {} and {}".format(
            args.graph, args.bound_treatment, args.bound_outcome))

    runner = EnumerationNCMRunner(DivergencePipeline, VALID_GENERATORS[args.gen], FF_NCM)
    gpu_used = 0 if args.gpu is None else [int(args.gpu)]

    hyperparams = {
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
        "full-batch": args.full_batch,
        "positivity": not args.no_positivity,
        "equiv-class-file": args.equiv_class_file,
        "max-enum-dags": args.max_enum_dags,
        "id-reruns": args.id_reruns,
        "train-seed-offset": args.train_seed_offset,
    }

    if args.trial_index:
        trial_indices = args.trial_index
    else:
        trial_indices = range(args.n_trials)

    do_var_list = get_experimental_variables(args.graph)
    if args.bound_query:
        bound_values = args.bound_treatment_value if args.bound_treatment_value else [0, 1]
        query_jobs = [
            _build_bound_queries(
                args.graph,
                args.bound_treatment,
                treatment_value,
                args.bound_outcome,
                args.bound_outcome_value,
            )
            for treatment_value in bound_values
        ]
    else:
        eval_query, _ = get_query(args.graph, args.query_track)
        query_jobs = [(eval_query, None)]

    for eval_query, query_bounds in query_jobs:
        hyperparams["do-var-list"] = do_var_list
        hyperparams["eval-query"] = eval_query
        hyperparams["query-track"] = eval_query.name if args.bound_query else args.query_track
        if query_bounds is not None:
            hyperparams["query-bound-spec"] = query_bounds
        else:
            hyperparams.pop("query-bound-spec", None)

        n = int(args.n_samples)
        d = int(args.dim)
        hyperparams["data-bs"] = min(args.data_bs, n)
        hyperparams["ncm-bs"] = min(args.ncm_bs, n)
        if args.full_batch:
            hyperparams["data-bs"] = n
            hyperparams["ncm-bs"] = n

        for trial_index in trial_indices:
            if trial_index < 0:
                raise ValueError("--trial-index values must be nonnegative")
            cg_file = "dat/cg/{}.cg".format(args.graph)
            data_bundle = None
            if args.reuse_data_from or args.reuse_data_root:
                if args.reuse_data_from and args.reuse_data_root:
                    raise ValueError("use only one of --reuse-data-from or --reuse-data-root")
                source_path = args.reuse_data_from
                if args.reuse_data_root:
                    source_path = _find_reuse_source(
                        args.reuse_data_root,
                        args.graph,
                        n,
                        d,
                        trial_index,
                        args.train_seed_offset,
                    )
                key = runner.get_key(cg_file, n, d, trial_index)
                base_seed = runner.get_data_seed(key)
                data_bundle, recovered_seed = build_reused_data_bundle(
                    VALID_GENERATORS[args.gen],
                    cg_file,
                    n,
                    d,
                    hyperparams,
                    base_seed,
                    source_path,
                    max_seed_steps=args.data_seed_search_limit,
                )
                print("Reused data from:", source_path)
                print("Recovered data seed:", recovered_seed)
            runner.run(
                args.name,
                cg_file,
                n,
                d,
                trial_index,
                hyperparams=hyperparams,
                gpu=gpu_used,
                data_bundle=data_bundle,
                verbose=args.verbose,
            )


if __name__ == "__main__":
    main()
