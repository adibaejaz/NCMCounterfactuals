"""Minimal entrypoint for masked FF-NCM estimation experiments.

This is a masked counterpart to ``src/main.py`` with a deliberately smaller
surface area:
- divergence training only
- masked feedforward NCM only
- single-model runner only

It is intended as the cleanest way to launch masked experiments without
threading masked-specific arguments through the older unmasked CLI.
"""

import itertools
import os
import argparse

import numpy as np
import torch as T

from src.ds import CTF, CTFTerm
from src.ds.causal_graph import CausalGraph
from src.metric.queries import get_query, get_experimental_variables, is_q_id_in_G
from src.pipeline import MaskedDivergencePipeline
from src.run import MaskedNCMMinMaxRunner, MaskedNCMRunner
from src.scm.ctm import CTM
from src.scm.masked_scm import (
    DEFAULT_GATE_SHARPNESS,
    DEFAULT_MASK_MODE,
    DEFAULT_MASK_THRESHOLD,
    DEFAULT_MAX_ITERS,
    DEFAULT_TOL,
    DEFAULT_USE_DAG_UPDATES,
)
from src.scm.model_classes import XORModel, RoundModel
from src.scm.ncm.masked_feedforward_ncm import MaskedFF_NCM

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

valid_generators = {
    "ctm": CTM,
    "xor": XORModel,
    "round": RoundModel,
}

valid_graphs = {
    "backdoor", "bow", "frontdoor", "napkin", "simple", "chain", "bdm", "med", "expl", "double_bow", "iv", "bad_fd",
    "extended_bow", "bad_m", "m",
    "zid_a", "zid_b", "zid_c",
    "gid_a", "gid_b", "gid_c", "gid_d",
    "med_c1", "med_c2",
    "expl_xm", "expl_xm_dox", "expl_xy", "expl_dox", "expl_xy_dox", "expl_my", "expl_my_dox",
}

graph_sets = {
    "all": {"backdoor", "bow", "frontdoor", "napkin", "simple", "chain", "med", "expl", "zid_a", "zid_b", "zid_c",
            "gid_a", "gid_b", "gid_c", "gid_d"},
    "standard": {"backdoor", "bow", "frontdoor", "napkin", "chain", "iv", "extended_bow", "m", "bad_m", "bdm"},
    "zid": {"zid_a", "zid_b", "zid_c"},
    "gid": {"gid_a", "gid_b", "gid_c", "gid_d"},
    "expl_set": {"expl", "expl_dox", "expl_xm", "expl_xm_dox", "expl_xy", "expl_xy_dox", "expl_my", "expl_my_dox"},
}

valid_queries = {"ate", "ett", "nde", "ctfde"}


parser = argparse.ArgumentParser(description="Masked FF-NCM Runner")
parser.add_argument('name', help="name of the experiment")

parser.add_argument('--gen', default="ctm", help="data generating model (default: ctm)")

parser.add_argument('--lr', type=float, default=4e-3, help="optimizer learning rate (default: 4e-3)")
parser.add_argument('--theta-lr', type=float, default=None,
                    help="optimizer learning rate for neural parameters; defaults to --lr")
parser.add_argument('--mask-lr', type=float, default=None,
                    help="optimizer learning rate for mask parameters; defaults to --lr")
parser.add_argument('--data-bs', type=int, default=1000, help="batch size of data (default: 1000)")
parser.add_argument('--ncm-bs', type=int, default=1000, help="batch size of NCM samples (default: 1000)")
parser.add_argument('--h-layers', type=int, default=2, help="number of hidden layers (default: 2)")
parser.add_argument('--h-size', type=int, default=128, help="neural network hidden layer size (default: 128)")
parser.add_argument('--u-size', type=int, default=1, help="dimensionality of U variables (default: 1)")
parser.add_argument('--neural-pu', action="store_true", help="use neural parameters in U distributions")
parser.add_argument('--layer-norm', action="store_true", help="set flag to use layer norm")

parser.add_argument('--full-batch', action="store_true", help="use n as the batch size")

parser.add_argument('--regions', type=int, default=20, help="number of regions for CTM (default: 20)")
parser.add_argument('--scale-regions', action="store_true", help="scale regions by C-cliques in CTM")
parser.add_argument('--gen-bs', type=int, default=10000, help="batch size of ctm data generation (default: 10000)")
parser.add_argument('--no-positivity', action="store_true", help="does not enforce positivity for ID experiments")

parser.add_argument("--query-track", default="ate", help="choice of query to track")
parser.add_argument("--bound-query", action="store_true", help="run a masked min-max bound experiment for P(Y=1 | do(Z=z))")
parser.add_argument("--bound-treatment", default="Z", help="treatment variable for bound experiments")
parser.add_argument("--bound-outcome", default="Y", help="outcome variable for bound experiments")
parser.add_argument("--bound-outcome-value", type=int, default=1, help="outcome value for bound experiments")
parser.add_argument("--bound-treatment-value", action="append", type=int, default=[],
                    help="treatment value for bound experiments; may be repeated")
parser.add_argument("--id-reruns", type=int, default=1, help="number of min-max reruns")
parser.add_argument("--train-seed-offset", type=int, default=0,
                    help="offset added to min-max model initialization seeds without changing data seeds")
parser.add_argument("--max-query-iters", type=int, default=3000, help="number of min-max training epochs")
parser.add_argument("--max-lambda", type=float, default=1.0, help="initial query regularization weight")
parser.add_argument("--min-lambda", type=float, default=0.001, help="final query regularization weight")
parser.add_argument("--mc-sample-size", type=int, default=10000, help="sample size for query optimization")
parser.add_argument("--theta-only-extra-epochs", type=int, default=0,
                    help="after min-max training, freeze learned masks and train theta only for this many extra epochs")
parser.add_argument("--theta-only-extra-lr", type=float, default=None,
                    help="theta learning rate for --theta-only-extra-epochs; defaults to --theta-lr/--lr")

parser.add_argument('--graph', '-G', default="standard", help="name of preset graph")
parser.add_argument('--n-trials', '-t', type=int, default=1, help="number of trials")
parser.add_argument('--trial-index', action="append", type=int, default=[],
                    help="specific trial index to run; may be repeated. Overrides --n-trials when provided")
parser.add_argument('--n-samples', '-n', type=int, default=10000, help="number of samples (default: 10000)")
parser.add_argument('--dim', '-d', type=int, default=1, help="dimensionality of variables (default: 1)")
parser.add_argument('--gpu', help="GPU to use")

parser.add_argument('--mask-mode', default=DEFAULT_MASK_MODE,
                    choices=["threshold", "multiply", "gate", "st-gate"],
                    help="masking rule; st-gate uses hard forward gates with sigmoid straight-through gradients")
parser.add_argument('--mask-threshold', type=float, default=DEFAULT_MASK_THRESHOLD,
                    help="threshold for threshold/gate masking")
parser.add_argument('--gate-sharpness', type=float, default=DEFAULT_GATE_SHARPNESS,
                    help="sigmoid sharpness for gate masking")
parser.add_argument('--max-iters', type=int, default=DEFAULT_MAX_ITERS,
                    help="max synchronous iterations")
parser.add_argument('--tol', type=float, default=None, help="optional convergence tolerance")
parser.add_argument('--use-dag-updates', action="store_true",
                    help="use DAG-style ordered updates instead of synchronous updates")
parser.add_argument('--learn-mask', action="store_true", help="learn the mask values")
parser.add_argument('--fixed-mask', action="store_true", help="keep the mask fixed")
parser.add_argument('--mask-init-mode', default="constant",
                    choices=["constant", "uniform", "zeros", "ones"],
                    help="initialization mode for the realized mask")
parser.add_argument('--mask-init-value', type=float, default=0.5,
                    help="constant initialization value for the realized mask")
parser.add_argument('--mask-init-low', type=float, default=0.25,
                    help="lower bound for uniform mask initialization")
parser.add_argument('--mask-init-high', type=float, default=0.75,
                    help="upper bound for uniform mask initialization")
parser.add_argument('--mask-fixed-zero', action='append', default=[],
                    help="edge to constrain to 0, formatted as SRC->DST; may be repeated")
parser.add_argument('--cycle-lambda', type=float, default=0.0,
                    help="weight on the DAG penalty")
parser.add_argument('--cycle-penalty', default="dagma", choices=["notears", "dagma"],
                    help="DAG penalty type")
parser.add_argument('--dagma-s', type=float, default=1.0,
                    help="DAGMA log-det scale parameter")
parser.add_argument('--mask-l1-lambda', type=float, default=1.0,
                    help="weight on L1 regularization of realized mask entries; set to 0 to disable")
parser.add_argument('--mask-binary-lambda', type=float, default=0.0,
                    help="weight on regularization that pushes realized mask entries away from 0.5")
parser.add_argument('--dag-alm', action="store_true",
                    help="use a simplified augmented Lagrangian for the NOTEARS acyclicity constraint")
parser.add_argument('--alm-alpha-init', type=float, default=0.0,
                    help="initial dual variable for the DAG augmented Lagrangian")
parser.add_argument('--alm-rho-init', type=float, default=1.0,
                    help="initial quadratic penalty weight for the DAG augmented Lagrangian")
parser.add_argument('--alm-rho-mult', type=float, default=5.0,
                    help="multiplicative increase for the DAG augmented Lagrangian penalty weight")
parser.add_argument('--alm-rho-max', type=float, default=1e4,
                    help="maximum quadratic penalty weight for the DAG augmented Lagrangian")
parser.add_argument('--alm-update-every', type=int, default=50,
                    help="epochs between DAG augmented Lagrangian dual updates")
parser.add_argument('--alm-improve-ratio', type=float, default=0.9,
                    help="required relative improvement in dag_h before keeping the current ALM penalty weight")
parser.add_argument('--alt-opt', action="store_true",
                    help="alternate updates between neural parameters and the mask")
parser.add_argument('--theta-steps-per-mask', type=int, default=5,
                    help="number of theta-only updates per mask phase in alternating optimization")
parser.add_argument('--mask-steps-per-theta', type=int, default=1,
                    help="number of mask-only updates per theta phase in alternating optimization")
parser.add_argument('--log-grad-norms', action="store_true",
                    help="log mask/theta gradient norm diagnostics during training")

parser.add_argument('--verbose', action="store_true", help="print more information")


def _print_query_info(eval_query):
    sign_char = {1: '+', -1: '-'}
    if isinstance(eval_query, CTF):
        print("Eval Query: {}".format(eval_query))
        print()
        return

    print("Eval Query: ", end="")
    for quer, sig in eval_query:
        print("{} {} ".format(sign_char[sig], quer), end="")
    print("\n")


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


def _build_bound_queries(graph_name, treatment_var, treatment_value, outcome_var, outcome_value):
    eval_query = CTF(
        {CTFTerm({outcome_var}, {treatment_var: treatment_value}, {outcome_var: outcome_value})},
        set(),
        name="P({}={} | do({}={}))".format(outcome_var, outcome_value, treatment_var, treatment_value),
    )
    opt_query = (
        eval_query,
        CTF(
            {CTFTerm({outcome_var}, {treatment_var: treatment_value}, {outcome_var: 1 - outcome_value})},
            set(),
            name="P({}={} | do({}={}))".format(outcome_var, 1 - outcome_value, treatment_var, treatment_value),
        ),
    )
    query_bounds = {
        "graph_name": graph_name,
        "outcome_var": outcome_var,
        "outcome_value": outcome_value,
        "treatment_var": treatment_var,
        "treatment_values": (treatment_value,),
    }
    return eval_query, opt_query, query_bounds


def _graph_has_vars(graph_name, required_vars):
    cg = CausalGraph.read("dat/cg/{}.cg".format(graph_name))
    return set(required_vars).issubset(set(cg.v))


def main():
    args = parser.parse_args()

    gen_choice = args.gen.lower()
    graph_choice = args.graph.lower()
    bound_mode = args.bound_query
    query_track = args.query_track.lower() if args.query_track is not None and not bound_mode else None

    assert gen_choice in valid_generators
    assert graph_choice in valid_graphs or graph_choice in graph_sets
    assert bound_mode or query_track is None or query_track in valid_queries
    assert not (args.learn_mask and args.fixed_mask)
    assert not args.alt_opt or not args.fixed_mask
    assert args.bound_outcome_value in {0, 1}
    assert args.theta_steps_per_mask >= 0
    assert args.mask_steps_per_theta >= 0
    assert (args.theta_steps_per_mask + args.mask_steps_per_theta) > 0
    assert args.theta_only_extra_epochs >= 0
    assert args.theta_only_extra_lr is None or args.theta_only_extra_lr > 0
    assert args.alm_rho_init > 0
    assert args.alm_rho_mult >= 1.0
    assert args.alm_rho_max >= args.alm_rho_init
    assert args.alm_update_every > 0
    assert 0 < args.alm_improve_ratio <= 1.0
    assert not args.dag_alm or args.cycle_penalty == "notears"

    pipeline = MaskedDivergencePipeline
    dat_model = valid_generators[gen_choice]
    ncm_model = MaskedFF_NCM
    runner_cls = MaskedNCMMinMaxRunner if bound_mode else MaskedNCMRunner
    runner = runner_cls(pipeline, dat_model, ncm_model)

    gpu_used = 0 if args.gpu is None else [int(args.gpu)]

    arg_data_bs = args.data_bs
    arg_ncm_bs = args.ncm_bs

    learn_mask = True
    if args.fixed_mask:
        learn_mask = False
    elif args.learn_mask:
        learn_mask = True

    hyperparams = {
        'lr': args.lr,
        'theta-lr': args.theta_lr if args.theta_lr is not None else args.lr,
        'mask-lr': args.mask_lr if args.mask_lr is not None else args.lr,
        'data-bs': args.data_bs,
        'ncm-bs': args.ncm_bs,
        'h-layers': args.h_layers,
        'h-size': args.h_size,
        'u-size': args.u_size,
        'neural-pu': args.neural_pu,
        'layer-norm': args.layer_norm,
        'regions': args.regions,
        'c2-scale': 2.0 if args.scale_regions else 1.0,
        'gen-bs': args.gen_bs,
        'query-track': query_track,
        'full-batch': args.full_batch,
        'positivity': not args.no_positivity,
        'mask-mode': args.mask_mode,
        'mask-threshold': args.mask_threshold,
        'gate-sharpness': args.gate_sharpness,
        'max-iters': args.max_iters,
        'tol': args.tol if args.tol is not None else DEFAULT_TOL,
        'use-dag-updates': args.use_dag_updates if args.use_dag_updates else DEFAULT_USE_DAG_UPDATES,
        'learn-mask': learn_mask,
        'mask-init-mode': args.mask_init_mode,
        'mask-init-value': args.mask_init_value,
        'mask-init-range': (args.mask_init_low, args.mask_init_high),
        'mask-fixed-zero-edges': _parse_fixed_zero_edges(args.mask_fixed_zero),
        'cycle-lambda': args.cycle_lambda,
        'cycle-penalty': args.cycle_penalty,
        'dagma-s': args.dagma_s,
        'mask-l1-lambda': args.mask_l1_lambda,
        'mask-binary-lambda': args.mask_binary_lambda,
        'dag-alm': args.dag_alm,
        'alm-alpha-init': args.alm_alpha_init,
        'alm-rho-init': args.alm_rho_init,
        'alm-rho-mult': args.alm_rho_mult,
        'alm-rho-max': args.alm_rho_max,
        'alm-update-every': args.alm_update_every,
        'alm-improve-ratio': args.alm_improve_ratio,
        'alt-opt': args.alt_opt,
        'theta-steps-per-mask': args.theta_steps_per_mask,
        'mask-steps-per-theta': args.mask_steps_per_theta,
        'log-grad-norms': args.log_grad_norms,
        'id-reruns': args.id_reruns,
        'train-seed-offset': args.train_seed_offset,
        'max-query-iters': args.max_query_iters,
        'max-lambda': args.max_lambda,
        'min-lambda': args.min_lambda,
        'mc-sample-size': args.mc_sample_size,
    }
    if args.theta_only_extra_epochs > 0:
        hyperparams['theta-only-extra-epochs'] = args.theta_only_extra_epochs
        hyperparams['theta-only-final-query-reg'] = True
        if args.theta_only_extra_lr is not None:
            hyperparams['theta-only-extra-lr'] = args.theta_only_extra_lr

    if graph_choice in graph_sets:
        if bound_mode:
            graph_set = {
                graph for graph in graph_sets[graph_choice]
                if _graph_has_vars(graph, {args.bound_treatment, args.bound_outcome})
            }
        else:
            graph_set = {graph for graph in graph_sets[graph_choice] if is_q_id_in_G(graph, query_track)}
    else:
        if bound_mode and not _graph_has_vars(graph_choice, {args.bound_treatment, args.bound_outcome}):
            raise ValueError("graph {} does not contain {} and {}".format(graph_choice, args.bound_treatment, args.bound_outcome))
        graph_set = {graph_choice}

    if bound_mode and not graph_set:
        raise ValueError("no selected graphs contain {} and {}".format(args.bound_treatment, args.bound_outcome))

    if args.n_samples == -1:
        n_list = 10.0 ** np.linspace(3, 5, 5)
    else:
        n_list = [args.n_samples]

    if args.dim == -1:
        d_list = [1, 16]
    else:
        d_list = [args.dim]

    bound_values = args.bound_treatment_value if args.bound_treatment_value else [0, 1]

    if args.trial_index:
        trial_indices = args.trial_index
        if any(i < 0 for i in trial_indices):
            raise ValueError("--trial-index values must be nonnegative")
    else:
        trial_indices = range(args.n_trials)

    for graph in graph_set:
        do_var_list = get_experimental_variables(graph)
        if bound_mode:
            query_jobs = [
                _build_bound_queries(
                    graph,
                    args.bound_treatment,
                    treatment_value,
                    args.bound_outcome,
                    args.bound_outcome_value,
                )
                for treatment_value in bound_values
            ]
        else:
            eval_query, _ = get_query(graph, query_track)
            query_jobs = [(eval_query, (None, None), None)]

        for eval_query, opt_queries, query_bounds in query_jobs:
            hyperparams["do-var-list"] = do_var_list
            hyperparams["eval-query"] = eval_query
            hyperparams["query-track"] = eval_query.name if bound_mode else query_track
            if bound_mode:
                hyperparams["max-query-1"] = opt_queries[0]
                hyperparams["max-query-2"] = opt_queries[1]
                hyperparams["query-bound-spec"] = query_bounds
            else:
                hyperparams.pop("max-query-1", None)
                hyperparams.pop("max-query-2", None)
                hyperparams.pop("query-bound-spec", None)

            if args.verbose:
                print("========== {} ==========".format(graph.upper()))
                print("Do Set: {}".format(do_var_list))
                _print_query_info(eval_query)
                if bound_mode:
                    print("Max Query: {}".format(opt_queries[0]))
                    print("Min Query: {}".format(opt_queries[1]))
                    print()

            for (n, d) in itertools.product(n_list, d_list):
                n = int(n)
                hyperparams["data-bs"] = min(arg_data_bs, n)
                hyperparams["ncm-bs"] = min(arg_ncm_bs, n)

                if arg_data_bs == -1:
                    hyperparams["data-bs"] = n // 10
                if arg_ncm_bs == -1:
                    hyperparams["ncm-bs"] = n // 10

                if hyperparams["full-batch"]:
                    hyperparams["data-bs"] = n
                    hyperparams["ncm-bs"] = n

                for i in trial_indices:
                    cg_file = "dat/cg/{}.cg".format(graph)
                    try:
                        if not runner.run(
                                args.name,
                                cg_file,
                                n,
                                d,
                                i,
                                hyperparams=hyperparams,
                                gpu=gpu_used,
                                verbose=args.verbose):
                            continue
                    except Exception as e:
                        print(e)
                        print("[failed]", i, args.name)
                        raise


if __name__ == "__main__":
    main()
