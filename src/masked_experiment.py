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
import json
from pathlib import Path

import numpy as np
import torch as T

from src.ds import CTF, CTFTerm
from src.ds.causal_graph import CausalGraph
from src.ds.equivalence_class import read_equivalence_class
from src.metric.queries import get_query, get_experimental_variables, is_q_id_in_G
from src.pipeline import MaskedDivergencePipeline
from src.run import MaskedNCMMinMaxRunner, MaskedNCMRunner
from src.run.data_setup import (
    DataBundle,
    _build_dat_model,
    _build_v_sizes,
    _parse_v_size_payload,
    parse_var_dim_overrides,
)
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
    "extended_bow", "bad_m", "m", "square", "four_clique", "barley",
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
parser.add_argument('--num-workers', type=int, default=0,
                    help="number of DataLoader workers (default: 0)")
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
parser.add_argument("--bound-query", action="store_true", help="run a masked min-max bound experiment for P(Y=1 | do(X=x))")
parser.add_argument("--bound-query-track", choices=sorted(valid_queries),
                    help="directly optimize a named query in bound mode, e.g. ate")
parser.add_argument("--bound-treatment", default="X", help="treatment variable for bound experiments")
parser.add_argument("--bound-outcome", default="Y", help="outcome variable for bound experiments")
parser.add_argument("--bound-outcome-value", type=int, default=1, help="outcome value for bound experiments")
parser.add_argument("--bound-treatment-value", action="append", type=int, default=[],
                    help="treatment value for bound experiments; may be repeated")
parser.add_argument("--id-reruns", type=int, default=1, help="number of min-max reruns")
parser.add_argument("--train-seed-offset", type=int, default=0,
                    help="offset added to min-max model initialization seeds without changing data seeds")
parser.add_argument("--max-query-iters", type=int, default=3000, help="number of min-max training epochs")
parser.add_argument("--early-stop-patience", type=int, default=100,
                    help="stop min-max training after this many epochs without sufficient selection_loss improvement")
parser.add_argument("--early-stop-min-delta", type=float, default=1e-6,
                    help="minimum selection_loss improvement required to reset min-max early stopping patience")
parser.add_argument("--max-lambda", type=float, default=1.0, help="initial query regularization weight")
parser.add_argument("--min-lambda", type=float, default=0.001, help="final query regularization weight")
parser.add_argument("--selection-query-lambda", type=float, default=None,
                    help="constant query weight for checkpoint/early-stopping selection loss; defaults to --min-lambda")
parser.add_argument("--mc-sample-size", type=int, default=10000, help="sample size for query optimization")
parser.add_argument("--query-update-target", choices=["mask", "theta", "all"], default="mask",
                    help="which parameter block receives query-loss gradients during alternating optimization")
parser.add_argument("--theta-only-extra-epochs", type=int, default=0,
                    help="after min-max training, freeze learned masks and train theta only for this many extra epochs")
parser.add_argument("--theta-only-extra-lr", type=float, default=None,
                    help="theta learning rate for --theta-only-extra-epochs; defaults to --theta-lr/--lr")
parser.add_argument("--no-theta-only-final-query-reg", dest="theta_only_final_query_reg",
                    action="store_false", default=True,
                    help="do not include the query regularizer during theta-only extra epochs")
parser.add_argument("--reuse-data-from",
                    help="path to one generated dataset directory or dat.th file to reuse")
parser.add_argument("--reuse-data-root",
                    help="root directory of generated datasets to reuse by graph/n/dim/trial_index")

parser.add_argument('--graph', '-G', default="standard", help="name of preset graph")
parser.add_argument('--n-trials', '-t', type=int, default=1, help="number of trials")
parser.add_argument('--trial-index', action="append", type=int, default=[],
                    help="specific trial index to run; may be repeated. Overrides --n-trials when provided")
parser.add_argument('--n-samples', '-n', type=int, default=10000, help="number of samples (default: 10000)")
parser.add_argument('--dim', '-d', type=int, default=1, help="dimensionality of variables (default: 1)")
parser.add_argument('--var-dim', action="append", default=[],
                    help="per-variable dimension override formatted as VAR=DIM; may be repeated")
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
parser.add_argument('--mask-fixed-one', action='append', default=[],
                    help="edge to constrain to 1, formatted as SRC->DST; may be repeated")
parser.add_argument('--mask-coupled-edge', action='append', default=[],
                    help="unordered edge to orient with one coupled logit, formatted as SRC--DST; may be repeated")
parser.add_argument('--mask-equiv-class-file',
                    help="equivalence-class graph file; directed edges are fixed, undirected edges are coupled, non-edges are fixed zero")
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
parser.add_argument('--mask-non-collider', action='append', default=[],
                    help="triple X,Y,Z for which X->Y<-Z is penalized; may be repeated")
parser.add_argument('--mask-non-collider-lambda', type=float, default=0.1,
                    help="weight on non-collider regularization")
parser.add_argument('--mask-fit-loss-weight', type=float, default=1.0,
                    help="weight on data-fit loss during mask updates; set to 0 to make mask updates use only query and structure losses")
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


def _parse_fixed_edges(specs, label):
    edges = []
    for spec in specs:
        if "->" not in spec:
            raise ValueError("{} edge '{}' must have form SRC->DST".format(label, spec))
        src, dst = spec.split("->", 1)
        src = src.strip()
        dst = dst.strip()
        if not src or not dst:
            raise ValueError("{} edge '{}' must have non-empty SRC and DST".format(label, spec))
        edges.append((src, dst))
    return edges


def _parse_fixed_zero_edges(specs):
    return _parse_fixed_edges(specs, "fixed-zero")


def _parse_fixed_one_edges(specs):
    return _parse_fixed_edges(specs, "fixed-one")


def _parse_coupled_edges(specs):
    edges = []
    for spec in specs:
        if "--" not in spec:
            raise ValueError("coupled edge '{}' must have form SRC--DST".format(spec))
        src, dst = spec.split("--", 1)
        src = src.strip()
        dst = dst.strip()
        if not src or not dst:
            raise ValueError("coupled edge '{}' must have non-empty SRC and DST".format(spec))
        if src == dst:
            raise ValueError("coupled edge '{}' cannot be a self-edge".format(spec))
        edges.append(tuple(sorted((src, dst))))
    return edges


def _merge_unique_edges(*edge_lists):
    seen = set()
    merged = []
    for edge_list in edge_lists:
        for edge in edge_list:
            edge = tuple(edge)
            if edge in seen:
                continue
            seen.add(edge)
            merged.append(edge)
    return merged


def _equiv_skeleton_neighbors(vertices, directed_edges, undirected_edges):
    neighbors = {v: set() for v in vertices}
    for src, dst in directed_edges:
        neighbors[src].add(dst)
        neighbors[dst].add(src)
    for src, dst in undirected_edges:
        neighbors[src].add(dst)
        neighbors[dst].add(src)
    return neighbors


def _equiv_unshielded_colliders(vertices, directed_edges, skeleton_neighbors):
    parents = {v: set() for v in vertices}
    for src, dst in directed_edges:
        parents[dst].add(src)

    colliders = set()
    for mid in vertices:
        for left, right in itertools.combinations(sorted(parents[mid]), 2):
            if right not in skeleton_neighbors[left]:
                colliders.add((left, mid, right))
    return colliders


def _equiv_non_collider_triples(vertices, directed_edges, undirected_edges):
    skeleton_neighbors = _equiv_skeleton_neighbors(vertices, directed_edges, undirected_edges)
    target_colliders = _equiv_unshielded_colliders(vertices, directed_edges, skeleton_neighbors)
    non_colliders = []
    for mid in vertices:
        for left, right in itertools.combinations(sorted(skeleton_neighbors[mid]), 2):
            if right in skeleton_neighbors[left]:
                continue
            triple = (left, mid, right)
            if triple not in target_colliders:
                non_colliders.append(triple)
    return non_colliders


def _build_equiv_mask_constraints(equiv_class_file, graph_name):
    spec = read_equivalence_class(equiv_class_file)
    cg = CausalGraph.read("dat/cg/{}.cg".format(graph_name))
    if set(spec.vertices) != set(cg.v):
        raise ValueError(
            "equivalence class vertices {} do not match graph {} vertices {}".format(
                sorted(spec.vertices), graph_name, sorted(cg.v)))

    directed_edges = [tuple(edge) for edge in spec.directed_edges]
    coupled_edges = [tuple(edge) for edge in spec.undirected_edges]
    non_collider_triples = _equiv_non_collider_triples(
        spec.vertices, directed_edges, coupled_edges)
    skeleton = {tuple(sorted(edge)) for edge in directed_edges}
    skeleton.update(tuple(edge) for edge in coupled_edges)

    fixed_zero_edges = []
    for src, dst in directed_edges:
        fixed_zero_edges.append((dst, src))
    for src, dst in itertools.combinations(spec.vertices, 2):
        pair = tuple(sorted((src, dst)))
        if pair not in skeleton:
            fixed_zero_edges.append((src, dst))
            fixed_zero_edges.append((dst, src))

    return fixed_zero_edges, directed_edges, coupled_edges, non_collider_triples


def _parse_non_collider_triples(specs):
    triples = []
    for spec in specs:
        parts = [part.strip() for part in spec.split(",")]
        if len(parts) != 3 or any(not part for part in parts):
            raise ValueError(
                "non-collider triple '{}' must have form X,Y,Z".format(spec))
        triples.append(tuple(parts))
    return triples


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


def _query_display_name(eval_query):
    if isinstance(eval_query, CTF):
        return eval_query.name
    if eval_query:
        return eval_query[0][0].name
    return str(eval_query)


def _graph_has_vars(graph_name, required_vars):
    cg = CausalGraph.read("dat/cg/{}.cg".format(graph_name))
    return set(required_vars).issubset(set(cg.v))


def _coerce_generated_hyperparams(raw_hyperparams):
    hyperparams = dict()
    for key in ("regions", "gen-bs"):
        if key in raw_hyperparams:
            hyperparams[key] = int(raw_hyperparams[key])
    if "c2-scale" in raw_hyperparams:
        hyperparams["c2-scale"] = float(raw_hyperparams["c2-scale"])
    for key in ("v-sizes", "v_sizes"):
        if key in raw_hyperparams:
            hyperparams["v-sizes"] = _parse_v_size_payload(raw_hyperparams[key])
            break
    return hyperparams


def _dimension_label(dim, v_sizes=None):
    if not v_sizes:
        return str(dim)
    parts = ["{}{}".format(var, v_sizes[var]) for var in sorted(v_sizes)]
    return "{}-v{}".format(dim, "_".join(parts))


def _generated_dataset_dir(root_dir, graph, n, dim, trial_index, v_sizes=None):
    return Path(root_dir) / "graph={}-n_samples={}-dim={}-trial_index={}".format(
        graph, n, _dimension_label(dim, v_sizes), trial_index)


def _load_generated_data_bundle(source_path, graph, n, dim, hyperparams):
    source_path = Path(source_path)
    source_dir = source_path if source_path.is_dir() else source_path.parent
    metadata_path = source_dir / "data_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError("generated data metadata not found at {}".format(metadata_path))

    with open(metadata_path) as file:
        metadata = json.load(file)
    seed = int(metadata["candidate_seed"])
    metadata_v_sizes = metadata.get("v_sizes")

    dat_path = source_path if source_path.is_file() else source_dir / "{}_dat.th".format(graph)
    if not dat_path.is_file():
        dat_path = source_dir / "dat.th"
    if not dat_path.is_file():
        raise FileNotFoundError("generated dataset not found at {}".format(dat_path))

    stored_path = source_dir / "{}_stored_metrics.th".format(graph)
    if not stored_path.is_file():
        stored_path = source_dir / "stored_metrics.th"

    generated_hp = dict(hyperparams)
    generated_hp_path = source_dir / "hyperparams.json"
    if generated_hp_path.is_file():
        with open(generated_hp_path) as file:
            generated_hp.update(_coerce_generated_hyperparams(json.load(file)))
    if metadata_v_sizes is not None:
        generated_hp["v-sizes"] = metadata_v_sizes

    cg = CausalGraph.read("dat/cg/{}.cg".format(graph))
    dat_m = _build_dat_model(CTM, cg, dim, generated_hp, seed)
    dat_sets = T.load(dat_path)
    if stored_path.is_file():
        stored_metrics = T.load(stored_path)
    else:
        stored_metrics = dict()
    return DataBundle(cg=cg, dat_m=dat_m, dat_sets=dat_sets, stored_metrics=stored_metrics)


def main():
    args = parser.parse_args()

    gen_choice = args.gen.lower()
    graph_choice = args.graph.lower()
    bound_mode = args.bound_query
    query_track = args.query_track.lower() if args.query_track is not None and not bound_mode else None
    bound_query_track = args.bound_query_track.lower() if args.bound_query_track is not None else None
    if bound_mode and graph_choice == "barley":
        if args.bound_treatment == "X":
            args.bound_treatment = "sort"
        if args.bound_outcome == "Y":
            args.bound_outcome = "protein"

    assert gen_choice in valid_generators
    assert graph_choice in valid_graphs or graph_choice in graph_sets
    assert bound_mode or query_track is None or query_track in valid_queries
    assert bound_query_track is None or bound_query_track in valid_queries
    assert not (args.learn_mask and args.fixed_mask)
    assert not args.alt_opt or not args.fixed_mask
    assert args.bound_outcome_value in {0, 1}
    assert args.theta_steps_per_mask >= 0
    assert args.mask_steps_per_theta >= 0
    assert (args.theta_steps_per_mask + args.mask_steps_per_theta) > 0
    assert args.theta_only_extra_epochs >= 0
    assert args.theta_only_extra_lr is None or args.theta_only_extra_lr > 0
    assert args.early_stop_patience > 0
    assert args.early_stop_min_delta >= 0
    assert args.selection_query_lambda is None or args.selection_query_lambda >= 0
    assert not (args.reuse_data_from and args.reuse_data_root)
    assert args.alm_rho_init > 0
    assert args.alm_rho_mult >= 1.0
    assert args.alm_rho_max >= args.alm_rho_init
    assert args.alm_update_every > 0
    assert 0 < args.alm_improve_ratio <= 1.0
    assert not args.dag_alm or args.cycle_penalty == "notears"
    assert args.mask_non_collider_lambda >= 0
    assert args.mask_fit_loss_weight >= 0

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

    manual_fixed_zero_edges = _parse_fixed_zero_edges(args.mask_fixed_zero)
    manual_fixed_one_edges = _parse_fixed_one_edges(args.mask_fixed_one)
    manual_coupled_edges = _parse_coupled_edges(args.mask_coupled_edge)
    manual_non_collider_triples = _parse_non_collider_triples(args.mask_non_collider)

    hyperparams = {
        'lr': args.lr,
        'theta-lr': args.theta_lr if args.theta_lr is not None else args.lr,
        'mask-lr': args.mask_lr if args.mask_lr is not None else args.lr,
        'data-bs': args.data_bs,
        'ncm-bs': args.ncm_bs,
        'num-workers': args.num_workers,
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
        'bound-query-track': bound_query_track,
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
        'mask-fixed-zero-edges': manual_fixed_zero_edges,
        'mask-fixed-one-edges': manual_fixed_one_edges,
        'mask-coupled-edges': manual_coupled_edges,
        'mask-equiv-class-file': args.mask_equiv_class_file,
        'cycle-lambda': args.cycle_lambda,
        'cycle-penalty': args.cycle_penalty,
        'dagma-s': args.dagma_s,
        'mask-l1-lambda': args.mask_l1_lambda,
        'mask-binary-lambda': args.mask_binary_lambda,
        'mask-non-collider-triples': manual_non_collider_triples,
        'mask-non-collider-lambda': args.mask_non_collider_lambda,
        'mask-fit-loss-weight': args.mask_fit_loss_weight,
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
        'early-stop-patience': args.early_stop_patience,
        'early-stop-min-delta': args.early_stop_min_delta,
        'max-lambda': args.max_lambda,
        'min-lambda': args.min_lambda,
        'selection-query-lambda': (
            args.selection_query_lambda
            if args.selection_query_lambda is not None
            else args.min_lambda
        ),
        'mc-sample-size': args.mc_sample_size,
        'query-update-target': args.query_update_target,
    }
    if args.theta_only_extra_epochs > 0:
        hyperparams['theta-only-extra-epochs'] = args.theta_only_extra_epochs
        hyperparams['theta-only-final-query-reg'] = args.theta_only_final_query_reg
        if args.theta_only_extra_lr is not None:
            hyperparams['theta-only-extra-lr'] = args.theta_only_extra_lr

    if graph_choice in graph_sets:
        if bound_mode:
            required_vars = (
                {"X", "Y"} if bound_query_track is not None
                else {args.bound_treatment, args.bound_outcome}
            )
            graph_set = {
                graph for graph in graph_sets[graph_choice]
                if _graph_has_vars(graph, required_vars)
            }
        else:
            graph_set = {graph for graph in graph_sets[graph_choice] if is_q_id_in_G(graph, query_track)}
    else:
        if bound_mode:
            required_vars = (
                {"X", "Y"} if bound_query_track is not None
                else {args.bound_treatment, args.bound_outcome}
            )
            if not _graph_has_vars(graph_choice, required_vars):
                raise ValueError("graph {} does not contain {}".format(
                    graph_choice, sorted(required_vars)))
        graph_set = {graph_choice}

    if bound_mode and not graph_set:
        if bound_query_track is not None:
            raise ValueError("no selected graphs contain X and Y")
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
        graph_cg = CausalGraph.read("dat/cg/{}.cg".format(graph))
        v_size_overrides = parse_var_dim_overrides(args.var_dim, graph_cg, strict=False)
        if v_size_overrides:
            hyperparams['v-sizes'] = _build_v_sizes(
                graph_cg, args.dim, {'v-sizes': v_size_overrides})
        else:
            hyperparams.pop('v-sizes', None)

        graph_fixed_zero_edges = list(manual_fixed_zero_edges)
        graph_fixed_one_edges = list(manual_fixed_one_edges)
        graph_coupled_edges = list(manual_coupled_edges)
        graph_non_collider_triples = list(manual_non_collider_triples)
        if args.mask_equiv_class_file:
            (
                equiv_fixed_zero,
                equiv_fixed_one,
                equiv_coupled,
                equiv_non_colliders,
            ) = _build_equiv_mask_constraints(
                args.mask_equiv_class_file, graph)
            graph_fixed_zero_edges = _merge_unique_edges(graph_fixed_zero_edges, equiv_fixed_zero)
            graph_fixed_one_edges = _merge_unique_edges(graph_fixed_one_edges, equiv_fixed_one)
            graph_coupled_edges = _merge_unique_edges(graph_coupled_edges, equiv_coupled)
            graph_non_collider_triples = _merge_unique_edges(
                graph_non_collider_triples, equiv_non_colliders)

        hyperparams['mask-fixed-zero-edges'] = graph_fixed_zero_edges
        hyperparams['mask-fixed-one-edges'] = graph_fixed_one_edges
        hyperparams['mask-coupled-edges'] = graph_coupled_edges
        hyperparams['mask-non-collider-triples'] = graph_non_collider_triples

        do_var_list = get_experimental_variables(graph)
        if bound_mode:
            if bound_query_track is not None:
                eval_query, opt_queries = get_query(graph, bound_query_track)
                query_jobs = [(eval_query, opt_queries, None)]
            else:
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
            hyperparams["query-track"] = _query_display_name(eval_query) if bound_mode else query_track
            if bound_mode:
                hyperparams["max-query-1"] = opt_queries[0]
                hyperparams["max-query-2"] = opt_queries[1]
                if query_bounds is not None:
                    hyperparams["query-bound-spec"] = query_bounds
                else:
                    hyperparams.pop("query-bound-spec", None)
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
                    data_bundle = None
                    if args.reuse_data_from or args.reuse_data_root:
                        source_path = args.reuse_data_from
                        if args.reuse_data_root:
                            source_path = _generated_dataset_dir(
                                args.reuse_data_root, graph, n, d, i, hyperparams.get("v-sizes"))
                        data_bundle = _load_generated_data_bundle(
                            source_path, graph, n, d, hyperparams)
                    try:
                        if not runner.run(
                                args.name,
                                cg_file,
                                n,
                                d,
                                i,
                                hyperparams=hyperparams,
                                gpu=gpu_used,
                                data_bundle=data_bundle,
                                verbose=args.verbose):
                            continue
                    except Exception as e:
                        print(e)
                        print("[failed]", i, args.name)
                        raise


if __name__ == "__main__":
    main()
