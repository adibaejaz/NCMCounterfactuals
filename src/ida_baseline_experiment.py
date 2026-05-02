import argparse
import glob
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np
import torch as T

from src.baselines.empirical_adjustment import empirical_adjustment, empirical_marginal
from src.baselines.ida import (
    ida_adjustment_sets_from_file,
    outcome_is_undirected_neighbor,
)
from src.ds.equivalence_class import read_equivalence_class
from src.ds import CTF, CTFTerm
from src.ds.causal_graph import CausalGraph
from src.metric import evaluation
from src.metric.queries import get_experimental_variables
from src.run.data_setup import (
    DataBundle,
    _build_dat_model,
    _build_v_sizes,
    _parse_v_size_payload,
    build_data_bundle,
    parse_var_dim_overrides,
)
from src.scm.ctm import CTM
from src.scm.model_classes import RoundModel, XORModel


VALID_GENERATORS = {
    "ctm": CTM,
    "xor": XORModel,
    "round": RoundModel,
}

VALID_GRAPHS = {
    "backdoor", "bow", "frontdoor", "napkin", "simple", "chain", "bdm", "med", "expl", "double_bow", "iv", "bad_fd",
    "extended_bow", "bad_m", "m", "square", "four_clique", "barley",
    "zid_a", "zid_b", "zid_c",
    "gid_a", "gid_b", "gid_c", "gid_d",
    "med_c1", "med_c2",
    "expl_xm", "expl_xm_dox", "expl_xy", "expl_dox", "expl_xy_dox", "expl_my", "expl_my_dox",
    "sachs"
}


def _jsonable(value):
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if T.is_tensor(value):
        return _jsonable(value.detach().cpu().tolist())
    return str(value)


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        json.dump(_jsonable(payload), file, indent=2, sort_keys=True)


def _dimension_label(dim, v_sizes=None):
    if not v_sizes:
        return str(dim)
    parts = ["{}{}".format(var, v_sizes[var]) for var in sorted(v_sizes)]
    return "{}-v{}".format(dim, "_".join(parts))


def _generated_dataset_dir(root_dir, graph, n, dim, trial_index, v_sizes=None):
    return Path(root_dir) / "graph={}-n_samples={}-dim={}-trial_index={}".format(
        graph, n, _dimension_label(dim, v_sizes), trial_index)


def _find_generated_dataset_dir(root_dir, graph, n, dim, trial_index, v_sizes=None):
    exact = _generated_dataset_dir(root_dir, graph, n, dim, trial_index, v_sizes)
    if exact.is_dir():
        return exact

    pattern = "graph={}-n_samples={}-dim={}*-trial_index={}".format(
        graph, n, dim, trial_index)
    matches = sorted(
        path for path in Path(root_dir).glob(pattern)
        if path.is_dir() and (path / "data_metadata.json").is_file()
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            "multiple generated datasets match {} under {}: {}".format(
                pattern, root_dir, [str(path) for path in matches]))
    return None


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


def _load_generated_data_bundle(source_path, graph, dim, hyperparams):
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
    stored_metrics = T.load(stored_path) if stored_path.is_file() else dict()
    return DataBundle(cg=cg, dat_m=dat_m, dat_sets=dat_sets, stored_metrics=stored_metrics), seed


def _load_saved_data_bundle(source_path, dat_model, cg_file, dim, hyperparams, seed):
    source_path = Path(source_path)
    source_dir = source_path if source_path.is_dir() else source_path.parent
    dat_path = source_path if source_path.is_file() else source_dir / "dat.th"
    if not dat_path.is_file():
        raise FileNotFoundError("saved dataset not found at {}".format(dat_path))

    saved_hp_path = source_dir / "hyperparams.json"
    saved_hp = dict(hyperparams)
    if saved_hp_path.is_file():
        with open(saved_hp_path) as file:
            saved_hp.update(_coerce_generated_hyperparams(json.load(file)))

    cg = CausalGraph.read(cg_file)
    dat_m = _build_dat_model(dat_model, cg, dim, saved_hp, seed)
    dat_sets = T.load(dat_path)
    return DataBundle(cg=cg, dat_m=dat_m, dat_sets=dat_sets, stored_metrics=dict()), seed


def _find_run_reuse_source(root_dir, graph, n, dim, trial_index, train_seed_offset):
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
    if len(matches) != 1:
        raise ValueError("expected one run reuse source for {}, found {}".format(pattern, matches))
    return matches[0]


def _resolve_data_bundle(args, graph, n, dim, trial_index, hyperparams, dat_model, cg_file, data_seed):
    if args.reuse_data_from:
        source = Path(args.reuse_data_from)
        if source.is_dir() and (source / "data_metadata.json").is_file():
            return _load_generated_data_bundle(source, graph, dim, hyperparams)
        return _load_saved_data_bundle(source, dat_model, cg_file, dim, hyperparams, data_seed)

    if args.reuse_data_root:
        generated_source = _find_generated_dataset_dir(
            args.reuse_data_root, graph, n, dim, trial_index, hyperparams.get("v-sizes"))
        if generated_source is not None:
            return _load_generated_data_bundle(generated_source, graph, dim, hyperparams)
        try:
            run_source = _find_run_reuse_source(
                args.reuse_data_root, graph, n, dim, trial_index, args.train_seed_offset)
        except ValueError as exc:
            raise ValueError(
                "no generated dataset found under {} for graph={}, n={}, dim={}, trial_index={}; "
                "also failed saved-run lookup: {}".format(
                    args.reuse_data_root, graph, n, dim, trial_index, exc)) from exc
        return _load_saved_data_bundle(run_source, dat_model, cg_file, dim, hyperparams, data_seed)

    return build_data_bundle(dat_model, cg_file, n, dim, hyperparams, data_seed), data_seed


def _stable_seed(payload):
    import hashlib
    return int(hashlib.sha512(payload.encode()).hexdigest(), 16) & 0xffffffff


def _data_key(dat_model, graph, n, dim, trial_index):
    return "gen={}-graph={}-n_samples={}-dim={}-trial_index={}".format(
        dat_model.__name__, graph, n, dim, trial_index)


def _true_query_value(data_bundle, outcome_var, outcome_value, treatment_var, treatment_value):
    query = CTF(
        {CTFTerm({outcome_var}, {treatment_var: treatment_value}, {outcome_var: outcome_value})},
        set(),
        name="P({}={} | do({}={}))".format(outcome_var, outcome_value, treatment_var, treatment_value),
    )
    return float(evaluation.eval_query(data_bundle.dat_m, query, n=1000000))


def build_parser():
    parser = argparse.ArgumentParser(description="IDA-style empirical adjustment baseline")
    parser.add_argument("name", help="output experiment directory")
    parser.add_argument("--equiv-class-file", "--mask-equiv-class-file", dest="equiv_class_file",
                        required=True, help="equivalence-class graph file with directed and undirected edges")
    parser.add_argument("--gen", default="ctm", choices=sorted(VALID_GENERATORS.keys()))
    parser.add_argument("--graph", default="chain", choices=sorted(VALID_GRAPHS))
    parser.add_argument("--n-samples", "-n", type=int, default=10000)
    parser.add_argument("--n-trials", "-t", type=int, default=1)
    parser.add_argument("--trial-index", action="append", type=int, default=[])
    parser.add_argument("--dim", "-d", type=int, default=1)
    parser.add_argument("--var-dim", action="append", default=[])
    parser.add_argument("--regions", type=int, default=20)
    parser.add_argument("--scale-regions", action="store_true")
    parser.add_argument("--gen-bs", type=int, default=10000)
    parser.add_argument("--bound-treatment", default="X")
    parser.add_argument("--bound-outcome", default="Y")
    parser.add_argument("--bound-treatment-value", action="append", type=int, default=[])
    parser.add_argument("--bound-outcome-value", type=int, default=1)
    parser.add_argument("--reuse-data-from")
    parser.add_argument("--reuse-data-root")
    parser.add_argument("--train-seed-offset", type=int, default=0)
    parser.add_argument("--no-true-query", action="store_true",
                        help="do not evaluate the true CTM query diagnostic")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.reuse_data_from and args.reuse_data_root:
        raise ValueError("use only one of --reuse-data-from or --reuse-data-root")
    if args.bound_outcome_value not in {0, 1}:
        raise ValueError("--bound-outcome-value must be 0 or 1")

    dat_model = VALID_GENERATORS[args.gen]
    cg_file = "dat/cg/{}.cg".format(args.graph)
    graph_cg = CausalGraph.read(cg_file)
    required = {args.bound_treatment, args.bound_outcome}
    if not required.issubset(set(graph_cg.v)):
        raise ValueError("graph {} does not contain {}".format(args.graph, sorted(required)))

    v_size_overrides = parse_var_dim_overrides(args.var_dim, graph_cg, strict=False)
    hyperparams = {
        "regions": args.regions,
        "c2-scale": 2.0 if args.scale_regions else 1.0,
        "gen-bs": args.gen_bs,
        "train-seed-offset": args.train_seed_offset,
        "do-var-list": get_experimental_variables(args.graph),
    }
    if v_size_overrides:
        hyperparams["v-sizes"] = _build_v_sizes(graph_cg, args.dim, {"v-sizes": v_size_overrides})

    equiv_spec = read_equivalence_class(args.equiv_class_file)
    adjustment_sets = ida_adjustment_sets_from_file(
        args.equiv_class_file, args.bound_treatment, outcome=args.bound_outcome)
    include_marginal_outcome = outcome_is_undirected_neighbor(
        equiv_spec, args.bound_treatment, args.bound_outcome)
    treatment_values = args.bound_treatment_value if args.bound_treatment_value else [0, 1]
    trial_indices = args.trial_index if args.trial_index else range(args.n_trials)

    for n, dim, trial_index, treatment_value in itertools.product(
            [int(args.n_samples)], [int(args.dim)], trial_indices, treatment_values):
        key = _data_key(dat_model, args.graph, n, dim, trial_index)
        data_seed = _stable_seed("data|" + key)
        data_bundle, recovered_seed = _resolve_data_bundle(
            args, args.graph, n, dim, trial_index, hyperparams, dat_model, cg_file, data_seed)
        if not data_bundle.dat_sets:
            raise ValueError("data bundle contains no datasets")
        obs = data_bundle.dat_sets[0]

        candidates = []
        if include_marginal_outcome:
            candidates.append(empirical_marginal(
                obs,
                args.bound_outcome,
                args.bound_outcome_value,
            ))
        for adjustment_vars in adjustment_sets:
            result = empirical_adjustment(
                obs,
                args.bound_outcome,
                args.bound_outcome_value,
                args.bound_treatment,
                treatment_value,
                adjustment_vars,
            )
            candidates.append(result)

        finite = [item["estimate"] for item in candidates if not np.isnan(item["estimate"])]
        lower = min(finite) if finite else float("nan")
        upper = max(finite) if finite else float("nan")
        true_value = None if args.no_true_query else _true_query_value(
            data_bundle,
            args.bound_outcome,
            args.bound_outcome_value,
            args.bound_treatment,
            treatment_value,
        )

        for item in candidates:
            if true_value is not None and not np.isnan(item["estimate"]):
                item["abs_error"] = abs(item["estimate"] - true_value)

        result = {
            "graph": args.graph,
            "equiv_class_file": args.equiv_class_file,
            "n_samples": n,
            "dim": dim,
            "trial_index": trial_index,
            "data_seed": recovered_seed,
            "query": "P({}={} | do({}={}))".format(
                args.bound_outcome, args.bound_outcome_value, args.bound_treatment, treatment_value),
            "outcome_var": args.bound_outcome,
            "outcome_value": args.bound_outcome_value,
            "treatment_var": args.bound_treatment,
            "treatment_value": treatment_value,
            "adjustment_sets": [list(item) for item in adjustment_sets],
            "include_marginal_outcome": include_marginal_outcome,
            "candidates": candidates,
            "lower": lower,
            "upper": upper,
            "true_query": true_value,
        }

        out_dir = os.path.join(
            args.name,
            "{}-ida_treatment={}".format(key, treatment_value),
        )
        _write_json(os.path.join(out_dir, "hyperparams.json"), hyperparams)
        _write_json(os.path.join(out_dir, "ida_adjustment_sets.json"), {
            "treatment_var": args.bound_treatment,
            "outcome_var": args.bound_outcome,
            "adjustment_sets": [list(item) for item in adjustment_sets],
            "include_marginal_outcome": include_marginal_outcome,
        })
        _write_json(os.path.join(out_dir, "results.json"), result)
        T.save(data_bundle.dat_sets, os.path.join(out_dir, "dat.th"))

        if args.verbose:
            print("{}: [{}, {}] true={}".format(result["query"], lower, upper, true_value))


if __name__ == "__main__":
    main()
