import random
import ast
import os
import json
import glob
from dataclasses import dataclass

import torch as T

from src.metric import evaluation
from src.ds.causal_graph import CausalGraph
from src.scm.ctm import CTM
from src.scm.scm import expand_do


@dataclass
class DataBundle:
    cg: CausalGraph
    dat_m: object
    dat_sets: list
    stored_metrics: dict


DEFAULT_SCALAR_VARS = {'X', 'Y', 'M', 'W'}


def _parse_v_size_payload(value):
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = ast.literal_eval(value)
    if not isinstance(value, dict):
        raise ValueError("v-sizes must be a dictionary, got {}".format(type(value).__name__))
    return {str(k): int(v) for k, v in value.items()}


def _build_v_sizes(cg, dim, hyperparams=None):
    if hyperparams is None:
        hyperparams = {}
    v_sizes = {k: 1 if k in DEFAULT_SCALAR_VARS else int(dim) for k in cg}
    v_sizes.update(_parse_v_size_payload(hyperparams.get('v-sizes')))
    invalid = sorted(set(v_sizes) - set(cg))
    if invalid:
        raise ValueError("v-sizes contain variables not in graph: {}".format(invalid))
    for var, size in v_sizes.items():
        if size < 1:
            raise ValueError("dimension for {} must be positive, got {}".format(var, size))
    return v_sizes


def parse_var_dim_overrides(var_dim_args, cg, strict=True):
    graph_vars = set(cg)
    overrides = {}
    for item in var_dim_args or []:
        if "=" not in item:
            raise ValueError("--var-dim must be formatted as VAR=DIM, got {}".format(item))
        var, raw_dim = item.split("=", 1)
        var = var.strip()
        if var not in graph_vars:
            if strict:
                raise ValueError("--var-dim variable {} is not in graph".format(var))
            continue
        try:
            value = int(raw_dim)
        except ValueError as exc:
            raise ValueError("--var-dim dimension must be an integer: {}".format(item)) from exc
        if value < 1:
            raise ValueError("--var-dim dimension must be positive: {}".format(item))
        overrides[var] = value
    return overrides


def _build_stored_metrics(dat_m, dat_sets, do_var_list):
    stored_metrics = dict()
    for i, dat_do_set in enumerate(do_var_list):
        name = evaluation.serialize_do(dat_do_set)
        stored_metrics["true_{}".format(name)] = evaluation.probability_table(
            dat_m, n=1000000, do={k: expand_do(v, n=1000000) for (k, v) in dat_do_set.items()})
        stored_metrics["dat_{}".format(name)] = evaluation.probability_table(
            dat_m, n=1000000, do={k: expand_do(v, n=1000000) for (k, v) in dat_do_set.items()},
            dat=dat_sets[i])
    return stored_metrics


def _build_dat_model(dat_model, cg, dim, hyperparams, seed, runtime=False):
    if dat_model is CTM:
        v_sizes = _build_v_sizes(cg, dim, hyperparams)
        return dat_model(
            cg,
            v_size=v_sizes,
            regions=hyperparams.get('regions', 20),
            c2_scale=hyperparams.get('c2-scale', 1.0),
            batch_size=hyperparams.get('gen-bs', 10000),
            seed=seed,
        )
    if runtime:
        p = random.random()
        return dat_model(cg, p=p, dim=dim, seed=seed)
    return dat_model(cg, dim=dim, seed=seed)


def build_data_bundle(dat_model, cg_file, n, dim, hyperparams, seed, runtime=False):
    cg = CausalGraph.read(cg_file)
    dat_m = _build_dat_model(dat_model, cg, dim, hyperparams, seed, runtime=runtime)

    dat_sets = []
    for dat_do_set in hyperparams["do-var-list"]:
        expand_do_set = {k: expand_do(v, n=n) for (k, v) in dat_do_set.items()}
        dat_sets.append(dat_m(n=n, do=expand_do_set))

    stored_metrics = _build_stored_metrics(dat_m, dat_sets, hyperparams["do-var-list"])

    return DataBundle(cg=cg, dat_m=dat_m, dat_sets=dat_sets, stored_metrics=stored_metrics)


def _dat_sets_equal(dat_sets_a, dat_sets_b):
    if len(dat_sets_a) != len(dat_sets_b):
        return False
    for dat_a, dat_b in zip(dat_sets_a, dat_sets_b):
        if set(dat_a.keys()) != set(dat_b.keys()):
            return False
        for key in dat_a:
            if not T.equal(dat_a[key], dat_b[key]):
                return False
    return True


def _coerce_numeric_hyperparams(raw_hyperparams):
    coerced = dict()
    for key in ("regions", "gen-bs"):
        value = raw_hyperparams.get(key)
        if value is not None:
            coerced[key] = int(value)
    value = raw_hyperparams.get("c2-scale")
    if value is not None:
        coerced["c2-scale"] = float(value)
    for key in ("v-sizes", "v_sizes"):
        if key in raw_hyperparams:
            coerced["v-sizes"] = _parse_v_size_payload(raw_hyperparams[key])
            break
    return coerced


def build_reused_data_bundle(dat_model, cg_file, n, dim, hyperparams, base_seed, source_path, max_seed_steps=1000):
    source_path = os.path.abspath(source_path)
    source_dir = source_path if os.path.isdir(source_path) else os.path.dirname(source_path)
    dat_path = source_path if os.path.isfile(source_path) else os.path.join(source_dir, "dat.th")
    if not os.path.isfile(dat_path):
        raise FileNotFoundError("reused data file not found at {}".format(dat_path))

    source_hp_path = os.path.join(source_dir, "hyperparams.json")
    search_hyperparams = dict(hyperparams)
    if os.path.isfile(source_hp_path):
        with open(source_hp_path) as file:
            source_hyperparams = json.load(file)
        search_hyperparams.update(_coerce_numeric_hyperparams(source_hyperparams))

    saved_dat_sets = T.load(dat_path)
    cg = CausalGraph.read(cg_file)

    for seed_step in range(max_seed_steps + 1):
        seed = (base_seed + seed_step) & 0xffffffff
        dat_m = _build_dat_model(dat_model, cg, dim, search_hyperparams, seed)
        generated_dat_sets = []
        for dat_do_set in hyperparams["do-var-list"]:
            expand_do_set = {k: expand_do(v, n=n) for (k, v) in dat_do_set.items()}
            generated_dat_sets.append(dat_m(n=n, do=expand_do_set))
        if _dat_sets_equal(generated_dat_sets, saved_dat_sets):
            stored_metrics = _build_stored_metrics(dat_m, saved_dat_sets, hyperparams["do-var-list"])
            return DataBundle(cg=cg, dat_m=dat_m, dat_sets=saved_dat_sets, stored_metrics=stored_metrics), seed

    raise ValueError(
        "failed to recover saved data seed from {} within {} increments of base seed {}".format(
            dat_path, max_seed_steps, base_seed))


def _dimension_label(dim, v_sizes=None):
    if not v_sizes:
        return str(dim)
    parts = ["{}{}".format(var, v_sizes[var]) for var in sorted(v_sizes)]
    return "{}-v{}".format(dim, "_".join(parts))


def generated_dataset_dir(root_dir, graph, n, dim, trial_index, v_sizes=None):
    return os.path.join(
        root_dir,
        "graph={}-n_samples={}-dim={}-trial_index={}".format(
            graph, n, _dimension_label(dim, v_sizes), trial_index))


def find_generated_dataset_dir(root_dir, graph, n, dim, trial_index, v_sizes=None):
    exact = generated_dataset_dir(root_dir, graph, n, dim, trial_index, v_sizes)
    if os.path.isdir(exact):
        return exact

    pattern = os.path.join(
        root_dir,
        "graph={}-n_samples={}-dim={}*-trial_index={}".format(graph, n, dim, trial_index))
    matches = sorted(
        path for path in glob.glob(pattern)
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "data_metadata.json"))
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError("multiple generated datasets match {}: {}".format(pattern, matches))
    return None


def load_generated_data_bundle(source_path, graph, dim, hyperparams):
    source_path = os.path.abspath(source_path)
    source_dir = source_path if os.path.isdir(source_path) else os.path.dirname(source_path)
    metadata_path = os.path.join(source_dir, "data_metadata.json")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError("generated data metadata not found at {}".format(metadata_path))

    with open(metadata_path) as file:
        metadata = json.load(file)
    seed = int(metadata["candidate_seed"])
    metadata_v_sizes = metadata.get("v_sizes")

    dat_path = source_path if os.path.isfile(source_path) else os.path.join(source_dir, "{}_dat.th".format(graph))
    if not os.path.isfile(dat_path):
        dat_path = os.path.join(source_dir, "dat.th")
    if not os.path.isfile(dat_path):
        raise FileNotFoundError("generated dataset not found at {}".format(dat_path))

    stored_path = os.path.join(source_dir, "{}_stored_metrics.th".format(graph))
    if not os.path.isfile(stored_path):
        stored_path = os.path.join(source_dir, "stored_metrics.th")

    generated_hp = dict(hyperparams)
    source_hp_path = os.path.join(source_dir, "hyperparams.json")
    if os.path.isfile(source_hp_path):
        with open(source_hp_path) as file:
            generated_hp.update(_coerce_numeric_hyperparams(json.load(file)))
    if metadata_v_sizes is not None:
        generated_hp["v-sizes"] = metadata_v_sizes

    cg = CausalGraph.read("dat/cg/{}.cg".format(graph))
    dat_m = _build_dat_model(CTM, cg, dim, generated_hp, seed)
    dat_sets = T.load(dat_path)
    stored_metrics = T.load(stored_path) if os.path.isfile(stored_path) else dict()
    return DataBundle(cg=cg, dat_m=dat_m, dat_sets=dat_sets, stored_metrics=stored_metrics), seed
