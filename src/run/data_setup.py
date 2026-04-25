import random
import os
import json
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
        v_sizes = {k: 1 if k in {'X', 'Y', 'M', 'W'} else dim for k in cg}
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
