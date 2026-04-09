import random
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

    stored_metrics = dict()
    for i, dat_do_set in enumerate(hyperparams["do-var-list"]):
        name = evaluation.serialize_do(dat_do_set)
        stored_metrics["true_{}".format(name)] = evaluation.probability_table(
            dat_m, n=1000000, do={k: expand_do(v, n=1000000) for (k, v) in dat_do_set.items()})
        stored_metrics["dat_{}".format(name)] = evaluation.probability_table(
            dat_m, n=1000000, do={k: expand_do(v, n=1000000) for (k, v) in dat_do_set.items()},
            dat=dat_sets[i])

    return DataBundle(cg=cg, dat_m=dat_m, dat_sets=dat_sets, stored_metrics=stored_metrics)
