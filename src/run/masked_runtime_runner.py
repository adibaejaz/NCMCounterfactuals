"""Runtime-focused runner for single-model masked NCM experiments.

This is the masked analogue of ``RunTimeRunner``:
- generate or reuse one dataset
- build one masked pipeline/model for a specific ``pipeline_choice``
- compute initial metrics for bookkeeping
- train for a fixed number of epochs without early stopping
- record wall-clock training time

Its primary output is timing information, not final quality metrics.
"""

import os
import glob
import shutil
import hashlib
import json

import numpy as np
import torch as T
import pytorch_lightning as pl

from src.metric import evaluation
from src.ds.causal_graph import CausalGraph
from src.scm.ctm import CTM
from src.scm.model_classes import XORModel, RoundModel
from src.scm.scm import expand_do
from .base_runner import BaseRunner

import random
import time


class MaskedRunTimeRunner(BaseRunner):
    """Run a single-model masked experiment and track training runtime."""

    def __init__(self, pipeline, dat_model, ncm_model):
        super().__init__(pipeline, dat_model, ncm_model)

    def create_trainer(self, directory, max_epochs, pipeline_choice, gpu=None):
        checkpoint = pl.callbacks.ModelCheckpoint(dirpath=f'{directory}/{pipeline_choice}/checkpoints/',
                                                  monitor="train_loss")
        return pl.Trainer(
            callbacks=[checkpoint],
            max_epochs=max_epochs,
            accumulate_grad_batches=1,
            logger=pl.loggers.TensorBoardLogger(f'{directory}/{pipeline_choice}/logs/'),
            log_every_n_steps=1,
            terminate_on_nan=True,
            gpus=gpu
        ), checkpoint

    def _ncm_kwargs(self, pl_model):
        return dict(
            mask=pl_model.get_mask(),
            use_dag_updates=pl_model.use_dag_updates,
        )

    def run(self, exp_name, cg_file, n, dim, trial_index, hyperparams=None, gpu=None, data_bundle=None,
            lockinfo=os.environ.get('SLURM_JOB_ID', ''), verbose=False):
        if hyperparams is None:
            hyperparams = dict()
        key = self.get_key(cg_file, n, dim, trial_index)
        run_key = self.get_run_key(cg_file, n, dim, trial_index, hyperparams)
        d = 'out/%s/%s' % (exp_name, run_key)

        with self.lock(f'{d}/lock', lockinfo) as acquired_lock:
            if not acquired_lock:
                print('[locked]', d)
                return

            try:
                if os.path.isfile(f'{d}/{hyperparams["pipeline_choice"]}/best.th'):
                    print('[done]', f'{d}/{hyperparams["pipeline_choice"]}')
                    return

                print('[running]', d)

                seed = int(hashlib.sha512(key.encode()).hexdigest(), 16) & 0xffffffff
                if data_bundle is not None:
                    cg = data_bundle.cg
                    dat_m = data_bundle.dat_m
                    dat_sets = data_bundle.dat_sets

                positivity = data_bundle is not None
                while not positivity:
                    seed += 1
                    T.manual_seed(seed)
                    np.random.seed(seed)
                    print('Key:', key)
                    print('Seed:', seed)

                    print('Generating data')
                    cg = CausalGraph.read(cg_file)
                    v_sizes = {k: 1 if k in {'X', 'Y', 'M'} else dim for k in cg}
                    if self.dat_model is CTM:
                        dat_m = self.dat_model(cg, v_size=v_sizes, regions=hyperparams.get('regions', 20),
                                               c2_scale=hyperparams.get('c2-scale', 1.0),
                                               batch_size=hyperparams.get('gen-bs', 10000),
                                               seed=seed)
                    else:
                        p = random.random()
                        dat_m = self.dat_model(cg, p=p, dim=dim, seed=seed)
                    dat_sets = []
                    for dat_do_set in hyperparams["do-var-list"]:
                        var_dims = 0
                        for k in v_sizes:
                            if k not in dat_do_set:
                                var_dims += v_sizes[k]
                        expand_do_set = {k: expand_do(v, n=n) for (k, v) in dat_do_set.items()}
                        dat_set = dat_m(n=n, do=expand_do_set)
                        prob_table = evaluation.probability_table(dat=dat_set)
                        positivity = True
                        print(prob_table)
                        if len(prob_table) == 2 ** var_dims:
                            positivity = True
                            print(prob_table)

                        dat_sets.append(dat_m(n=n, do=expand_do_set))

                m = self.pipeline(dat_m, hyperparams["do-var-list"], dat_sets, cg, dim, hyperparams=hyperparams,
                                  ncm_model=self.ncm_model)
                print("Calculating metrics")
                if data_bundle is None:
                    stored_metrics = dict()
                    for i, dat_do_set in enumerate(hyperparams["do-var-list"]):
                        name = evaluation.serialize_do(dat_do_set)
                        stored_metrics["true_{}".format(name)] = evaluation.probability_table(
                            dat_m, n=1000000, do={k: expand_do(v, n=1000000) for (k, v) in dat_do_set.items()})
                        stored_metrics["dat_{}".format(name)] = evaluation.probability_table(
                            dat_m, n=1000000, do={k: expand_do(v, n=1000000) for (k, v) in dat_do_set.items()},
                            dat=dat_sets[i])
                else:
                    stored_metrics = dict(data_bundle.stored_metrics)
                start_metrics = evaluation.all_metrics(
                    m.generator, m.ncm, hyperparams["do-var-list"], dat_sets,
                    n=1000000, stored=stored_metrics,
                    query_track=hyperparams['eval-query'],
                    ncm_kwargs=self._ncm_kwargs(m))
                if hyperparams['query-track'] is not None:
                    true_q = 'true_{}'.format(evaluation.serialize_query(hyperparams['eval-query']))
                    stored_metrics[true_q] = start_metrics[true_q]
                m.update_metrics(stored_metrics)

                start = time.time()
                if gpu is None:
                    gpu = int(T.cuda.is_available())
                trainer, checkpoint = self.create_trainer(d, hyperparams.get('max-iters', 100),
                                                          hyperparams['pipeline_choice'], gpu)
                trainer.fit(m)
                ckpt = T.load(checkpoint.best_model_path)
                m.load_state_dict(ckpt['state_dict'])
                end = time.time()
                time_cost = end - start

                if os.path.isfile("{}/time_results.json".format(d)):
                    with open("{}/time_results.json".format(d), 'r') as f:
                        time_results = json.load(f)
                else:
                    time_results = dict()
                time_results[hyperparams["pipeline_choice"]] = time_cost

                with open(f'{d}/time_results.json', 'w') as file:
                    json.dump(time_results, file)
                with open(f'{d}/{hyperparams["pipeline_choice"]}/hyperparams.json', 'w') as file:
                    new_hp = {k: str(v) for (k, v) in hyperparams.items()}
                    json.dump(new_hp, file)
                mask = m.get_mask().detach().cpu()
                T.save(mask, f'{d}/{hyperparams["pipeline_choice"]}/mask.th')
                with open(f'{d}/{hyperparams["pipeline_choice"]}/mask.json', 'w') as file:
                    json.dump({
                        'variables': list(m.ncm.v),
                        'mask': mask.tolist(),
                    }, file)
                T.save(m.state_dict(), f'{d}/{hyperparams["pipeline_choice"]}/best.th')

                return m, time_results
            except Exception:
                e = d.replace("out/", "err/").rsplit('-', 1)[0]
                e_index = len(glob.glob(e + '/*'))
                e += '/%s' % e_index
                os.makedirs(e.rsplit('/', 1)[0], exist_ok=True)
                shutil.move(d, e)
                print(f'moved {d} to {e}')
                raise
