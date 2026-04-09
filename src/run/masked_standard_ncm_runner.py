"""Standard train-and-evaluate runner for single-model masked NCM experiments.

This is the masked analogue of ``NCMRunner``:
- generate or reuse one dataset
- build one masked pipeline/model
- compute initial metrics using mask-aware sampling
- train with early stopping
- reload the best checkpoint
- compute final metrics and save them
"""

import os
import glob
import shutil
import hashlib
import json
import random

import numpy as np
import torch as T
import pytorch_lightning as pl

from src.metric import evaluation
from src.ds.causal_graph import CausalGraph
from src.scm.ctm import CTM
from src.scm.scm import expand_do
from .base_runner import BaseRunner
from .data_setup import build_data_bundle


class MaskedNCMRunner(BaseRunner):
    """Run a standard single-model masked experiment and report quality metrics."""

    def __init__(self, pipeline, dat_model, ncm_model):
        super().__init__(pipeline, dat_model, ncm_model)

    def create_trainer(self, directory, gpu=None):
        checkpoint = pl.callbacks.ModelCheckpoint(dirpath=f'{directory}/checkpoints/', monitor="train_loss")
        return pl.Trainer(
            callbacks=[
                checkpoint,
                pl.callbacks.EarlyStopping(monitor='train_loss',
                                           patience=self.pipeline.patience,
                                           min_delta=self.pipeline.min_delta,
                                           check_on_train_epoch_end=True)
            ],
            max_epochs=self.pipeline.max_epochs,
            accumulate_grad_batches=1,
            logger=pl.loggers.TensorBoardLogger(f'{directory}/logs/'),
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
                if os.path.isfile(f'{d}/best.th'):
                    print('[done]', d)
                    return

                print('[running]', d)
                for file in glob.glob(f'{d}/*'):
                    if os.path.basename(file) != 'lock':
                        if os.path.isdir(file):
                            shutil.rmtree(file)
                        else:
                            try:
                                os.remove(file)
                            except FileNotFoundError:
                                pass

                data_seed = self.get_data_seed(key)
                train_seed = self.get_train_seed(key, hyperparams)
                print('Data Key:', key)
                print('Run Key:', run_key)
                print('Data Seed:', data_seed)
                print('Train Seed:', train_seed)

                print('Generating data')
                if data_bundle is None:
                    T.manual_seed(data_seed)
                    T.cuda.manual_seed_all(data_seed)
                    np.random.seed(data_seed)
                    random.seed(data_seed)
                    data_bundle = build_data_bundle(self.dat_model, cg_file, n, dim, hyperparams, data_seed)
                T.manual_seed(train_seed)
                T.cuda.manual_seed_all(train_seed)
                np.random.seed(train_seed)
                random.seed(train_seed)
                cg = data_bundle.cg
                dat_m = data_bundle.dat_m
                dat_sets = data_bundle.dat_sets
                m = self.pipeline(dat_m, hyperparams["do-var-list"], dat_sets, cg, dim, hyperparams=hyperparams,
                                  ncm_model=self.ncm_model)

                print("Calculating metrics")
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

                if gpu is None:
                    gpu = int(T.cuda.is_available())
                trainer, checkpoint = self.create_trainer(d, gpu)
                trainer.fit(m)
                ckpt = T.load(checkpoint.best_model_path)
                m.load_state_dict(ckpt['state_dict'])
                results = evaluation.all_metrics(
                    m.generator, m.ncm, hyperparams["do-var-list"], dat_sets,
                    n=1000000, query_track=hyperparams['eval-query'],
                    ncm_kwargs=self._ncm_kwargs(m))
                dag_h = m.dag_penalty(
                    penalty_type=m.cycle_penalty_type,
                    dagma_s=m.dagma_s)
                results["dag_h"] = dag_h.item() if T.is_tensor(dag_h) else dag_h
                print(results)

                with open(f'{d}/results.json', 'w') as file:
                    json.dump(results, file)
                with open(f'{d}/hyperparams.json', 'w') as file:
                    new_hp = {k: str(v) for (k, v) in hyperparams.items()}
                    json.dump(new_hp, file)
                T.save(dat_sets, f'{d}/dat.th')
                mask = m.get_mask().detach().cpu()
                T.save(mask, f'{d}/mask.th')
                with open(f'{d}/mask.json', 'w') as file:
                    json.dump({
                        'variables': list(m.ncm.v),
                        'mask': mask.tolist(),
                    }, file)
                T.save(m.state_dict(), f'{d}/best.th')

                return m, results
            except Exception:
                e = d.replace("out/", "err/").rsplit('-', 1)[0]
                e_index = len(glob.glob(e + '/*'))
                e += '/%s' % e_index
                os.makedirs(e.rsplit('/', 1)[0], exist_ok=True)
                shutil.move(d, e)
                print(f'moved {d} to {e}')
                raise
