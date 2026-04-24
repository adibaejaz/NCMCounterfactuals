"""Min-max runner for paired masked NCM experiments.

This is the masked analogue of ``NCMMinMaxRunner``:
- train a ``max`` masked model
- train a ``min`` masked model
- evaluate both using mask-aware sampling

It is intended for paired masked experiments rather than single-model quality
or runtime runs.
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
from src.scm.scm import expand_do
from .base_runner import BaseRunner


class MaskedNCMMinMaxRunner(BaseRunner):
    """Run a paired masked min/max experiment and report metrics for both models."""

    def __init__(self, pipeline, dat_model, ncm_model):
        super().__init__(pipeline, dat_model, ncm_model)

    def create_trainer(self, directory, max_epochs, r, gpu=None, phase=None):
        checkpoint_dir = f'{directory}/{r}/checkpoints/'
        if phase is not None:
            checkpoint_dir = f'{directory}/{r}/checkpoints_{phase}/'
        checkpoint = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_dir, monitor="objective_loss")
        return pl.Trainer(
            callbacks=[checkpoint],
            max_epochs=max_epochs,
            accumulate_grad_batches=1,
            logger=pl.loggers.TensorBoardLogger(f'{directory}/{r}/logs/'),
            log_every_n_steps=1,
            terminate_on_nan=True,
            gpus=gpu
        ), checkpoint

    def _ncm_kwargs(self, pl_model):
        return dict(
            mask=pl_model.get_mask(),
            use_dag_updates=pl_model.use_dag_updates,
        )

    def print_metrics(self, pl_model, do_var_list, dat_sets, verbose=False, stored_metrics=None, query_track=None, query_bounds=None):
        if stored_metrics is None:
            stored_metrics = dict()

        print("Calculating metrics")
        for i, dat_do_set in enumerate(do_var_list):
            name = evaluation.serialize_do(dat_do_set)
            if "true_{}".format(name) not in stored_metrics:
                stored_metrics["true_{}".format(name)] = evaluation.probability_table(
                    pl_model.generator, n=1000000, do={k: expand_do(v, n=1000000) for (k, v) in dat_do_set.items()})
            if "dat_{}".format(name) not in stored_metrics:
                stored_metrics["dat_{}".format(name)] = evaluation.probability_table(
                    pl_model.generator, n=1000000, do={k: expand_do(v, n=1000000) for (k, v) in dat_do_set.items()},
                    dat=dat_sets[i])

        if query_bounds is not None:
            stored_metrics.update(evaluation.scm_query_bound_metrics(
                pl_model.generator,
                n=1000000,
                stored=stored_metrics,
                **query_bounds))

        start_metrics = evaluation.all_metrics(
            pl_model.generator, pl_model.ncm, do_var_list, dat_sets,
            n=1000000, stored=stored_metrics,
            query_track=query_track,
            ncm_kwargs=self._ncm_kwargs(pl_model))
        if verbose:
            print(start_metrics)

        if query_track is not None:
            true_q = 'true_{}'.format(evaluation.serialize_query(query_track))
            if true_q not in stored_metrics:
                stored_metrics[true_q] = start_metrics[true_q]

        pl_model.update_metrics(stored_metrics)
        return stored_metrics

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
                    v_sizes = {k: 1 if k in {'X', 'Y', 'M', 'W'} else dim for k in cg}
                    if self.dat_model is CTM:
                        dat_m = self.dat_model(cg, v_size=v_sizes, regions=hyperparams.get('regions', 20),
                                               c2_scale=hyperparams.get('c2-scale', 1.0),
                                               batch_size=hyperparams.get('gen-bs', 10000),
                                               seed=seed)
                    else:
                        dat_m = self.dat_model(cg, dim=dim, seed=seed)

                    if os.path.isfile(f'{d}/dat.th'):
                        dat_sets = T.load(f'{d}/dat.th')
                        positivity = True
                    else:
                        dat_sets = []
                        all_positive = True
                        for dat_do_set in hyperparams["do-var-list"]:
                            var_dims = 0
                            for k in v_sizes:
                                if k not in dat_do_set:
                                    var_dims += v_sizes[k]

                            expand_do_set = {k: expand_do(v, n=n) for (k, v) in dat_do_set.items()}
                            dat_set = dat_m(n=n, do=expand_do_set)
                            if hyperparams["positivity"]:
                                prob_table = evaluation.probability_table(dat=dat_set)
                                if len(prob_table) != (2 ** var_dims):
                                    all_positive = False
                                    print(prob_table)

                            dat_sets.append(dat_m(n=n, do=expand_do_set))

                        positivity = all_positive
                        if positivity:
                            T.save(dat_sets, f'{d}/dat.th')

                with open(f'{d}/hyperparams.json', 'w') as file:
                    new_hp = {k: str(v) for (k, v) in hyperparams.items()}
                    json.dump(new_hp, file)

                if gpu is None:
                    gpu = int(T.cuda.is_available())
                stored_metrics = dict(data_bundle.stored_metrics) if data_bundle is not None else dict()
                for r in range(hyperparams.get("id-reruns", 1)):
                    os.makedirs(f'{d}/{r}/', exist_ok=True)
                    if not os.path.isfile(f'{d}/{r}/best_max.th'):
                        for file in glob.glob(f'{d}/{r}/*'):
                            if os.path.isdir(file):
                                shutil.rmtree(file)
                            else:
                                try:
                                    os.remove(file)
                                except FileNotFoundError:
                                    pass

                        new_key = "{}-run={}".format(key, r)
                        seed = int(hashlib.sha512(new_key.encode()).hexdigest(), 16) & 0xffffffff
                        seed = (seed + int(hyperparams.get("train-seed-offset", 0))) & 0xffffffff
                        T.manual_seed(seed)
                        np.random.seed(seed)
                        print("Run {} seed: {}".format(r, seed))

                        m_max = self.pipeline(dat_m, hyperparams["do-var-list"], dat_sets, cg, dim,
                                              hyperparams=hyperparams, ncm_model=self.ncm_model,
                                              max_query=hyperparams.get('max-query-1', None))
                        m_min = self.pipeline(dat_m, hyperparams["do-var-list"], dat_sets, cg, dim,
                                              hyperparams=hyperparams, ncm_model=self.ncm_model,
                                              max_query=hyperparams.get('max-query-2', None))

                        trainer_max, checkpoint_max = self.create_trainer(
                            d, hyperparams.get('max-query-iters', 3000), r, gpu)
                        trainer_min, checkpoint_min = self.create_trainer(
                            d, hyperparams.get('max-query-iters', 3000), r, gpu)

                        print("\nTraining max model...")
                        stored_metrics = self.print_metrics(
                            m_max, hyperparams['do-var-list'], dat_sets,
                            verbose=verbose, stored_metrics=stored_metrics,
                            query_track=hyperparams["eval-query"],
                            query_bounds=hyperparams.get("query-bound-spec"))
                        trainer_max.fit(m_max)
                        ckpt_max = T.load(checkpoint_max.best_model_path)
                        m_max.load_state_dict(ckpt_max['state_dict'])

                        print("\nTraining min model...")
                        stored_metrics = self.print_metrics(
                            m_min, hyperparams['do-var-list'], dat_sets,
                            verbose=verbose, stored_metrics=stored_metrics,
                            query_track=hyperparams["eval-query"],
                            query_bounds=hyperparams.get("query-bound-spec"))
                        trainer_min.fit(m_min)
                        ckpt_min = T.load(checkpoint_min.best_model_path)
                        m_min.load_state_dict(ckpt_min['state_dict'])

                        theta_only_extra_epochs = int(hyperparams.get('theta-only-extra-epochs', 0))
                        if theta_only_extra_epochs > 0:
                            theta_only_extra_lr = hyperparams.get('theta-only-extra-lr', None)

                            print("\nTraining max model theta-only extension...")
                            m_max.start_theta_only_phase(
                                theta_lr=theta_only_extra_lr,
                                final_query_reg=hyperparams.get('theta-only-final-query-reg', True))
                            trainer_max_extra, checkpoint_max_extra = self.create_trainer(
                                d, theta_only_extra_epochs, r, gpu, phase='theta_only_max')
                            trainer_max_extra.fit(m_max)
                            if checkpoint_max_extra.best_model_path:
                                ckpt_max = T.load(checkpoint_max_extra.best_model_path)
                                m_max.load_state_dict(ckpt_max['state_dict'])

                            print("\nTraining min model theta-only extension...")
                            m_min.start_theta_only_phase(
                                theta_lr=theta_only_extra_lr,
                                final_query_reg=hyperparams.get('theta-only-final-query-reg', True))
                            trainer_min_extra, checkpoint_min_extra = self.create_trainer(
                                d, theta_only_extra_epochs, r, gpu, phase='theta_only_min')
                            trainer_min_extra.fit(m_min)
                            if checkpoint_min_extra.best_model_path:
                                ckpt_min = T.load(checkpoint_min_extra.best_model_path)
                                m_min.load_state_dict(ckpt_min['state_dict'])

                        results = evaluation.all_metrics_minmax(
                            m_max.generator, m_min.ncm, m_max.ncm, hyperparams["do-var-list"], dat_sets,
                            n=100000, stored=stored_metrics, query_track=hyperparams["eval-query"],
                            query_bounds=hyperparams.get("query-bound-spec"),
                            ncm_min_kwargs=self._ncm_kwargs(m_min),
                            ncm_max_kwargs=self._ncm_kwargs(m_max))
                        print(results)

                        with open(f'{d}/{r}/results.json', 'w') as file:
                            json.dump(results, file)
                        mask_min = m_min.get_mask().detach().cpu()
                        mask_max = m_max.get_mask().detach().cpu()
                        T.save(mask_min, f'{d}/{r}/mask_min.th')
                        T.save(mask_max, f'{d}/{r}/mask_max.th')
                        with open(f'{d}/{r}/mask_min.json', 'w') as file:
                            json.dump({
                                'variables': list(m_min.ncm.v),
                                'mask': mask_min.tolist(),
                            }, file)
                        with open(f'{d}/{r}/mask_max.json', 'w') as file:
                            json.dump({
                                'variables': list(m_max.ncm.v),
                                'mask': mask_max.tolist(),
                            }, file)
                        T.save(m_min.state_dict(), f'{d}/{r}/best_min.th')
                        T.save(m_max.state_dict(), f'{d}/{r}/best_max.th')
                    else:
                        print("Done with run {}".format(r))

                T.save(dict(), f'{d}/best.th')
                return True
            except Exception:
                e = d.replace("out/", "err/").rsplit('-', 1)[0]
                e_index = len(glob.glob(e + '/*'))
                e += '/%s' % e_index
                os.makedirs(e.rsplit('/', 1)[0], exist_ok=True)
                shutil.move(d, e)
                print(f'moved {d} to {e}')
                raise
