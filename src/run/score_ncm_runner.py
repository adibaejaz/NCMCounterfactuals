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


class ScoreNCMRunner(BaseRunner):
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
    

    def get_key(self, true_graph, num_samples, dim, trial_index):
        return ('graph=%s-n_samples=%s-dim=%s-trial_index=%s'
                % (true_graph, num_samples, dim, trial_index))
    
    def get_graph_filename(self, graph):
        return f'dat/cg/{graph}.cg'
    

    def generate_data(self, true_graph, num_samples, dim, trial_index, hyperparams=None, \
                      lockinfo=os.environ.get('SLURM_JOB_ID', ''), verbose=False):
       
        key = self.get_key(true_graph, num_samples, dim, trial_index)
        data_filepath = 'out/%s/data.th' % key  # name of the data file
        model_filepath = 'out/%s/dat_m.th' % key

        data_exists = os.path.exists(data_filepath)
        model_exists = os.path.exists(model_filepath)

        
        if data_exists and not model_exists:
            os.remove(data_filepath)
            raise FileNotFoundError(f'Only data file found for {key}')
        if model_exists and not data_exists:
            os.remove(model_filepath)
            raise FileNotFoundError(f'Only model file found for {key}')
        elif data_exists and model_exists:
            print(f'Data already generated for {key}')
            try:
                with open(data_filepath, 'rb') as file:
                    dat_sets = T.load(file)
                with open(model_filepath, 'rb') as file:
                    dat_m = T.load(file)
                return dat_sets, dat_m
            except Exception as e:
                print(f'Error loading data or model: {e}')
                print(f'Removing files and regenerating data for {key}')
                os.remove(data_filepath)
                os.remove(model_filepath)
                data_exists = False
                model_exists = False
        elif not data_exists and not model_exists: # generate data 
                # set random seed to a hash of the parameter settings for reproducibility
                try:
                    seed = int(hashlib.sha512(key.encode()).hexdigest(), 16) & 0xffffffff
                    T.manual_seed(seed)
                    np.random.seed(seed)
                    print('Key:', key)
                    print('Seed:', seed)

                    # generate data-generating model, data, and model
                    print('Generating data')
                    cg = CausalGraph.read(self.get_graph_filename(true_graph))

                    # initialise data generating model
                    if self.dat_model is CTM:
                        v_sizes = {k: 1 if k in {'X', 'Y', 'M', 'W'} else dim for k in cg}
                        dat_m = self.dat_model(cg, v_size=v_sizes, regions=20,
                                    c2_scale=1.0,
                                    batch_size=10,
                                    seed=seed)
                    else:
                        raise NotImplementedError("Only CTM is supported for data generation.")

                    # generate data
                    # TODO: note that do-var-list is just [{}] for observational data
                    dat_sets = []
                    for dat_do_set in hyperparams["do-var-list"]:
                        expand_do_set = {k: expand_do(v, n=num_samples) for (k, v) in dat_do_set.items()}
                        dat_sets.append(dat_m(n=num_samples, do=expand_do_set))

                    # save data to file
                    os.makedirs(os.path.dirname(data_filepath), exist_ok=True)
                    with open(data_filepath, 'wb') as file:
                        T.save(dat_sets, file)
                    
                    # save data-generating model to file
                    with open(model_filepath, 'wb') as file:
                        T.save(dat_m, file)

                        
                    return dat_sets, dat_m
                except Exception as e:
                    print(f'Error generating data or model: {e} for {key}')
                    if os.path.exists(data_filepath):
                        os.remove(data_filepath)
                    if os.path.exists(model_filepath):
                        os.remove(model_filepath)
                    return None, None
    
    def test_graph(self, dat_m, dat_sets, true_graph, test_graph, num_samples, dim, num_trials, \
                  gpu=None, hyperparams=None, lockinfo=os.environ.get('SLURM_JOB_ID', ''), verbose=False):
        
        try:
            key = self.get_key(true_graph, num_samples, dim, num_trials)
            model_dir = 'out/%s/%s/' % (key, test_graph)

            # Check if best.th already exists
            if os.path.exists(f'{model_dir}/best.th'):
                print(f'Model already trained and saved for {key} and {test_graph}')
                return

            # Initialise pipeline
            cg = CausalGraph.read(self.get_graph_filename(true_graph))
            m = self.pipeline(dat_m, hyperparams["do-var-list"], dat_sets, cg, dim, \
                            hyperparams=hyperparams,
                                    ncm_model=self.ncm_model)
            
            # print info
            # TODO: note that do-var-list is just [{}] for observational data
            print("Calculating metrics")
            stored_metrics = dict()
            for i, dat_do_set in enumerate(hyperparams["do-var-list"]):
                name = evaluation.serialize_do(dat_do_set)
                stored_metrics["true_{}".format(name)] = evaluation.probability_table(
                    dat_m, n=1000000, do={k: expand_do(v, n=1000000) for (k, v) in dat_do_set.items()})
                stored_metrics["dat_{}".format(name)] = evaluation.probability_table(
                    dat_m, n=1000000, do={k: expand_do(v, n=1000000) for (k, v) in dat_do_set.items()},
                    dat=dat_sets[i])
            m.update_metrics(stored_metrics)

            # train model
            trainer, checkpoint = self.create_trainer(model_dir, gpu)
            trainer.fit(m)
            ckpt = T.load(checkpoint.best_model_path)
            m.load_state_dict(ckpt['state_dict'], strict=False)  # Allow partial loading
            results = evaluation.all_metrics(m.generator, m.ncm, hyperparams["do-var-list"], dat_sets,
                                            n=1000000, query_track=hyperparams['eval-query'])
            print(str(results) + "\n")

            # save results
            with open(f'{model_dir}/results.json', 'w') as file:
                json.dump(results, file)
            with open(f'{model_dir}/hyperparams.json', 'w') as file:
                new_hp = {k: str(v) for (k, v) in hyperparams.items()}
                json.dump(new_hp, file)
            T.save(m.state_dict(), f'{model_dir}/best.th')

        except Exception as e:
            print(f'Error running score: {e} for {key} and {test_graph}')
            

    def run_score(self, true_graph, test_graphs, num_samples, dim, num_trials, \
                  gpu=None, hyperparams=None, lockinfo=os.environ.get('SLURM_JOB_ID', ''), verbose=False):
        
        for trial_index in range(num_trials):
            dat_sets, dat_m = self.generate_data(true_graph, num_samples, dim, trial_index, hyperparams, lockinfo, verbose)
            if dat_sets is None or dat_m is None:
                raise RuntimeError('Error generating data')
            
            for test_graph in test_graphs:
                self.test_graph(dat_m, dat_sets, true_graph, test_graph, num_samples, dim, num_trials, \
                                gpu, hyperparams, lockinfo, verbose)            
