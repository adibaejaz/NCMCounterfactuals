""" This file mimics the structure of main.py, but only trains a baseline NCM to maximize the likelihood of the data, with 10 samples for each (true_graph, test_graph) pair.
Run it as follows.
python -m example3 score gan --lr 2e-5 --data-bs 256 --ncm-bs 256 --h-size 64 --u-size 2 --layer-norm --gan-mode wgangp --d-iters 1 -r 4 --id-query ate --max-lambda 1e-4 --min-lambda 1e-5 --max-query-iters 1000 --single-disc --gen-sigmoid --mc-sample-size 256 -G backdoor -t 10 -n 10000 -d 1 --gpu 0
Make sure nmax_epochs in gan_pipeline.py is set to an appropriate value.
"""

import itertools
import os
import warnings
import argparse
from src.ds import CTF
import concurrent.futures

import numpy as np
import torch as T

from src.pipeline import DivergencePipeline, GANPipeline, MLEPipeline
from src.scm.model_classes import XORModel, RoundModel
from src.scm.ctm import CTM
from src.scm.ncm import FF_NCM, GAN_NCM, MLE_NCM
from src.run import NCMRunner, NCMMinMaxRunner, ScoreNCMRunner
from src.ds.causal_graph import CausalGraph
from src.metric.queries import get_query, get_experimental_variables, is_q_id_in_G

three_node_graphs = ["three_indep", "iv", "iv_equiv", "backdoor", "frontdoor", "three_unconst"]
four_node_graphs = ["four_indep", "verma", "verma_equiv_1", "verma_equiv_2", "four_dag", "four_unconst"]

candidate_graphs = three_node_graphs
num_trials = 10 
num_samples = 10000
dim = 1
gpu = [0,1,2,3,4,5,6,7]
pipeline = GANPipeline
dat_model = CTM
ncm_model = GAN_NCM

# For large-scale experiments
# Prompt  
# python -m example3 score gan --lr 2e-5 --data-bs 256 --ncm-bs 256 --h-size 64 --u-size 2 --layer-norm --gan-mode wgangp --d-iters 1 -r 4 --id-query ate --max-lambda 1e-4 --min-lambda 1e-5 --max-query-iters 1000 --single-disc --gen-sigmoid --mc-sample-size 256 -G backdoor -t 10 -n 10000 -d 1 --gpu 0
hyperparams = {
    'lr': 2e-5,
    'data-bs': 256,
    'ncm-bs': 256,
    'h-layers': 2,
    'h-size': 64,
    'u-size': 2,
    'neural-pu': False,
    'layer-norm': True,
    'regions': 20,
    'c2-scale': 1.0,
    'gen-bs': 10000,
    'gan-mode': 'wganp',
    'd-iters': 1,
    'grad-clamp': 0.01,
    'gp-weight': 10.0,
    'query-track': None,
    'id-reruns': 4,
    'max-query-iters': 1000,
    'min-lambda': 1e-5,
    'max-lambda': 1e-4,
    'mc-sample-size': 256,
    'single-disc': True,
    'gen-sigmoid': True,
    'perturb-sd': 0.1,
    'full-batch': False,
    'positivity': True,
    'eval-query': None,
    'do-var-list': [{}],
}

runner = ScoreNCMRunner(pipeline, dat_model, ncm_model)

# GENERATE DATA
def run_generation(args):
    true_graph, trial_index = args
    return runner.generate_data(true_graph, num_samples, dim, trial_index, hyperparams=hyperparams, 
                                lockinfo=os.environ.get('SLURM_JOB_ID', ''))

# Create argument pairs
tasks = [(true_graph, trial_index) for true_graph in candidate_graphs for trial_index in range(num_trials)]

# Use ProcessPoolExecutor for parallel execution
with concurrent.futures.ProcessPoolExecutor() as executor:
    results = list(executor.map(run_generation, tasks))

# TEST GRAPHS

# runner.run_score(true_graph="three_indep", test_graphs=["three_indep"], num_samples=100, dim=1, num_trials=1, \
#                  hyperparams=hyperparams, gpu=[0], lockinfo=os.environ.get('SLURM_JOB_ID', ''), verbose=False)
