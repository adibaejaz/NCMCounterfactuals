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
import multiprocessing

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

pipeline = GANPipeline
dat_model = CTM
ncm_model = GAN_NCM
hyperparams = {
    'lr': 4e-3,
    'data-bs': 100,
    'ncm-bs': 100,
    'h-layers': 2,
    'h-size': 64,
    'u-size': 1,
    'neural-pu': False,
    'layer-norm': False,
    'regions': 20,
    'c2-scale': 1.0,
    'gen-bs': 100,
    'gan-mode': 'vanilla',
    'd-iters': 1,
    'grad-clamp': 0.01,
    'gp-weight': 10.0,
    'query-track': None,
    'id-reruns': 1,
    'max-query-iters': 200,
    'min-lambda': 0.001,
    'max-lambda': 1.0,
    'mc-sample-size': 100,
    'single-disc': False,
    'gen-sigmoid': False,
    'perturb-sd': 0.1,
    'full-batch': False,
    'positivity': True,
    'eval-query': None,
    'do-var-list': [{}],
}

runner = ScoreNCMRunner(pipeline, dat_model, ncm_model)

runner.run_score(true_graph="three_indep", test_graphs=["three_indep"], num_samples=100, dim=1, num_trials=1, \
                 hyperparams=hyperparams, gpu=[0], lockinfo=os.environ.get('SLURM_JOB_ID', ''), verbose=False)

"""
Map for main function
- Generate data for each ground truth graph, if not found already
- Run NCMRunner for each ground truth graph, with the given pipeline, generated data, and test graphs
"""