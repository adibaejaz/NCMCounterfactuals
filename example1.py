from src.run import NCMRunner
from src.pipeline import GANPipeline
from src.scm.ctm import CTM
from src.scm.ncm import GAN_NCM


pipeline = GANPipeline
dat_model = CTM
ncm_model = GAN_NCM
gen_graph = "backdoor"
gen_cg_file = "dat/cg/backdoor.cg"
ncm_cg_file = "dat/cg/frontdoor.cg"
gpu = None

hyperparams = {
    'lr': 2e-5,
    'data-bs': 256,
    'ncm-bs': 256,
    'h-layers': 2,  # Default value as specified in the help
    'h-size': 64,
    'u-size': 2,
    'neural-pu': False,  # Default value as specified in the help
    'layer-norm': True,
    'regions': 20,  # Default value as specified in the help
    'c2-scale': 1.0,  # Default value as scale_regions is not provided
    'gen-bs': 10000,  # Default value as specified in the help
    'gan-mode': 'wgangp',
    'd-iters': 1,
    'grad-clamp': 0.01,  # Default value as specified in the help
    'gp-weight': 10.0,  # Default value as specified in the help
    'query-track': None,  # Not provided in the command
    'id-reruns': 4,
    'max-query-iters': 1000,
    'min-lambda': 1e-5,
    'max-lambda': 1e-4,
    'mc-sample-size': 256,
    'single-disc': True,
    'gen-sigmoid': True,
    'perturb-sd': 0.1,  # Default value as specified in the help
    'full-batch': False,  # Default value as specified in the help
    'positivity': True,  # Default value as no_positivity is not provided.
    'do-var-list': [],
    'query-track': None,
    'eval-query': None,
}


runner = NCMRunner(pipeline, dat_model, ncm_model)
runner.run_score("bdscore", gen_cg_file, ncm_cg_file, n=10000, dim=1, trial_index=1, hyperparams=hyperparams)

# from NeuralCausalModels.src.ds import CausalGraph
# import os

# # Three-node causal graphs
# bd_file = 'NeuralCausalModels/dat/cg/backdoor.cg'
# fd_file = 'NeuralCausalModels/dat/cg/frontdoor.cg'
# iv_file = 'NeuralCausalModels/at/cg/iv.cg'

# # Arguments
# n_epochs = 100
# n_trials = 1
# n_samples = 1000
# dim = 3
# n_reruns = 1
# gpu_used  = None

# # Generate observational data according to ground truth graph
# bd_ctm, bd_dat = datagen(cg_file=bd_file, n=n_samples)


# # Train and compute log-likelihood of various 3-node graphs

# fd_cg = CausalGraph.read(fd_file)
# fd_path = "/frontdoor"

# for i in range(n_trials):
#     while True:
#         if not run_score(fd_path, bd_ctm, bd_dat, fd_file, n_epochs, n_reruns,
#            lockinfo=os.environ.get('SLURM_JOB_ID', ''), gpu=None):
#             break
