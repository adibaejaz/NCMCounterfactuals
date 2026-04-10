import os
import json
import hashlib
import glob
from tempfile import NamedTemporaryFile
from contextlib import contextmanager


class BaseRunner:
    def __init__(self, pipeline, dat_model, ncm_model):
        self.pipeline = pipeline
        self.pipeline_name = pipeline.__name__
        self.dat_model = dat_model
        self.dat_model_name = dat_model.__name__
        self.ncm_model = ncm_model
        self.ncm_model_name = ncm_model.__name__

    @contextmanager
    def lock(self, file, lockinfo):  # attempt to acquire a file lock; yield whether or not lock was acquired
        os.makedirs(os.path.dirname(file), exist_ok=True)
        os.makedirs('tmp/', exist_ok=True)
        with NamedTemporaryFile(dir='tmp/') as tmp:
            try:
                os.link(tmp.name, file)
                acquired_lock = True
            except FileExistsError:
                acquired_lock = os.stat(tmp.name).st_nlink == 2
        if acquired_lock:
            with open(file, 'w') as fp:
                fp.write(lockinfo)
            try:
                yield True
            finally:
                try:
                    os.remove(file)
                except FileNotFoundError:
                    pass
        else:
            yield False

    def get_key(self, cg_file, n, dim, trial_index):
        graph = cg_file.split('/')[-1].split('.')[0]
        return ('gen=%s-graph=%s-n_samples=%s-dim=%s-trial_index=%s'
                % (self.dat_model_name, graph, n, dim, trial_index))

    def get_run_key(self, cg_file, n, dim, trial_index, hyperparams=None):
        data_key = self.get_key(cg_file, n, dim, trial_index)
        if hyperparams is None:
            hyperparams = dict()
        hp_payload = json.dumps({k: str(v) for (k, v) in hyperparams.items()}, sort_keys=True)
        hp_hash = hashlib.sha256(
            ("pipeline={}|ncm={}|hp={}".format(
                self.pipeline_name, self.ncm_model_name, hp_payload)).encode()
        ).hexdigest()[:12]
        return "{}-run={}".format(data_key, hp_hash)

    def _hash_to_seed(self, payload):
        return int(hashlib.sha512(payload.encode()).hexdigest(), 16) & 0xffffffff

    def get_data_seed(self, key):
        return self._hash_to_seed("data|" + key)

    def get_train_seed(self, key, hyperparams=None):
        if hyperparams is None:
            hyperparams = dict()
        hp_payload = json.dumps({k: str(v) for (k, v) in hyperparams.items()}, sort_keys=True)
        payload = "train|{}|pipeline={}|ncm={}|hp={}".format(
            key, self.pipeline_name, self.ncm_model_name, hp_payload)
        return self._hash_to_seed(payload)

    def get_latest_checkpoint(self, directory):
        checkpoints = glob.glob(os.path.join(directory, "checkpoints", "*.ckpt"))
        if not checkpoints:
            return None
        last_ckpt = os.path.join(directory, "checkpoints", "last.ckpt")
        if os.path.isfile(last_ckpt):
            return last_ckpt
        return max(checkpoints, key=os.path.getmtime)

    def run(self, exp_name, cg_file, n, dim, trial_index, hyperparams=None, gpu=None, data_bundle=None,
            lockinfo=os.environ.get('SLURM_JOB_ID', ''), verbose=False):
        raise NotImplementedError()
