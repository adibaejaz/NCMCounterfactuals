import os
import json
import hashlib
import glob
from tempfile import NamedTemporaryFile
from contextlib import contextmanager


def _stable_jsonable(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {
            str(k): _stable_jsonable(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, list):
        return [_stable_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return {
            "__type__": "tuple",
            "items": [_stable_jsonable(v) for v in value],
        }
    if isinstance(value, (set, frozenset)):
        items = [_stable_jsonable(v) for v in value]
        items.sort(key=lambda item: json.dumps(item, sort_keys=True))
        return {
            "__type__": "set",
            "items": items,
        }
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "tolist"):
        return {
            "__type__": type(value).__name__,
            "items": _stable_jsonable(value.detach().cpu().tolist()),
        }
    if hasattr(value, "__dict__"):
        payload = {
            k: _stable_jsonable(v)
            for k, v in sorted(vars(value).items(), key=lambda item: item[0])
            if not k.startswith("_")
        }
        return {
            "__type__": type(value).__name__,
            "attrs": payload,
        }
    return repr(value)


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

    def _stable_hyperparams_payload(self, hyperparams=None):
        if hyperparams is None:
            hyperparams = dict()
        return json.dumps(_stable_jsonable(hyperparams), sort_keys=True)

    def get_run_key(self, cg_file, n, dim, trial_index, hyperparams=None):
        data_key = self.get_key(cg_file, n, dim, trial_index)
        hp_payload = self._stable_hyperparams_payload(hyperparams)
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
        hp_payload = self._stable_hyperparams_payload(hyperparams)
        payload = "train|{}|pipeline={}|ncm={}|hp={}".format(
            key, self.pipeline_name, self.ncm_model_name, hp_payload)
        return self._hash_to_seed(payload)

    def run_metadata(self, cg_file, n, dim, trial_index, key, run_key, data_seed=None,
                     train_seed=None, train_seeds=None, extra=None):
        metadata = {
            "data_key": key,
            "run_key": run_key,
            "graph_file": cg_file,
            "n_samples": n,
            "dim": dim,
            "trial_index": trial_index,
            "runner": type(self).__name__,
            "pipeline": self.pipeline_name,
            "dat_model": self.dat_model_name,
            "ncm_model": self.ncm_model_name,
        }
        if data_seed is not None:
            metadata["data_seed"] = int(data_seed)
        if train_seed is not None:
            metadata["train_seed"] = int(train_seed)
        if train_seeds is not None:
            metadata["train_seeds"] = [int(seed) for seed in train_seeds]
        if extra:
            metadata.update(extra)
        return metadata

    def serializable_hyperparams(self, hyperparams, metadata=None):
        new_hp = {k: str(v) for (k, v) in hyperparams.items()}
        if metadata:
            for key in ("data_seed", "train_seed", "train_seeds", "data_key", "run_key"):
                if key in metadata:
                    new_hp[key] = str(metadata[key])
        return new_hp

    def write_run_metadata(self, directory, metadata, hyperparams=None):
        os.makedirs(directory, exist_ok=True)
        if hyperparams is not None:
            with open(os.path.join(directory, "hyperparams.json"), "w") as file:
                json.dump(self.serializable_hyperparams(hyperparams, metadata), file)
        with open(os.path.join(directory, "run_metadata.json"), "w") as file:
            json.dump(metadata, file)

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
