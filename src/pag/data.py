import os

import numpy as np
import pandas as pd
import torch as T
from torch.utils.data import Dataset

from src.ds.causal_graph import CausalGraph
from src.scm.ctm import CTM


def _validate_binary_tensor(data: T.Tensor, path=None):
    if data.ndim != 2:
        raise ValueError("Expected a 2D binary table, got shape {} from {}.".format(tuple(data.shape), path))
    if data.numel() == 0:
        raise ValueError("Binary table is empty{}.".format("" if path is None else " at {}".format(path)))
    if not T.all((data == 0) | (data == 1)):
        raise ValueError("Binary table must only contain 0/1 values{}.".format(
            "" if path is None else " at {}".format(path)))


def _to_binary_tensor(data):
    if isinstance(data, T.Tensor):
        tensor = data.clone().detach()
    else:
        tensor = T.as_tensor(data)

    if tensor.dtype == T.bool:
        tensor = tensor.long()
    elif tensor.dtype.is_floating_point:
        tensor = tensor.round().long()
    else:
        tensor = tensor.long()

    return tensor


def _full_support_probability_table(data, columns):
    tensor = _to_binary_tensor(data).cpu()
    _validate_binary_tensor(tensor)
    if tensor.shape[1] != 3:
        raise ValueError("Full-support probability table currently expects exactly three variables.")

    combos = T.cartesian_prod(T.tensor([0, 1]), T.tensor([0, 1]), T.tensor([0, 1])).long()
    combo_ids = combos[:, 0] * 4 + combos[:, 1] * 2 + combos[:, 2]
    sample_ids = tensor[:, 0].long() * 4 + tensor[:, 1].long() * 2 + tensor[:, 2].long()
    counts = T.bincount(sample_ids, minlength=8).float()
    probs = counts / counts.sum()

    df = pd.DataFrame(combos.numpy(), columns=[str(col) for col in columns])
    df['P(V)'] = probs.numpy()
    df['_combo_id'] = combo_ids.numpy()
    return df.sort_values('_combo_id').drop(columns=['_combo_id']).reset_index(drop=True)


def _load_pt_like(path):
    loaded = T.load(path)
    if isinstance(loaded, dict):
        if 'data' in loaded:
            data = loaded['data']
            columns = loaded.get('columns')
        else:
            first_key = next(iter(loaded))
            data = loaded[first_key]
            columns = loaded.get('columns')
    else:
        data = loaded
        columns = None
    return data, columns


def load_binary_table(path):
    ext = os.path.splitext(path)[1].lower()
    columns = None

    if ext == '.csv':
        df = pd.read_csv(path)
        columns = list(df.columns)
        data = df.to_numpy()
    elif ext == '.npy':
        data = np.load(path)
    elif ext == '.npz':
        loaded = np.load(path)
        if 'data' in loaded:
            data = loaded['data']
        else:
            first_key = loaded.files[0]
            data = loaded[first_key]
        if 'columns' in loaded:
            columns = [str(v) for v in loaded['columns'].tolist()]
    elif ext in {'.pt', '.pth'}:
        data, columns = _load_pt_like(path)
    else:
        raise ValueError("Unsupported data format '{}'. Use csv, npy, npz, pt, or pth.".format(ext))

    tensor = _to_binary_tensor(data)
    _validate_binary_tensor(tensor, path=path)

    if columns is None:
        columns = ["v{}".format(i) for i in range(tensor.shape[1])]
    elif len(columns) != tensor.shape[1]:
        raise ValueError("Column count does not match table width for {}.".format(path))

    return tensor.float(), [str(col) for col in columns]


def make_synthetic_binary_table(
    num_rows,
    graph_name,
    seed=0,
    regions=20,
    c2_scale=1.0,
    batch_size=10000,
    min_state_prob=0.01,
    resample_seed_offset=2026,
    max_resample_attempts=100,
):
    if graph_name != 'chain':
        raise ValueError("Only graph='chain' is supported right now.")
    if min_state_prob < 0.0:
        raise ValueError("min_state_prob must be non-negative.")

    cg_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'dat', 'cg', '{}.cg'.format(graph_name)))
    columns = ["X", "Y", "Z"]
    cg = CausalGraph.read(cg_path)
    v_sizes = {k: 1 for k in cg}

    for attempt in range(max_resample_attempts):
        cur_seed = seed + attempt * resample_seed_offset
        dat_m = CTM(cg, v_size=v_sizes, regions=regions, c2_scale=c2_scale, batch_size=batch_size, seed=cur_seed)
        observational_samples = dat_m(n=num_rows, do={})
        samples = T.cat([observational_samples[col].float() for col in columns], dim=1)

        if min_state_prob == 0.0:
            return samples, columns

        prob_table = _full_support_probability_table(samples, columns)
        if float(prob_table['P(V)'].min()) >= min_state_prob:
            return samples, columns

    raise RuntimeError(
        "Failed to generate chain data with all 8 states having probability >= {:.4f} after {} attempts."
        .format(min_state_prob, max_resample_attempts)
    )


def split_binary_table(data, val_fraction=0.1, seed=0):
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1).")

    n = data.shape[0]
    if val_fraction == 0.0 or n < 2:
        return data, None

    rng = np.random.RandomState(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    val_size = max(1, int(round(n * val_fraction)))
    if val_size >= n:
        val_size = n - 1

    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    return data[train_idx], data[val_idx]


def binary_table_probability_table(data, columns):
    tensor = _to_binary_tensor(data).cpu()
    _validate_binary_tensor(tensor)

    df = pd.DataFrame(tensor.numpy(), columns=[str(col) for col in columns])
    return (df.groupby(list(df.columns))
            .apply(lambda x: len(x) / len(df))
            .rename('P(V)').reset_index()
            [[*df.columns, 'P(V)']])


def probability_table_kl(truth_table, model_table, eps=1e-7):
    cols = list(truth_table.columns[:-1])
    joined = truth_table.merge(model_table, how='left', on=cols, suffixes=['_t', '_m']).fillna(eps)
    p_t = joined['P(V)_t']
    p_m = joined['P(V)_m']
    return float((p_t * (np.log(p_t) - np.log(p_m))).sum())


class BinaryTableDataset(Dataset):
    def __init__(self, data: T.Tensor):
        _validate_binary_tensor(_to_binary_tensor(data))
        self.data = data.float()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]
