import sys
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch as T

from src.scm.scm import expand_do
from src.ds import CTF, CTFTerm


def eval_query(m, ctf, n=1000000):
    if isinstance(ctf, CTF):
        return m.compute_ctf(ctf, n=n, evaluating=True)
    else:
        return ctf_sum(m, ctf, n=n)


def ctf_sum(m, ctf_list, n=1000000):
    total = 0
    for ctf, sign in ctf_list:
        total += sign * m.compute_ctf(ctf, n=n, evaluating=True)
    return total


def probability_table(m=None, n=1000000, do={}, dat=None):
    assert m is not None or dat is not None

    if dat is None:
        dat = m(n, do=do, evaluating=True)

    cols = dict()
    for v in sorted(dat):
        result = dat[v].detach().numpy()
        for i in range(result.shape[1]):
            cols["{}{}".format(v, i)] = np.squeeze(result[:, i])

    df = pd.DataFrame(cols)
    return (df.groupby(list(df.columns))
            .apply(lambda x: len(x) / len(df))
            .rename('P(V)').reset_index()
            [[*df.columns, 'P(V)']])


def kl(truth, ncm, n=1000000, do={}, true_pv=None):
    m_table = probability_table(ncm, n=n, do=do)
    t_table = true_pv if true_pv is not None else probability_table(truth, n=n, do=do)
    cols = list(t_table.columns[:-1])
    joined_table = t_table.merge(m_table, how='left', on=cols, suffixes=['_t', '_m']).fillna(0.0000001)
    p_t = joined_table['P(V)_t']
    p_m = joined_table['P(V)_m']
    return (p_t * (np.log(p_t) - np.log(p_m))).sum()


def supremum_norm(truth, ncm, n=1000000, do={}, true_pv=None):
    m_table = probability_table(ncm, n=n, do=do)
    t_table = true_pv if true_pv is not None else probability_table(truth, n=n)
    cols = list(t_table.columns[:-1])
    joined_table = t_table.merge(m_table, how='outer', on=cols, suffixes=['_t', '_m']).fillna(0.0)
    p_t = joined_table['P(V)_t']
    p_m = joined_table['P(V)_m']
    return (p_m - p_t).abs().max()


def serialize_do(do_set):
    if len(do_set) == 0:
        return "P(V)"
    name = "P(V | do("
    for (k, v) in do_set.items():
        name += "{}={}, ".format(k, v)
    name = name[:-2] + "))"
    return name


def serialize_query(query):
    if isinstance(query, CTF):
        if query.name is not None:
            return query.name
    else:
        q = query[0][0]
        if q.name is not None:
            return q.name
    return str(query)


def _extract_atomic_query_assignment(query):
    if not isinstance(query, CTF):
        raise ValueError("bounds currently require an atomic CTF query.")
    if len(query.term_set) != 1 or len(query.cond_term_set) != 0:
        raise ValueError("bounds currently require a single unconditioned atomic query.")

    term = next(iter(query.term_set))
    if term.vars != {'Y'} or set(term.var_vals.keys()) != {'Y'}:
        raise ValueError("bounds currently only support queries over Y.")
    if set(term.do_vals.keys()) != {'X'}:
        raise ValueError("bounds currently only support interventions on X.")

    return int(term.do_vals['X']), int(term.var_vals['Y'])


def _joint_probability(prob_table, assignments):
    mask = prob_table['P(V)'] >= 0
    for column, value in assignments.items():
        if column not in prob_table.columns:
            raise ValueError("probability table must contain column '{}'.".format(column))
        mask = mask & (prob_table[column] == value)
    return float(prob_table.loc[mask, 'P(V)'].sum())


def _conditional_yx_given_z(prob_table, y_value, x_value, z_value):
    p_z = _joint_probability(prob_table, {'Z0': z_value})
    if p_z <= 0:
        raise ValueError("probability table has zero mass on Z={}.".format(z_value))
    p_yxz = _joint_probability(prob_table, {'Y0': y_value, 'X0': x_value, 'Z0': z_value})
    return p_yxz / p_z


def _iv_family_y1_bounds(prob_table, x_value):
    p_y0x0_z0 = _conditional_yx_given_z(prob_table, 0, 0, 0)
    p_y0x1_z0 = _conditional_yx_given_z(prob_table, 0, 1, 0)
    p_y1x0_z0 = _conditional_yx_given_z(prob_table, 1, 0, 0)
    p_y1x1_z0 = _conditional_yx_given_z(prob_table, 1, 1, 0)
    p_y0x0_z1 = _conditional_yx_given_z(prob_table, 0, 0, 1)
    p_y0x1_z1 = _conditional_yx_given_z(prob_table, 0, 1, 1)
    p_y1x0_z1 = _conditional_yx_given_z(prob_table, 1, 0, 1)
    p_y1x1_z1 = _conditional_yx_given_z(prob_table, 1, 1, 1)

    if x_value == 0:
        lower_bound = max(
            p_y1x0_z1,
            p_y1x0_z0,
            p_y1x0_z0 + p_y1x1_z0 - p_y0x0_z1 - p_y1x1_z1,
            p_y0x1_z0 + p_y1x0_z0 - p_y0x0_z1 - p_y0x1_z1,
        )
        upper_bound = min(
            1.0 - p_y0x0_z1,
            1.0 - p_y0x0_z0,
            p_y0x1_z0 + p_y1x0_z0 + p_y1x0_z1 + p_y1x1_z1,
            p_y1x0_z0 + p_y1x1_z0 + p_y0x1_z1 + p_y1x0_z1,
        )
        return lower_bound, upper_bound
    if x_value == 1:
        lower_bound = max(
            p_y1x1_z0,
            p_y1x1_z1,
            -p_y0x0_z0 - p_y0x1_z0 + p_y0x0_z1 + p_y1x1_z1,
            -p_y0x1_z0 - p_y1x0_z0 + p_y1x0_z1 + p_y1x1_z1,
        )
        upper_bound = min(
            1.0 - p_y0x1_z1,
            1.0 - p_y0x1_z0,
            p_y0x0_z0 + p_y1x1_z0 + p_y1x0_z1 + p_y1x1_z1,
            p_y1x0_z0 + p_y1x1_z0 + p_y0x0_z1 + p_y1x1_z1,
        )
        return lower_bound, upper_bound
    raise ValueError("x_value must be binary.")


def atomic_query_bounds_from_probability_table(graph_name, query, prob_table):
    graph_name = graph_name.lower()
    x_value, y_value = _extract_atomic_query_assignment(query)
    if 'P(V)' not in prob_table.columns or 'X0' not in prob_table.columns or 'Y0' not in prob_table.columns:
        raise ValueError("probability table must contain X0, Y0, and P(V) columns.")

    if graph_name == "bow":
        p_yx = _joint_probability(prob_table, {'X0': x_value, 'Y0': y_value})
        p_x = _joint_probability(prob_table, {'X0': x_value})
        lower_bound = p_yx
        upper_bound = p_yx + 1.0 - p_x
        return lower_bound, upper_bound

    if graph_name in {"iv", "doublebow", "double_bow"}:
        if 'Z0' not in prob_table.columns:
            raise ValueError("probability table must contain Z0 for graph='{}'.".format(graph_name))
        y1_lower, y1_upper = _iv_family_y1_bounds(prob_table, x_value)
        if y_value == 1:
            return y1_lower, y1_upper
        return 1.0 - y1_upper, 1.0 - y1_lower

    raise ValueError("bounds are currently only implemented for graph='bow', 'iv', and 'doublebow'.")


def all_metrics(truth, ncm, dat_dos, dat_sets, n=1000000, stored=None, query_track=None, include_sup=False):
    true_ps = dict()
    dat_ps = dict()
    m = dict()
    m["total_true_KL"] = 0
    m["total_dat_KL"] = 0
    if include_sup:
        m["total_true_supnorm"] = 0
        m["total_dat_supnorm"] = 0
    for i, do_set in enumerate(dat_dos):
        name = serialize_do(do_set)
        true_name = "true_{}".format(name)
        dat_name = "dat_{}".format(name)
        expanded_do_dat = {k: expand_do(v, n) for (k, v) in do_set.items()}
        if stored is None or true_name not in stored:
            true_ps[name] = None
        else:
            true_ps[name] = stored[true_name]
        if stored is None or dat_name not in stored:
            dat_ps[name] = probability_table(m=None, n=n, do=expanded_do_dat, dat=dat_sets[i])
        else:
            dat_ps[name] = stored[dat_name]

        m["true_KL_{}".format(name)] = kl(truth, ncm, n=n, do=expanded_do_dat, true_pv=true_ps[name])
        m["dat_KL_{}".format(name)] = kl(truth, ncm, n=n, do=expanded_do_dat, true_pv=dat_ps[name])
        m["total_true_KL"] += m["true_KL_{}".format(name)]
        m["total_dat_KL"] += m["dat_KL_{}".format(name)]
        if include_sup:
            m["true_supnorm_{}".format(name)] = supremum_norm(truth, ncm, n=n, do=expanded_do_dat,
                                                              true_pv=true_ps[name])
            m["dat_supnorm_{}".format(name)] = supremum_norm(truth, ncm, n=n, do=expanded_do_dat,
                                                             true_pv=dat_ps[name])
            m["total_true_supnorm"] += m["true_supnorm_{}".format(name)]
            m["total_dat_supnorm"] += m["dat_supnorm_{}".format(name)]

    if query_track is not None:
        true_q = 'true_{}'.format(serialize_query(query_track))
        m[true_q] = eval_query(truth, query_track, n) if stored is None or true_q not in stored else stored[true_q]
        ncm_q = 'ncm_{}'.format(serialize_query(query_track))
        m[ncm_q] = eval_query(ncm, query_track, n)
        err_q = 'err_ncm_{}'.format(serialize_query(query_track))
        m[err_q] = m[true_q] - m[ncm_q]
    return m


def all_metrics_minmax(truth, ncm_min, ncm_max, dat_dos, dat_sets, n=1000000, stored=None, query_track=None):
    true_ps = dict()
    dat_ps = dict()
    m = dict()
    m["min_total_true_KL"] = 0
    m["min_total_dat_KL"] = 0
    m["min_total_true_supnorm"] = 0
    m["min_total_dat_supnorm"] = 0
    m["max_total_true_KL"] = 0
    m["max_total_dat_KL"] = 0
    m["max_total_true_supnorm"] = 0
    m["max_total_dat_supnorm"] = 0
    for i, do_set in enumerate(dat_dos):
        name = serialize_do(do_set)
        true_name = "true_{}".format(name)
        dat_name = "dat_{}".format(name)
        expanded_do_dat = {k: expand_do(v, n) for (k, v) in do_set.items()}
        if stored is None or true_name not in stored:
            true_ps[name] = None
        else:
            true_ps[name] = stored[true_name]
        if stored is None or dat_name not in stored:
            dat_ps[name] = probability_table(m=None, n=n, do=expanded_do_dat, dat=dat_sets[i])
        else:
            dat_ps[name] = stored[dat_name]

        m["min_true_KL_{}".format(name)] = kl(truth, ncm_min, n=n, do=expanded_do_dat, true_pv=true_ps[name])
        m["max_true_KL_{}".format(name)] = kl(truth, ncm_max, n=n, do=expanded_do_dat, true_pv=true_ps[name])
        m["min_dat_KL_{}".format(name)] = kl(truth, ncm_min, n=n, do=expanded_do_dat, true_pv=dat_ps[name])
        m["max_dat_KL_{}".format(name)] = kl(truth, ncm_max, n=n, do=expanded_do_dat, true_pv=dat_ps[name])
        m["min_true_supnorm_{}".format(name)] = supremum_norm(truth, ncm_min, n=n, do=expanded_do_dat,
                                                              true_pv=true_ps[name])
        m["max_true_supnorm_{}".format(name)] = supremum_norm(truth, ncm_max, n=n, do=expanded_do_dat,
                                                              true_pv=true_ps[name])
        m["min_dat_supnorm_{}".format(name)] = supremum_norm(truth, ncm_min, n=n, do=expanded_do_dat,
                                                             true_pv=dat_ps[name])
        m["max_dat_supnorm_{}".format(name)] = supremum_norm(truth, ncm_max, n=n, do=expanded_do_dat,
                                                             true_pv=dat_ps[name])

        m["min_total_true_KL"] += m["min_true_KL_{}".format(name)]
        m["min_total_dat_KL"] += m["min_dat_KL_{}".format(name)]
        m["min_total_true_supnorm"] += m["min_true_supnorm_{}".format(name)]
        m["min_total_dat_supnorm"] += m["min_dat_supnorm_{}".format(name)]
        m["max_total_true_KL"] += m["max_true_KL_{}".format(name)]
        m["max_total_dat_KL"] += m["max_dat_KL_{}".format(name)]
        m["max_total_true_supnorm"] += m["max_true_supnorm_{}".format(name)]
        m["max_total_dat_supnorm"] += m["max_dat_supnorm_{}".format(name)]

    if query_track is not None:
        true_q = 'true_{}'.format(serialize_query(query_track))
        m[true_q] = eval_query(truth, query_track, n) if stored is None or true_q not in stored else stored[true_q]
        min_ncm_q = 'min_ncm_{}'.format(serialize_query(query_track))
        max_ncm_q = 'max_ncm_{}'.format(serialize_query(query_track))
        m[min_ncm_q] = eval_query(ncm_min, query_track, n)
        m[max_ncm_q] = eval_query(ncm_max, query_track, n)
        min_err_q = 'min_err_ncm_{}'.format(serialize_query(query_track))
        max_err_q = 'max_err_ncm_{}'.format(serialize_query(query_track))
        m[min_err_q] = m[true_q] - m[min_ncm_q]
        m[max_err_q] = m[true_q] - m[max_ncm_q]
        minmax_gap = 'minmax_{}_gap'.format(serialize_query(query_track))
        m[minmax_gap] = m[max_ncm_q] - m[min_ncm_q]
    return m


def naive_metrics(truth, gan, do, n=1000000, dat_set=None, stored=None, query_track=None):

    m = dict()
    true_pv = stored['true_pv']
    dat_pv = stored['dat_pv']
    gan_pv = naive_probability_table(m=gan, n=n)
    true_q = 'true_{}'.format(serialize_query(query_track))
    m[true_q] = stored[true_q]
    # m[true_q] = eval_query(truth, query_track, n) if stored is None or true_q not in stored else stored[true_q]

    m["true_KL"] = naive_kl(true_pv, gan_pv)
    m["dat_KL"] = naive_kl(dat_pv, gan_pv)
    m["gan_naive_query"] = naive_query(gan, do=do, n=n)
    m["dat_naive_query"] = naive_query(do=do, dat=dat_set)

    err_q = 'err_gan_{}'.format(serialize_query(query_track))
    m[err_q] = m[true_q] - m["gan_naive_query"]
    return m


def naive_query(gan=None, do={}, dat=None, n=1000000):
    if dat is None:
        dat = gan(n, evaluating=True)
    if len(do) == 0:
        p_y1x1 = ((dat['X'] == 1) & (dat['Y'] == 1)).float().mean()
        p_y1x0 = ((dat['X'] == 0) & (dat['Y'] == 1)).float().mean()
        p_x1 = (dat['X'] == 1).float().mean()
        p_x0 = (dat['X'] == 0).float().mean()
        res = (p_y1x1 / (p_x1 + 1e-10) - p_y1x0 / (p_x0 + 1e-10)).item()
        return res
    else:
        p_y1 = (dat['Y'] == 1).float().mean().item()
        return p_y1


def naive_probability_table(m=None, dat_set=None, n=1000000):
    if dat_set is None:
        dat_set = m(n, evaluating=True)

    dat = {k: v.cpu() for (k, v) in dat_set.items()}
    return probability_table(n=n, dat=dat)


def naive_kl(t_table, m_table):
    cols = list(t_table.columns[:-1])
    joined_table = t_table.merge(m_table, how='left', on=cols, suffixes=['_t', '_m']).fillna(0.0000001)
    p_t = joined_table['P(V)_t']
    p_m = joined_table['P(V)_m']
    return (p_t * (np.log(p_t) - np.log(p_m))).sum()