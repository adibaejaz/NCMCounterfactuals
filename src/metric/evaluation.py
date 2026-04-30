import sys
import itertools
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch as T

from src.scm.scm import check_equal, expand_do
from src.ds import CTF, CTFTerm


def eval_query(m, ctf, n=1000000, model_kwargs=None):
    if model_kwargs is None:
        model_kwargs = dict()
    if isinstance(ctf, CTF):
        return m.compute_ctf(ctf, n=n, evaluating=True, **model_kwargs)
    else:
        return ctf_sum(m, ctf, n=n, model_kwargs=model_kwargs)


def ctf_sum(m, ctf_list, n=1000000, model_kwargs=None):
    if model_kwargs is None:
        model_kwargs = dict()
    total = 0
    for ctf, sign in ctf_list:
        total += sign * m.compute_ctf(ctf, n=n, evaluating=True, **model_kwargs)
    return total


def probability_table(m=None, n=1000000, do={}, dat=None, model_kwargs=None):
    assert m is not None or dat is not None
    if model_kwargs is None:
        model_kwargs = dict()

    if dat is None:
        dat = m(n, do=do, evaluating=True, **model_kwargs)

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


def kl(truth, ncm, n=1000000, do={}, true_pv=None, truth_kwargs=None, ncm_kwargs=None):
    m_table = probability_table(ncm, n=n, do=do, model_kwargs=ncm_kwargs)
    t_table = true_pv if true_pv is not None else probability_table(truth, n=n, do=do, model_kwargs=truth_kwargs)
    cols = list(t_table.columns[:-1])
    joined_table = t_table.merge(m_table, how='left', on=cols, suffixes=['_t', '_m']).fillna(0.0000001)
    p_t = joined_table['P(V)_t']
    p_m = joined_table['P(V)_m']
    return (p_t * (np.log(p_t) - np.log(p_m))).sum()


def supremum_norm(truth, ncm, n=1000000, do={}, true_pv=None, truth_kwargs=None, ncm_kwargs=None):
    m_table = probability_table(ncm, n=n, do=do, model_kwargs=ncm_kwargs)
    t_table = true_pv if true_pv is not None else probability_table(truth, n=n, model_kwargs=truth_kwargs)
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


def serialize_probability(var_vals, cond_vals=None, do_vals=None):
    event = ", ".join("{}={}".format(k, v) for (k, v) in sorted(var_vals.items()))
    prefix = "P({}".format(event)
    clauses = []
    if cond_vals:
        clauses.append(", ".join("{}={}".format(k, v) for (k, v) in sorted(cond_vals.items())))
    if do_vals:
        clauses.append("do({})".format(
            ", ".join("{}={}".format(k, v) for (k, v) in sorted(do_vals.items()))))
    if clauses:
        return prefix + " | " + ", ".join(clauses) + ")"
    return prefix + ")"


def event_probability(m=None, event=None, given=None, n=1000000, do=None, dat=None, model_kwargs=None):
    assert event is not None and len(event) > 0
    assert m is not None or dat is not None

    if given is None:
        given = dict()
    if do is None:
        do = dict()
    if model_kwargs is None:
        model_kwargs = dict()

    if dat is None:
        dat = m(n, do=do, evaluating=True, **model_kwargs)

    total = len(dat[next(iter(dat))])
    match = T.ones(total, dtype=T.bool)
    for (k, v) in given.items():
        match = match & check_equal(dat[k], v)

    match_count = int(match.long().sum().item())
    if match_count == 0:
        return float("nan")

    for (k, v) in event.items():
        match = match & check_equal(dat[k], v)

    return (match.float().sum() / match_count).item()


def _metric_event_probability(metrics, stored, key, truth, event, n, truth_kwargs, given=None):
    if key not in metrics:
        if stored is not None and key in stored:
            metrics[key] = stored[key]
        else:
            metrics[key] = event_probability(
                truth,
                event=event,
                given=given,
                n=n,
                model_kwargs=truth_kwargs)
    return metrics[key]


def _chain_query_bound_metrics(
        metrics,
        truth,
        outcome_event,
        treatment_var,
        treatment_value,
        do_name,
        n,
        truth_kwargs,
        stored):
    marginal_key = "true_{}".format(serialize_probability(outcome_event))
    p_y = _metric_event_probability(
        metrics, stored, marginal_key, truth, outcome_event, n, truth_kwargs)

    cond_key = "true_{}".format(serialize_probability(
        outcome_event,
        cond_vals={treatment_var: treatment_value}))
    p_y_given_t = _metric_event_probability(
        metrics, stored, cond_key, truth, outcome_event, n, truth_kwargs,
        given={treatment_var: treatment_value})

    return {
        "marginal": p_y,
        "conditional": p_y_given_t,
        "lower": min(p_y, p_y_given_t),
        "upper": max(p_y, p_y_given_t),
    }


def _backdoor_query_bound_metrics(
        metrics,
        truth,
        outcome_event,
        treatment_var,
        treatment_value,
        do_name,
        n,
        truth_kwargs,
        stored,
        adjustment_var="Z"):
    candidates = _chain_query_bound_metrics(
        metrics,
        truth,
        outcome_event,
        treatment_var,
        treatment_value,
        do_name,
        n,
        truth_kwargs,
        stored)

    adjusted = 0.0
    for adjustment_value in (0, 1):
        adjust_key = "true_{}".format(serialize_probability({adjustment_var: adjustment_value}))
        p_adjust = _metric_event_probability(
            metrics,
            stored,
            adjust_key,
            truth,
            {adjustment_var: adjustment_value},
            n,
            truth_kwargs)

        cond_key = "true_{}".format(serialize_probability(
            outcome_event,
            cond_vals={treatment_var: treatment_value, adjustment_var: adjustment_value}))
        p_y_given_t_adjust = _metric_event_probability(
            metrics,
            stored,
            cond_key,
            truth,
            outcome_event,
            n,
            truth_kwargs,
            given={treatment_var: treatment_value, adjustment_var: adjustment_value})
        adjusted += p_y_given_t_adjust * p_adjust

    adjusted_key = "true_adjusted_{}".format(do_name)
    metrics[adjusted_key] = adjusted

    values = [candidates["marginal"], candidates["conditional"], adjusted]
    return {
        "marginal": candidates["marginal"],
        "conditional": candidates["conditional"],
        "adjusted": adjusted,
        "lower": min(values),
        "upper": max(values),
    }


def _conditional_query_candidate(
        metrics,
        truth,
        outcome_event,
        given_var,
        given_value,
        n,
        truth_kwargs,
        stored):
    cond_key = "true_{}".format(serialize_probability(
        outcome_event,
        cond_vals={given_var: given_value}))
    return _metric_event_probability(
        metrics,
        stored,
        cond_key,
        truth,
        outcome_event,
        n,
        truth_kwargs,
        given={given_var: given_value})


def _adjusted_query_candidate(
        metrics,
        truth,
        outcome_event,
        treatment_var,
        treatment_value,
        adjustment_vars,
        do_name,
        n,
        truth_kwargs,
        stored):
    adjusted = 0.0
    adjustment_vars = tuple(adjustment_vars)
    for adjustment_values in itertools.product((0, 1), repeat=len(adjustment_vars)):
        adjustment_event = dict(zip(adjustment_vars, adjustment_values))
        adjust_key = "true_{}".format(serialize_probability(adjustment_event))
        p_adjust = _metric_event_probability(
            metrics,
            stored,
            adjust_key,
            truth,
            adjustment_event,
            n,
            truth_kwargs)

        cond_vals = {treatment_var: treatment_value, **adjustment_event}
        cond_key = "true_{}".format(serialize_probability(
            outcome_event,
            cond_vals=cond_vals))
        p_y_given_t_adjust = _metric_event_probability(
            metrics,
            stored,
            cond_key,
            truth,
            outcome_event,
            n,
            truth_kwargs,
            given=cond_vals)
        adjusted += p_y_given_t_adjust * p_adjust

    adjusted_key = "true_adjusted_{}_{}".format("_".join(adjustment_vars), do_name)
    metrics[adjusted_key] = adjusted
    return adjusted


def _square_query_bound_metrics(
        metrics,
        truth,
        outcome_event,
        treatment_var,
        treatment_value,
        do_name,
        n,
        truth_kwargs,
        stored):
    x_conditional = _conditional_query_candidate(
        metrics,
        truth,
        outcome_event,
        treatment_var,
        treatment_value,
        n,
        truth_kwargs,
        stored)
    z_adjusted = _adjusted_query_candidate(
        metrics,
        truth,
        outcome_event,
        treatment_var,
        treatment_value,
        ("Z",),
        do_name,
        n,
        truth_kwargs,
        stored)
    w_adjusted = _adjusted_query_candidate(
        metrics,
        truth,
        outcome_event,
        treatment_var,
        treatment_value,
        ("W",),
        do_name,
        n,
        truth_kwargs,
        stored)

    values = [x_conditional, z_adjusted, w_adjusted]
    return {
        "conditional_x": x_conditional,
        "adjusted_z": z_adjusted,
        "adjusted_w": w_adjusted,
        "lower": min(values),
        "upper": max(values),
    }


def _four_clique_query_bound_metrics(
        metrics,
        truth,
        outcome_event,
        treatment_var,
        treatment_value,
        do_name,
        n,
        truth_kwargs,
        stored):
    candidates = _chain_query_bound_metrics(
        metrics,
        truth,
        outcome_event,
        treatment_var,
        treatment_value,
        do_name,
        n,
        truth_kwargs,
        stored)
    zw_adjusted = _adjusted_query_candidate(
        metrics,
        truth,
        outcome_event,
        treatment_var,
        treatment_value,
        ("Z", "W"),
        do_name,
        n,
        truth_kwargs,
        stored)
    z_adjusted = _adjusted_query_candidate(
        metrics,
        truth,
        outcome_event,
        treatment_var,
        treatment_value,
        ("Z",),
        do_name,
        n,
        truth_kwargs,
        stored)
    w_adjusted = _adjusted_query_candidate(
        metrics,
        truth,
        outcome_event,
        treatment_var,
        treatment_value,
        ("W",),
        do_name,
        n,
        truth_kwargs,
        stored)

    values = [
        candidates["marginal"],
        candidates["conditional"],
        zw_adjusted,
        z_adjusted,
        w_adjusted,
    ]
    return {
        "marginal": candidates["marginal"],
        "conditional": candidates["conditional"],
        "adjusted_zw": zw_adjusted,
        "adjusted_z": z_adjusted,
        "adjusted_w": w_adjusted,
        "lower": min(values),
        "upper": max(values),
    }


def _barley_query_bound_metrics(
        metrics,
        truth,
        outcome_event,
        treatment_var,
        treatment_value,
        do_name,
        n,
        truth_kwargs,
        stored):
    srtprot_adjusted = _adjusted_query_candidate(
        metrics,
        truth,
        outcome_event,
        treatment_var,
        treatment_value,
        ("srtprot",),
        do_name,
        n,
        truth_kwargs,
        stored)
    sorttkv_adjusted = _adjusted_query_candidate(
        metrics,
        truth,
        outcome_event,
        treatment_var,
        treatment_value,
        ("sorttkv",),
        do_name,
        n,
        truth_kwargs,
        stored)
    srtsize_adjusted = _adjusted_query_candidate(
        metrics,
        truth,
        outcome_event,
        treatment_var,
        treatment_value,
        ("srtsize",),
        do_name,
        n,
        truth_kwargs,
        stored)

    values = [srtprot_adjusted, sorttkv_adjusted, srtsize_adjusted]
    return {
        "adjusted_srtprot": srtprot_adjusted,
        "adjusted_sorttkv": sorttkv_adjusted,
        "adjusted_srtsize": srtsize_adjusted,
        "lower": min(values),
        "upper": max(values),
    }


def scm_query_bound_metrics(
        truth,
        graph_name=None,
        outcome_var="Y",
        outcome_value=1,
        treatment_var="X",
        treatment_values=(0, 1),
        n=1000000,
        stored=None,
        truth_kwargs=None):
    metrics = dict()
    graph_name = graph_name.lower() if graph_name is not None else None
    if graph_name == "barley":
        if treatment_var == "X":
            treatment_var = "sort"
        if outcome_var == "Y":
            outcome_var = "protein"
    outcome_event = {outcome_var: outcome_value}

    for treatment_value in treatment_values:
        do_name = serialize_probability(
            outcome_event,
            do_vals={treatment_var: treatment_value})

        if graph_name == "backdoor":
            bound = _backdoor_query_bound_metrics(
                metrics,
                truth,
                outcome_event,
                treatment_var,
                treatment_value,
                do_name,
                n,
                truth_kwargs,
                stored)
        elif graph_name == "square":
            bound = _square_query_bound_metrics(
                metrics,
                truth,
                outcome_event,
                treatment_var,
                treatment_value,
                do_name,
                n,
                truth_kwargs,
                stored)
        elif graph_name == "four_clique":
            bound = _four_clique_query_bound_metrics(
                metrics,
                truth,
                outcome_event,
                treatment_var,
                treatment_value,
                do_name,
                n,
                truth_kwargs,
                stored)
        elif graph_name == "barley":
            bound = _barley_query_bound_metrics(
                metrics,
                truth,
                outcome_event,
                treatment_var,
                treatment_value,
                do_name,
                n,
                truth_kwargs,
                stored)
        else:
            bound = _chain_query_bound_metrics(
                metrics,
                truth,
                outcome_event,
                treatment_var,
                treatment_value,
                do_name,
                n,
                truth_kwargs,
                stored)

        metrics["true_lower_{}".format(do_name)] = bound["lower"]
        metrics["true_upper_{}".format(do_name)] = bound["upper"]

    return metrics


def all_metrics(
        truth,
        ncm,
        dat_dos,
        dat_sets,
        n=1000000,
        stored=None,
        query_track=None,
        include_sup=False,
        truth_kwargs=None,
        ncm_kwargs=None):
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

        m["true_KL_{}".format(name)] = kl(
            truth, ncm, n=n, do=expanded_do_dat, true_pv=true_ps[name],
            truth_kwargs=truth_kwargs, ncm_kwargs=ncm_kwargs)
        m["dat_KL_{}".format(name)] = kl(
            truth, ncm, n=n, do=expanded_do_dat, true_pv=dat_ps[name],
            truth_kwargs=truth_kwargs, ncm_kwargs=ncm_kwargs)
        m["total_true_KL"] += m["true_KL_{}".format(name)]
        m["total_dat_KL"] += m["dat_KL_{}".format(name)]
        if include_sup:
            m["true_supnorm_{}".format(name)] = supremum_norm(
                truth, ncm, n=n, do=expanded_do_dat, true_pv=true_ps[name],
                truth_kwargs=truth_kwargs, ncm_kwargs=ncm_kwargs)
            m["dat_supnorm_{}".format(name)] = supremum_norm(
                truth, ncm, n=n, do=expanded_do_dat, true_pv=dat_ps[name],
                truth_kwargs=truth_kwargs, ncm_kwargs=ncm_kwargs)
            m["total_true_supnorm"] += m["true_supnorm_{}".format(name)]
            m["total_dat_supnorm"] += m["dat_supnorm_{}".format(name)]

    if query_track is not None:
        true_q = 'true_{}'.format(serialize_query(query_track))
        m[true_q] = eval_query(
            truth, query_track, n, model_kwargs=truth_kwargs) if stored is None or true_q not in stored else stored[true_q]
        ncm_q = 'ncm_{}'.format(serialize_query(query_track))
        m[ncm_q] = eval_query(ncm, query_track, n, model_kwargs=ncm_kwargs)
        err_q = 'err_ncm_{}'.format(serialize_query(query_track))
        m[err_q] = m[true_q] - m[ncm_q]
    return m


def all_metrics_minmax(
        truth,
        ncm_min,
        ncm_max,
        dat_dos,
        dat_sets,
        n=1000000,
        stored=None,
        query_track=None,
        query_bounds=None,
        truth_kwargs=None,
        ncm_min_kwargs=None,
        ncm_max_kwargs=None):
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

        m["min_true_KL_{}".format(name)] = kl(
            truth, ncm_min, n=n, do=expanded_do_dat, true_pv=true_ps[name],
            truth_kwargs=truth_kwargs, ncm_kwargs=ncm_min_kwargs)
        m["max_true_KL_{}".format(name)] = kl(
            truth, ncm_max, n=n, do=expanded_do_dat, true_pv=true_ps[name],
            truth_kwargs=truth_kwargs, ncm_kwargs=ncm_max_kwargs)
        m["min_dat_KL_{}".format(name)] = kl(
            truth, ncm_min, n=n, do=expanded_do_dat, true_pv=dat_ps[name],
            truth_kwargs=truth_kwargs, ncm_kwargs=ncm_min_kwargs)
        m["max_dat_KL_{}".format(name)] = kl(
            truth, ncm_max, n=n, do=expanded_do_dat, true_pv=dat_ps[name],
            truth_kwargs=truth_kwargs, ncm_kwargs=ncm_max_kwargs)
        m["min_true_supnorm_{}".format(name)] = supremum_norm(
            truth, ncm_min, n=n, do=expanded_do_dat, true_pv=true_ps[name],
            truth_kwargs=truth_kwargs, ncm_kwargs=ncm_min_kwargs)
        m["max_true_supnorm_{}".format(name)] = supremum_norm(
            truth, ncm_max, n=n, do=expanded_do_dat, true_pv=true_ps[name],
            truth_kwargs=truth_kwargs, ncm_kwargs=ncm_max_kwargs)
        m["min_dat_supnorm_{}".format(name)] = supremum_norm(
            truth, ncm_min, n=n, do=expanded_do_dat, true_pv=dat_ps[name],
            truth_kwargs=truth_kwargs, ncm_kwargs=ncm_min_kwargs)
        m["max_dat_supnorm_{}".format(name)] = supremum_norm(
            truth, ncm_max, n=n, do=expanded_do_dat, true_pv=dat_ps[name],
            truth_kwargs=truth_kwargs, ncm_kwargs=ncm_max_kwargs)

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
        m[true_q] = eval_query(
            truth, query_track, n, model_kwargs=truth_kwargs) if stored is None or true_q not in stored else stored[true_q]
        min_ncm_q = 'min_ncm_{}'.format(serialize_query(query_track))
        max_ncm_q = 'max_ncm_{}'.format(serialize_query(query_track))
        m[min_ncm_q] = eval_query(ncm_min, query_track, n, model_kwargs=ncm_min_kwargs)
        m[max_ncm_q] = eval_query(ncm_max, query_track, n, model_kwargs=ncm_max_kwargs)
        min_err_q = 'min_err_ncm_{}'.format(serialize_query(query_track))
        max_err_q = 'max_err_ncm_{}'.format(serialize_query(query_track))
        m[min_err_q] = m[true_q] - m[min_ncm_q]
        m[max_err_q] = m[true_q] - m[max_ncm_q]
        minmax_gap = 'minmax_{}_gap'.format(serialize_query(query_track))
        m[minmax_gap] = m[max_ncm_q] - m[min_ncm_q]

    if query_bounds is not None:
        bound_metrics = scm_query_bound_metrics(
            truth,
            n=n,
            stored=stored,
            truth_kwargs=truth_kwargs,
            **query_bounds)
        m.update(bound_metrics)

        if query_track is not None:
            query_name = serialize_query(query_track)
            lower_key = "true_lower_{}".format(query_name)
            upper_key = "true_upper_{}".format(query_name)
            min_ncm_q = "min_ncm_{}".format(query_name)
            max_ncm_q = "max_ncm_{}".format(query_name)
            if lower_key in m and min_ncm_q in m:
                m["err_min_ncm_{}_lower_bound".format(query_name)] = m[lower_key] - m[min_ncm_q]
            if upper_key in m and max_ncm_q in m:
                m["err_max_ncm_{}_upper_bound".format(query_name)] = m[upper_key] - m[max_ncm_q]
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
