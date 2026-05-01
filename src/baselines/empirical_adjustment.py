import math

import torch as T

from src.scm.scm import check_equal


def _n_rows(dat):
    if not dat:
        raise ValueError("observational data is empty")
    return len(dat[next(iter(dat))])


def _as_cpu_tensor(value):
    if T.is_tensor(value):
        return value.detach().cpu()
    return T.as_tensor(value)


def _row_event(var, row):
    row = row.detach().cpu()
    if row.ndim == 0:
        row = row.reshape(1)
    return (var, tuple(int(x) for x in row.tolist()))


def _observed_adjustment_events(dat, adjustment_vars):
    adjustment_vars = tuple(adjustment_vars)
    if not adjustment_vars:
        return [(dict(), T.ones(_n_rows(dat), dtype=T.bool))]

    keys = []
    masks = {}
    for i in range(_n_rows(dat)):
        key = tuple(_row_event(var, dat[var][i]) for var in adjustment_vars)
        if key not in masks:
            keys.append(key)
            masks[key] = T.zeros(_n_rows(dat), dtype=T.bool)
        masks[key][i] = True

    events = []
    for key in keys:
        event = {
            var: _as_cpu_tensor(values)
            for var, values in key
        }
        events.append((event, masks[key]))
    return events


def empirical_adjustment(
        dat,
        outcome_var,
        outcome_value,
        treatment_var,
        treatment_value,
        adjustment_vars):
    """Estimate sum_z P(Y=y | X=x, Z=z) P(Z=z) from observed samples only."""
    total_n = _n_rows(dat)
    if outcome_var not in dat:
        raise ValueError("outcome variable {} is not in data".format(outcome_var))
    if treatment_var not in dat:
        raise ValueError("treatment variable {} is not in data".format(treatment_var))
    missing_adjustments = sorted(set(adjustment_vars) - set(dat))
    if missing_adjustments:
        raise ValueError("adjustment variables are not in data: {}".format(missing_adjustments))

    treatment_mask = check_equal(dat[treatment_var], treatment_value).detach().cpu()
    outcome_mask = check_equal(dat[outcome_var], outcome_value).detach().cpu()

    estimate = 0.0
    skipped_terms = []
    used_terms = []

    for event, z_mask in _observed_adjustment_events(dat, adjustment_vars):
        z_count = int(z_mask.long().sum().item())
        z_prob = z_count / total_n
        denom_mask = z_mask & treatment_mask
        denom = int(denom_mask.long().sum().item())
        if denom == 0:
            skipped_terms.append({
                "event": _jsonable_event(event),
                "p_z": z_prob,
                "z_count": z_count,
                "reason": "zero_treatment_count",
            })
            continue
        numerator = int((denom_mask & outcome_mask).long().sum().item())
        conditional = numerator / denom
        contribution = conditional * z_prob
        estimate += contribution
        used_terms.append({
            "event": _jsonable_event(event),
            "p_z": z_prob,
            "z_count": z_count,
            "treatment_count": denom,
            "outcome_treatment_count": numerator,
            "p_y_given_x_z": conditional,
            "contribution": contribution,
        })

    skipped_mass = sum(term["p_z"] for term in skipped_terms)
    if not used_terms:
        estimate = float("nan")

    return {
        "kind": "adjustment",
        "estimate": estimate,
        "adjustment_vars": list(adjustment_vars),
        "used_terms": used_terms,
        "skipped_terms": skipped_terms,
        "n_used_terms": len(used_terms),
        "n_skipped_terms": len(skipped_terms),
        "skipped_probability_mass": skipped_mass,
        "has_nan_terms": bool(skipped_terms),
    }


def empirical_marginal(dat, outcome_var, outcome_value):
    total_n = _n_rows(dat)
    if outcome_var not in dat:
        raise ValueError("outcome variable {} is not in data".format(outcome_var))
    outcome_mask = check_equal(dat[outcome_var], outcome_value).detach().cpu()
    count = int(outcome_mask.long().sum().item())
    return {
        "kind": "marginal_outcome",
        "estimate": count / total_n,
        "outcome_var": outcome_var,
        "outcome_value": outcome_value,
        "outcome_count": count,
        "n": total_n,
        "has_nan_terms": False,
        "n_skipped_terms": 0,
        "skipped_terms": [],
        "skipped_probability_mass": 0.0,
    }


def _jsonable_value(value):
    if T.is_tensor(value):
        value = value.detach().cpu().tolist()
    if isinstance(value, tuple):
        return [_jsonable_value(v) for v in value]
    if isinstance(value, list):
        return [_jsonable_value(v) for v in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        if isinstance(value, float) and math.isnan(value):
            return "nan"
        return value
    return str(value)


def _jsonable_event(event):
    return {
        var: _jsonable_value(value)
        for var, value in sorted(event.items())
    }
