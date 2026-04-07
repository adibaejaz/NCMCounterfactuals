import argparse
import json
import re
from pathlib import Path

import torch as T

from src.metric import evaluation
from src.metric.queries import get_atomic_query


RESULT_GLOB = "**/results.json"
BOUND_QUERY_RE = re.compile(r"Y = ([01]).*do\(X = ([01])\)")


def parse_atomic_query_name(bound_query_text: str) -> str:
    match = BOUND_QUERY_RE.search(bound_query_text)
    if not match:
        raise ValueError("Unsupported bound-query format: {}".format(bound_query_text))
    y_value, x_value = match.groups()
    return "y{}_dox{}".format(y_value, x_value)


def load_dat_sets(dat_path: Path):
    try:
        return T.load(dat_path, map_location="cpu", weights_only=False)
    except TypeError:
        return T.load(dat_path, map_location="cpu")


def update_trial(trial_dir: Path):
    hyperparams_path = trial_dir / "hyperparams.json"
    dat_path = trial_dir / "dat.th"
    if not hyperparams_path.exists() or not dat_path.exists():
        return 0

    with hyperparams_path.open() as fp:
        hyperparams = json.load(fp)

    bound_query_text = hyperparams.get("bound-query")
    if bound_query_text is None:
        return 0

    graph_name = trial_dir.parent.name.lower()
    query_name = parse_atomic_query_name(bound_query_text)
    query = get_atomic_query(graph_name, query_name)

    dat_sets = load_dat_sets(dat_path)
    if len(dat_sets) == 0:
        raise ValueError("No datasets found in {}".format(dat_path))

    prob_table = evaluation.probability_table(dat=dat_sets[0])
    lower_bound, upper_bound = evaluation.atomic_query_bounds_from_probability_table(
        graph_name,
        query,
        prob_table,
    )

    hyperparams["query-bound-lower"] = str(lower_bound)
    hyperparams["query-bound-upper"] = str(upper_bound)
    with hyperparams_path.open("w") as fp:
        json.dump(hyperparams, fp)

    query_key = evaluation.serialize_query(query)
    updated_count = 0
    for results_path in sorted(trial_dir.glob("*/results.json")):
        with results_path.open() as fp:
            results = json.load(fp)
        results["lower_bound"] = lower_bound
        results["upper_bound"] = upper_bound
        max_key = "max_ncm_{}".format(query_key)
        min_key = "min_ncm_{}".format(query_key)
        if max_key in results:
            results["err_upper_bound"] = results[max_key] - upper_bound
        if min_key in results:
            results["err_lower_bound"] = results[min_key] - lower_bound
        with results_path.open("w") as fp:
            json.dump(results, fp)
        updated_count += 1

    return updated_count


def update_experiment(base_dir: Path):
    trial_dirs = sorted({results_path.parent.parent for results_path in base_dir.glob(RESULT_GLOB)})
    updated_results = 0
    for trial_dir in trial_dirs:
        updated_results += update_trial(trial_dir)
    return updated_results, len(trial_dirs)


def main():
    parser = argparse.ArgumentParser(description="Recompute saved atomic bounds in experiment outputs.")
    parser.add_argument("experiment_dirs", nargs="+", help="experiment directories to update")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    for exp in args.experiment_dirs:
        base_dir = Path(exp)
        if not base_dir.is_absolute():
            base_dir = repo_root / base_dir
        updated_results, updated_trials = update_experiment(base_dir)
        print(
            "updated {} results across {} trials in {}".format(
                updated_results,
                updated_trials,
                base_dir,
            )
        )


if __name__ == "__main__":
    main()
