import glob
import hashlib
import json
import os
import random
import shutil
import time

import numpy as np
import pytorch_lightning as pl
import torch as T

from src.ds.equivalence_class import enumerate_dags_in_class, read_equivalence_class
from src.metric import evaluation
from .base_runner import BaseRunner
from .data_setup import build_data_bundle


class EnumerationNCMRunner(BaseRunner):
    def __init__(self, pipeline, dat_model, ncm_model):
        super().__init__(pipeline, dat_model, ncm_model)

    @staticmethod
    def _sync_cuda():
        if T.cuda.is_available():
            T.cuda.synchronize()

    def create_trainer(self, directory, gpu=None, hyperparams=None):
        if hyperparams is None:
            hyperparams = dict()
        checkpoint = pl.callbacks.ModelCheckpoint(
            dirpath=f"{directory}/checkpoints/",
            monitor="train_loss",
            save_last=True,
        )
        return pl.Trainer(
            callbacks=[
                checkpoint,
                pl.callbacks.EarlyStopping(
                    monitor="train_loss",
                    patience=int(hyperparams.get("early-stop-patience", self.pipeline.patience)),
                    min_delta=float(hyperparams.get("early-stop-min-delta", self.pipeline.min_delta)),
                    check_on_train_epoch_end=True,
                ),
            ],
            max_epochs=int(hyperparams.get("max-epochs", self.pipeline.max_epochs)),
            accumulate_grad_batches=1,
            logger=pl.loggers.TensorBoardLogger(f"{directory}/logs/"),
            log_every_n_steps=1,
            terminate_on_nan=True,
            gpus=gpu,
        ), checkpoint

    def _candidate_seed(self, key, hyperparams, dag_index, dag, rerun_index):
        dag_payload = json.dumps(
            {"vertices": list(dag.v), "directed_edges": list(dag.de)},
            sort_keys=True,
        )
        hp_payload = self._stable_hyperparams_payload(hyperparams)
        payload = "enum|{}|rerun={}|dag_index={}|dag={}|hp={}".format(
            key, rerun_index, dag_index, dag_payload, hp_payload)
        seed = self._hash_to_seed(payload)
        return (seed + int(hyperparams.get("train-seed-offset", 0))) & 0xffffffff

    def _load_existing_candidate(self, candidate_dir):
        results_path = os.path.join(candidate_dir, "results.json")
        if not os.path.isfile(results_path):
            return None
        with open(results_path) as file:
            return json.load(file)

    def _sample_indexed_dags(self, indexed_dags, hyperparams, key, rerun_index=0):
        sample_k = hyperparams.get("enum-sample-k")
        if sample_k is None:
            return indexed_dags, None
        sample_k = int(sample_k)
        if sample_k < 1:
            raise ValueError("enum-sample-k must be positive")
        if sample_k >= len(indexed_dags):
            return indexed_dags, None

        base_seed = hyperparams.get("enum-sample-seed")
        if base_seed is None:
            seed_payload = "enum-sample|{}|rerun={}|{}".format(
                key, rerun_index, self._stable_hyperparams_payload(hyperparams))
        elif int(rerun_index) == 0:
            seed_payload = "enum-sample|base={}|{}".format(int(base_seed), key)
        else:
            seed_payload = "enum-sample|base={}|{}|rerun={}".format(
                int(base_seed), key, rerun_index)
        effective_seed = self._hash_to_seed(seed_payload)
        rng = random.Random(effective_seed)
        sampled_positions = sorted(rng.sample(range(len(indexed_dags)), sample_k))
        return [indexed_dags[position] for position in sampled_positions], effective_seed

    def _legacy_single_rerun_dir(self, exp_name, cg_file, n, dim, trial_index, hyperparams):
        legacy_hyperparams = dict(hyperparams)
        legacy_hyperparams["id-reruns"] = 1
        legacy_run_key = self.get_run_key(cg_file, n, dim, trial_index, legacy_hyperparams)
        return "out/{}/{}".format(exp_name, legacy_run_key)

    def _copy_legacy_path(self, src, dst):
        if not os.path.exists(src) or os.path.exists(dst):
            return
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    def _migrate_legacy_single_rerun(self, exp_name, cg_file, n, dim, trial_index, hyperparams, out_dir):
        if int(hyperparams.get("id-reruns", 1)) <= 1:
            return False

        legacy_dir = self._legacy_single_rerun_dir(exp_name, cg_file, n, dim, trial_index, hyperparams)
        legacy_results = os.path.join(legacy_dir, "results.json")
        if legacy_dir == out_dir or not os.path.isfile(legacy_results):
            return False

        rerun_dir = os.path.join(out_dir, "0")
        rerun_results = os.path.join(rerun_dir, "results.json")
        if os.path.isfile(rerun_results):
            return False

        print("[migrate] legacy single-rerun result {} -> {}".format(legacy_dir, rerun_dir))
        os.makedirs(rerun_dir, exist_ok=True)
        self._copy_legacy_path(legacy_results, rerun_results)
        self._copy_legacy_path(os.path.join(legacy_dir, "dag_results.json"), os.path.join(rerun_dir, "dag_results.json"))
        self._copy_legacy_path(os.path.join(legacy_dir, "class_summary.json"), os.path.join(rerun_dir, "class_summary.json"))
        self._copy_legacy_path(os.path.join(legacy_dir, "dags"), os.path.join(rerun_dir, "dags"))
        self._copy_legacy_path(os.path.join(legacy_dir, "dat.th"), os.path.join(out_dir, "dat.th"))
        return True

    def _aggregate_results(self, dag_results, truth, query_track, query_bounds=None, stored_metrics=None):
        summary = {
            "num_dags": len(dag_results),
        }
        if stored_metrics is None:
            stored_metrics = dict()

        if query_track is not None:
            eval_n = int(stored_metrics.get("eval-n", 1000000))
            query_name = evaluation.serialize_query(query_track)
            true_key = "true_{}".format(query_name)
            true_value = stored_metrics.get(true_key)
            if true_value is None:
                true_value = evaluation.eval_query(truth, query_track, eval_n)
            summary[true_key] = true_value

            value_key = "ncm_{}".format(query_name)
            dag_results = sorted(dag_results, key=lambda row: row[value_key])
            min_row = dag_results[0]
            max_row = dag_results[-1]
            summary["enum_min_{}".format(value_key)] = min_row[value_key]
            summary["enum_max_{}".format(value_key)] = max_row[value_key]
            summary["enum_{}_gap".format(query_name)] = max_row[value_key] - min_row[value_key]
            summary["enum_min_err_ncm_{}".format(query_name)] = true_value - min_row[value_key]
            summary["enum_max_err_ncm_{}".format(query_name)] = true_value - max_row[value_key]
            summary["enum_min_dag_index"] = min_row["dag_index"]
            summary["enum_max_dag_index"] = max_row["dag_index"]
            summary["enum_min_graph_file"] = min_row["graph_file"]
            summary["enum_max_graph_file"] = max_row["graph_file"]
            summary["enum_min_total_true_KL"] = min_row.get("total_true_KL")
            summary["enum_max_total_true_KL"] = max_row.get("total_true_KL")
            summary["enum_min_total_dat_KL"] = min_row.get("total_dat_KL")
            summary["enum_max_total_dat_KL"] = max_row.get("total_dat_KL")

        if query_bounds is not None:
            eval_n = int(stored_metrics.get("eval-n", 1000000))
            summary.update(evaluation.scm_query_bound_metrics(
                truth,
                n=eval_n,
                stored=stored_metrics,
                **query_bounds,
            ))

        return summary

    def run(self, exp_name, cg_file, n, dim, trial_index, hyperparams=None, gpu=None, data_bundle=None,
            lockinfo=os.environ.get("SLURM_JOB_ID", ""), verbose=False):
        if hyperparams is None:
            hyperparams = dict()
        eval_n = int(hyperparams.get("eval-n", 1000000))

        eq_file = hyperparams.get("equiv-class-file")
        if not eq_file:
            raise ValueError("hyperparams['equiv-class-file'] is required")

        key = self.get_key(cg_file, n, dim, trial_index)
        run_key = self.get_run_key(cg_file, n, dim, trial_index, hyperparams)
        out_dir = "out/{}/{}".format(exp_name, run_key)
        id_reruns = int(hyperparams.get("id-reruns", 1))

        with self.lock(f"{out_dir}/lock", lockinfo) as acquired_lock:
            if not acquired_lock:
                print("[locked]", out_dir)
                return

            try:
                self._migrate_legacy_single_rerun(
                    exp_name, cg_file, n, dim, trial_index, hyperparams, out_dir)
                final_path = os.path.join(out_dir, "results.json")
                if os.path.isfile(final_path):
                    print("[done]", out_dir)
                    return

                print("[running]", out_dir)
                total_start = time.perf_counter()

                data_seed = self.get_data_seed(key)
                print("Data Key:", key)
                print("Run Key:", run_key)
                print("Data Seed:", data_seed)

                data_setup_start = time.perf_counter()
                if data_bundle is None:
                    T.manual_seed(data_seed)
                    T.cuda.manual_seed_all(data_seed)
                    np.random.seed(data_seed)
                    random.seed(data_seed)
                    data_bundle = build_data_bundle(self.dat_model, cg_file, n, dim, hyperparams, data_seed)
                data_setup_wall_seconds = time.perf_counter() - data_setup_start

                graph_generation_start = time.perf_counter()
                class_spec = read_equivalence_class(eq_file)
                if set(class_spec.vertices) != set(data_bundle.cg.v):
                    raise ValueError("equivalence class vertices must match the data graph vertices")

                dags = enumerate_dags_in_class(class_spec, max_dags=hyperparams.get("max-enum-dags"))
                if not dags:
                    raise ValueError("equivalence class enumeration produced no DAGs")
                all_indexed_dags = list(enumerate(dags))
                graph_generation_wall_seconds = time.perf_counter() - graph_generation_start

                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, "hyperparams.json"), "w") as file:
                    json.dump({k: str(v) for (k, v) in hyperparams.items()}, file)
                with open(os.path.join(out_dir, "class_summary.json"), "w") as file:
                    json.dump({
                        "equiv_class_file": eq_file,
                        "num_vertices": len(class_spec.vertices),
                        "num_directed_edges": len(class_spec.directed_edges),
                        "num_undirected_edges": len(class_spec.undirected_edges),
                        "num_enumerated_dags": len(dags),
                        "enum_sample_k": hyperparams.get("enum-sample-k"),
                        "enum_sample_seed": hyperparams.get("enum-sample-seed"),
                        "sampled_per_rerun": True,
                        "graph_generation_wall_seconds": graph_generation_wall_seconds,
                    }, file)

                if gpu is None:
                    gpu = int(T.cuda.is_available())

                rerun_summaries = []
                for rerun_index in range(id_reruns):
                    rerun_dir = out_dir if id_reruns == 1 else os.path.join(out_dir, str(rerun_index))
                    rerun_final_path = os.path.join(rerun_dir, "results.json")
                    if os.path.isfile(rerun_final_path):
                        with open(rerun_final_path) as file:
                            summary = json.load(file)
                        summary["rerun_index"] = rerun_index
                        rerun_summaries.append(summary)
                        continue

                    dag_results = []
                    rerun_start = time.perf_counter()
                    indexed_dags, effective_sample_seed = self._sample_indexed_dags(
                        all_indexed_dags, hyperparams, key, rerun_index=rerun_index)
                    rerun_class_summary = {
                        "num_sampled_dags": len(indexed_dags),
                        "sampled_dag_indices": [dag_index for (dag_index, _) in indexed_dags],
                        "enum_sample_k": hyperparams.get("enum-sample-k"),
                        "enum_sample_seed": hyperparams.get("enum-sample-seed"),
                        "enum_effective_sample_seed": effective_sample_seed,
                        "rerun_index": rerun_index,
                    }
                    os.makedirs(rerun_dir, exist_ok=True)
                    with open(os.path.join(rerun_dir, "class_summary.json"), "w") as file:
                        json.dump(rerun_class_summary, file)
                    for dag_index, dag in indexed_dags:
                        candidate_dir = os.path.join(rerun_dir, "dags", "{:04d}".format(dag_index))
                        existing = self._load_existing_candidate(candidate_dir)
                        if existing is not None:
                            dag_results.append(existing)
                            continue

                        candidate_start = time.perf_counter()
                        os.makedirs(candidate_dir, exist_ok=True)
                        graph_file = os.path.join(candidate_dir, "graph.cg")
                        dag.save(graph_file)

                        train_seed = self._candidate_seed(key, hyperparams, dag_index, dag, rerun_index)
                        print("Rerun {} DAG {} train seed: {}".format(rerun_index, dag_index, train_seed))
                        T.manual_seed(train_seed)
                        T.cuda.manual_seed_all(train_seed)
                        np.random.seed(train_seed)
                        random.seed(train_seed)

                        model = self.pipeline(
                            data_bundle.dat_m,
                            hyperparams["do-var-list"],
                            data_bundle.dat_sets,
                            dag,
                            dim,
                            hyperparams=hyperparams,
                            ncm_model=self.ncm_model,
                        )

                        stored_metrics = dict(data_bundle.stored_metrics)
                        stored_metrics["eval-n"] = eval_n
                        self._sync_cuda()
                        initial_eval_start = time.perf_counter()
                        start_metrics = evaluation.all_metrics(
                            model.generator,
                            model.ncm,
                            hyperparams["do-var-list"],
                            data_bundle.dat_sets,
                            n=eval_n,
                            stored=stored_metrics,
                            query_track=hyperparams["eval-query"],
                        )
                        self._sync_cuda()
                        initial_eval_wall_seconds = time.perf_counter() - initial_eval_start
                        true_q = "true_{}".format(evaluation.serialize_query(hyperparams["eval-query"]))
                        stored_metrics[true_q] = start_metrics[true_q]
                        model.update_metrics(stored_metrics)

                        trainer, checkpoint = self.create_trainer(candidate_dir, gpu, hyperparams=hyperparams)
                        self._sync_cuda()
                        train_start = time.perf_counter()
                        trainer.fit(model)
                        self._sync_cuda()
                        ckpt = T.load(checkpoint.best_model_path)
                        model.load_state_dict(ckpt["state_dict"])
                        train_wall_seconds = time.perf_counter() - train_start

                        self._sync_cuda()
                        final_eval_start = time.perf_counter()
                        results = evaluation.all_metrics(
                            model.generator,
                            model.ncm,
                            hyperparams["do-var-list"],
                            data_bundle.dat_sets,
                            n=eval_n,
                            query_track=hyperparams["eval-query"],
                        )
                        self._sync_cuda()
                        final_eval_wall_seconds = time.perf_counter() - final_eval_start
                        results["dag_index"] = dag_index
                        results["rerun_index"] = rerun_index
                        results["train_seed"] = train_seed
                        results["graph_file"] = graph_file
                        results["graph_edges"] = list(dag.de)
                        results["initial_eval_wall_seconds"] = initial_eval_wall_seconds
                        results["train_wall_seconds"] = train_wall_seconds
                        results["final_eval_wall_seconds"] = final_eval_wall_seconds
                        results["dag_total_wall_seconds"] = time.perf_counter() - candidate_start
                        if verbose:
                            print(results)

                        with open(os.path.join(candidate_dir, "results.json"), "w") as file:
                            json.dump(results, file)
                        T.save(model.state_dict(), os.path.join(candidate_dir, "best.th"))
                        dag_results.append(results)

                    summary = self._aggregate_results(
                        dag_results,
                        truth=data_bundle.dat_m,
                        query_track=hyperparams["eval-query"],
                        query_bounds=hyperparams.get("query-bound-spec"),
                        stored_metrics={**data_bundle.stored_metrics, "eval-n": eval_n},
                    )
                    summary["rerun_index"] = rerun_index
                    summary.update(rerun_class_summary)
                    summary["data_setup_wall_seconds"] = data_setup_wall_seconds
                    summary["enum_graph_generation_wall_seconds"] = graph_generation_wall_seconds
                    summary["enum_initial_eval_wall_seconds"] = sum(
                        float(row.get("initial_eval_wall_seconds", 0.0)) for row in dag_results)
                    summary["enum_training_wall_seconds"] = sum(
                        float(row.get("train_wall_seconds", 0.0)) for row in dag_results)
                    summary["enum_final_eval_wall_seconds"] = sum(
                        float(row.get("final_eval_wall_seconds", 0.0)) for row in dag_results)
                    summary["enum_candidate_wall_seconds"] = sum(
                        float(row.get("dag_total_wall_seconds", 0.0)) for row in dag_results)
                    summary["enum_rerun_wall_seconds"] = time.perf_counter() - rerun_start
                    summary["enum_total_wall_seconds"] = time.perf_counter() - total_start
                    with open(os.path.join(rerun_dir, "dag_results.json"), "w") as file:
                        json.dump(dag_results, file)
                    with open(rerun_final_path, "w") as file:
                        json.dump(summary, file)
                    rerun_summaries.append(summary)

                with open(final_path, "w") as file:
                    json.dump(rerun_summaries if id_reruns > 1 else rerun_summaries[0], file)
                T.save(data_bundle.dat_sets, os.path.join(out_dir, "dat.th"))
                return rerun_summaries if id_reruns > 1 else rerun_summaries[0]
            except Exception:
                err_dir = out_dir.replace("out/", "err/").rsplit("-", 1)[0]
                err_index = len(glob.glob(err_dir + "/*"))
                err_dir += "/{}".format(err_index)
                os.makedirs(err_dir.rsplit("/", 1)[0], exist_ok=True)
                shutil.move(out_dir, err_dir)
                print(f"moved {out_dir} to {err_dir}")
                raise
