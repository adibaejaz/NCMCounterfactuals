#!/usr/bin/env bash
set -euo pipefail

# Launch the overnight masked FF-NCM matrix.
#
# This script creates one tmux session per (coupling group, graph, GPU/offset).
# Each tmux session runs its variants sequentially, so variants for the same
# graph/seed/GPU are never concurrent. Chain sessions then continue with square;
# backdoor sessions then continue with four_clique.

MODE="${1:-launch}"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

ROOT_DIR="${ROOT_DIR:-/home/adiba.ejaz/NCMCounterfactuals}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EXP_NAME_PREFIX="${EXP_NAME_PREFIX:-paper_bound_masked_ff_overnight}"
SESSION_PREFIX="${SESSION_PREFIX:-pbovernight}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-8}"
N_TRIALS="${N_TRIALS:-5}"
FOLLOWUP_FOUR_CLIQUE_N_TRIALS="${FOLLOWUP_FOUR_CLIQUE_N_TRIALS:-4}"
MASK_LR="${MASK_LR:-0.1}"
SELECTION_QUERY_LAMBDA="${SELECTION_QUERY_LAMBDA:-1e-4}"
MAX_QUERY_ITERS="${MAX_QUERY_ITERS:-1000}"
THETA_ONLY_EXTRA_EPOCHS="${THETA_ONLY_EXTRA_EPOCHS:-50}"
DRY_RUN="${DRY_RUN:-0}"

COUPLED_GPUS=(${COUPLED_GPUS:-0 1 2})
NOCOUPLED_GPUS=(${NOCOUPLED_GPUS:-5 6 7})
OFFSETS=(${OFFSETS:-1 2 3})
GRAPHS=(${GRAPHS:-chain backdoor})

if [[ "${#COUPLED_GPUS[@]}" -ne "${#OFFSETS[@]}" ]]; then
  echo "COUPLED_GPUS and OFFSETS must have the same length" >&2
  exit 2
fi

if [[ "${#NOCOUPLED_GPUS[@]}" -ne "${#OFFSETS[@]}" ]]; then
  echo "NOCOUPLED_GPUS and OFFSETS must have the same length" >&2
  exit 2
fi

extra_mask_args() {
  local graph="$1"
  local coupling="$2"
  local -n out_args="$3"

  if [[ "$graph" == "chain" ]]; then
    out_args+=(--mask-fixed-zero "X->Y" --mask-fixed-zero "Y->X")
    if [[ "$coupling" != "coupled" ]]; then
      out_args+=(--mask-non-collider "X,Z,Y")
    fi
  fi

  if [[ "$graph" == "square" && "$coupling" != "coupled" ]]; then
    out_args+=(
      --mask-fixed-zero "Y->W"
      --mask-fixed-zero "Z->W"
      --mask-fixed-zero "X->Y"
      --mask-fixed-zero "Y->X"
      --mask-fixed-zero "W->Z"
      --mask-fixed-one "Z->Y"
      --mask-fixed-one "W->Y"
      --mask-non-collider "W,X,Z"
    )
  fi

  if [[ "$coupling" == "coupled" ]]; then
    out_args+=(--mask-equiv-class-file "dat/cg/${graph}_equiv.cg")
  fi
}

run_variant() {
  local graph="$1"
  local offset="$2"
  local coupling="$3"
  local n_trials="$4"
  local variant="$5"
  local mask_fit_loss_weight="$6"
  local theta_steps_per_mask="$7"
  local mask_steps_per_theta="$8"
  local mask_init_mode="$9"
  local mask_init_value="${10}"
  local mask_init_low="${11}"
  local mask_init_high="${12}"

  local exp_name="${EXP_NAME_PREFIX}_${coupling}_${variant}"
  local mask_args=()
  extra_mask_args "$graph" "$coupling" mask_args

  echo
  echo "========== ${exp_name} graph=${graph} offset=${offset} =========="
  echo "mask_fit_loss_weight=${mask_fit_loss_weight}"
  echo "theta/mask steps=${theta_steps_per_mask}:${mask_steps_per_theta}"
  echo "mask init=${mask_init_mode} value=${mask_init_value} low=${mask_init_low} high=${mask_init_high}"
  echo

  "$PYTHON_BIN" src/masked_experiment.py "$exp_name" \
    --bound-query \
    --bound-treatment X \
    --bound-outcome Y \
    --bound-outcome-value 1 \
    --bound-treatment-value 0 \
    --graph "$graph" \
    --reuse-data-root out/paper_bound_datasets_adjustment_gap \
    --n-samples 10000 \
    --n-trials "$n_trials" \
    --dim 1 \
    --lr 4e-3 \
    --theta-lr 4e-3 \
    --mask-lr "$MASK_LR" \
    --mask-fit-loss-weight "$mask_fit_loss_weight" \
    --gpu 0 \
    --mask-mode multiply \
    --learn-mask \
    --cycle-lambda 0.1 \
    --cycle-penalty notears \
    --mask-init-mode "$mask_init_mode" \
    --mask-init-value "$mask_init_value" \
    --mask-init-low "$mask_init_low" \
    --mask-init-high "$mask_init_high" \
    "${mask_args[@]}" \
    --mask-l1-lambda 0 \
    --alt-opt \
    --theta-steps-per-mask "$theta_steps_per_mask" \
    --mask-steps-per-theta "$mask_steps_per_theta" \
    --theta-only-extra-epochs "$THETA_ONLY_EXTRA_EPOCHS" \
    --no-theta-only-final-query-reg \
    --max-lambda 1e-2 \
    --min-lambda 1e-4 \
    --selection-query-lambda "$SELECTION_QUERY_LAMBDA" \
    --query-update-target mask \
    --train-seed-offset "$offset" \
    --max-query-iters "$MAX_QUERY_ITERS"
}

run_variant_sequence() {
  local graph="$1"
  local offset="$2"
  local coupling="$3"
  local n_trials="$4"

  run_variant "$graph" "$offset" "$coupling" "$n_trials" "v1_querymask_mfit1_1x1_uniform" 1.0 1 1 uniform 0.5 0.1 0.9
  run_variant "$graph" "$offset" "$coupling" "$n_trials" "v2_querymask_mfit0_1x1_uniform" 0.0 1 1 uniform 0.5 0.1 0.9

  if [[ "$coupling" == "coupled" ]]; then
    run_variant "$graph" "$offset" "$coupling" "$n_trials" "v3_querymask_mfit0_5x1_uniform" 0.0 5 1 uniform 0.5 0.1 0.9
    run_variant "$graph" "$offset" "$coupling" "$n_trials" "v4_querymask_mfit0_1x1_const05" 0.0 1 1 constant 0.5 0.1 0.9
  else
    run_variant "$graph" "$offset" "$coupling" "$n_trials" "v3_querymask_mfit0_1x1_const05" 0.0 1 1 constant 0.5 0.1 0.9
  fi
}

followup_graph_for() {
  local graph="$1"
  case "$graph" in
    chain)
      printf "%s" "square"
      ;;
    backdoor)
      printf "%s" "four_clique"
      ;;
    *)
      printf "%s" ""
      ;;
  esac
}

trial_count_for() {
  local graph="$1"
  if [[ "$graph" == "four_clique" ]]; then
    printf "%s" "$FOLLOWUP_FOUR_CLIQUE_N_TRIALS"
  else
    printf "%s" "$N_TRIALS"
  fi
}

worker() {
  local graph="$1"
  local gpu="$2"
  local offset="$3"
  local coupling="$4"
  local followup_graph

  cd "$ROOT_DIR"
  source start.sh
  export OMP_NUM_THREADS="$CPU_THREADS_PER_JOB"
  export MKL_NUM_THREADS="$CPU_THREADS_PER_JOB"
  export OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_JOB"
  export NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_JOB"
  export CUDA_VISIBLE_DEVICES="$gpu"

  echo "Worker graph=${graph} gpu=${gpu} offset=${offset} coupling=${coupling}"
  echo "Started at $(date)"

  run_variant_sequence "$graph" "$offset" "$coupling" "$(trial_count_for "$graph")"

  followup_graph="$(followup_graph_for "$graph")"
  if [[ -n "$followup_graph" ]]; then
    echo
    echo "Continuing session with follow-up graph=${followup_graph}"
    run_variant_sequence "$followup_graph" "$offset" "$coupling" "$(trial_count_for "$followup_graph")"
  fi

  echo
  echo "Completed at $(date)"
}

launch_session() {
  local graph="$1"
  local gpu="$2"
  local offset="$3"
  local coupling="$4"
  local session="${SESSION_PREFIX}_${coupling}_${graph}_s${offset}_gpu${gpu}"

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "[skip] tmux session exists: $session"
    return
  fi

  echo "[launch] session=$session gpu=$gpu graph=$graph offset=$offset coupling=$coupling"
  if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
    return
  fi

  tmux new-session -d -s "$session" \
    "bash '$SCRIPT_PATH' worker '$graph' '$gpu' '$offset' '$coupling'; status=\$?; echo; echo '[exit status]' \$status; exec bash"
}

launch_group() {
  local coupling="$1"
  shift
  local gpus=("$@")

  local i
  for i in "${!gpus[@]}"; do
    local gpu="${gpus[$i]}"
    local offset="${OFFSETS[$i]}"
    local graph
    for graph in "${GRAPHS[@]}"; do
      launch_session "$graph" "$gpu" "$offset" "$coupling"
    done
  done
}

if [[ "$MODE" == "worker" ]]; then
  if [[ "$#" -ne 5 ]]; then
    echo "usage: $0 worker GRAPH GPU OFFSET coupled|nocoupled" >&2
    exit 2
  fi
  worker "$2" "$3" "$4" "$5"
  exit
fi

if [[ "$MODE" != "launch" ]]; then
  echo "usage: $0 [launch]" >&2
  exit 2
fi

cd "$ROOT_DIR"
if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found" >&2
  exit 1
fi

launch_group coupled "${COUPLED_GPUS[@]}"
launch_group nocoupled "${NOCOUPLED_GPUS[@]}"

echo
echo "Launch complete."
echo "Session prefix: $SESSION_PREFIX"
echo "Experiment prefix: $EXP_NAME_PREFIX"
echo "Coupled GPUs: ${COUPLED_GPUS[*]}"
echo "No-coupling GPUs: ${NOCOUPLED_GPUS[*]}"
echo "Offsets: ${OFFSETS[*]}"
echo "Graphs: ${GRAPHS[*]}"
echo "N trials: $N_TRIALS"
echo "Four-clique follow-up trials: $FOLLOWUP_FOUR_CLIQUE_N_TRIALS"
echo "Mask LR: $MASK_LR"
echo "Selection query lambda: $SELECTION_QUERY_LAMBDA"
