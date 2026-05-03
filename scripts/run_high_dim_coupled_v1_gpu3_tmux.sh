#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-launch}"

ROOT_DIR="${ROOT_DIR:-/home/adiba.ejaz/NCMCounterfactuals}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EXP_NAME="${EXP_NAME:-overnight_runs/coupling/paper_bound_masked_ff_overnight_coupled_v1_querymask_mfit1_1x1_uniform}"
SESSION_PREFIX="${SESSION_PREFIX:-hd_coupled_v1_gpu3}"
GPU="${GPU:-3}"
SEEDS=(${SEEDS:-1 2 3})
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-8}"
DATA_ROOT="${DATA_ROOT:-out/high_dim_small_graphs}"
N_SAMPLES="${N_SAMPLES:-100000}"
N_TRIALS="${N_TRIALS:-5}"
DIM="${DIM:-8}"
U_SIZE="${U_SIZE:-8}"
SEED_GPU_PREFIX="${SEED_GPU_PREFIX:-hd_coupled_v1_u${U_SIZE}}"

common_args() {
  local graph="$1"
  local seed="$2"

  printf "%s" "src/masked_experiment.py ${EXP_NAME} \
    --bound-query \
    --bound-treatment X \
    --bound-outcome Y \
    --bound-outcome-value 1 \
    --bound-treatment-value 0 \
    --graph ${graph} \
    --reuse-data-root ${DATA_ROOT} \
    --n-samples ${N_SAMPLES} \
    --n-trials ${N_TRIALS} \
    --dim ${DIM} \
    --u-size ${U_SIZE} \
    --lr 4e-3 \
    --theta-lr 4e-3 \
    --mask-lr 0.1 \
    --mask-fit-loss-weight 1.0 \
    --gpu 0 \
    --data-bs 1000 \
    --ncm-bs 1000 \
    --mask-mode multiply \
    --learn-mask \
    --cycle-lambda 0.1 \
    --cycle-penalty notears \
    --mask-init-mode uniform \
    --mask-init-value 0.5 \
    --mask-init-low 0.1 \
    --mask-init-high 0.9 \
    --mask-equiv-class-file dat/cg/${graph}_equiv.cg \
    --mask-l1-lambda 0 \
    --alt-opt \
    --theta-steps-per-mask 1 \
    --mask-steps-per-theta 1 \
    --theta-only-extra-epochs 50 \
    --no-theta-only-final-query-reg \
    --max-lambda 1e-2 \
    --min-lambda 1e-4 \
    --selection-query-lambda 1e-4 \
    --query-update-target mask \
    --train-seed-offset ${seed} \
    --max-query-iters 1000"
}

graph_extra_args() {
  local graph="$1"
  case "$graph" in
    square)
      printf "%s" " --var-dim W=8 --var-dim Z=8"
      ;;
    *)
      printf "%s" ""
      ;;
  esac
}

run_graph_sequence() {
  local graph="$1"

  cd "$ROOT_DIR"
  source start.sh
  export CUDA_VISIBLE_DEVICES="$GPU"
  export OMP_NUM_THREADS="$CPU_THREADS_PER_JOB"
  export MKL_NUM_THREADS="$CPU_THREADS_PER_JOB"
  export OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_JOB"
  export NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_JOB"

  echo "Worker graph=${graph} gpu=${GPU} seeds=${SEEDS[*]}"
  echo "Started at $(date)"

  for seed in "${SEEDS[@]}"; do
    echo
    echo "========== graph=${graph} seed=${seed} =========="
    ${PYTHON_BIN} $(common_args "$graph" "$seed") $(graph_extra_args "$graph")
  done

  echo
  echo "Completed at $(date)"
}

run_single_graph_seed() {
  local graph="$1"
  local seed="$2"
  local gpu="$3"

  cd "$ROOT_DIR"
  source start.sh
  export CUDA_VISIBLE_DEVICES="$gpu"
  export OMP_NUM_THREADS="$CPU_THREADS_PER_JOB"
  export MKL_NUM_THREADS="$CPU_THREADS_PER_JOB"
  export OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_JOB"
  export NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_JOB"

  echo "Worker graph=${graph} seed=${seed} gpu=${gpu} trials=0..$((N_TRIALS - 1))"
  echo "Started at $(date)"
  ${PYTHON_BIN} $(common_args "$graph" "$seed") $(graph_extra_args "$graph")
  echo "Completed graph=${graph} seed=${seed} at $(date)"
}

run_seed_session() {
  local seed="$1"

  cd "$ROOT_DIR"
  mkdir -p logs

  echo "Seed worker seed=${seed}"
  echo "Graph/GPU map: chain=0 square=1 backdoor=2"
  echo "Started at $(date)"

  bash "$0" graph-seed-worker chain "$seed" 0 \
    2>&1 | tee "logs/${SEED_GPU_PREFIX}_seed${seed}_chain_gpu0.log" &
  local chain_pid=$!

  bash "$0" graph-seed-worker square "$seed" 1 \
    2>&1 | tee "logs/${SEED_GPU_PREFIX}_seed${seed}_square_gpu1.log" &
  local square_pid=$!

  bash "$0" graph-seed-worker backdoor "$seed" 2 \
    2>&1 | tee "logs/${SEED_GPU_PREFIX}_seed${seed}_backdoor_gpu2.log" &
  local backdoor_pid=$!

  local status=0
  wait "$chain_pid" || status=$?
  wait "$square_pid" || status=$?
  wait "$backdoor_pid" || status=$?

  echo "Completed seed=${seed} status=${status} at $(date)"
  return "$status"
}

launch_graph_session() {
  local graph="$1"
  local session="${SESSION_PREFIX}_${graph}"
  local log="logs/${session}.log"

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "[skip] tmux session exists: $session"
    return
  fi

  echo "[launch] session=${session} graph=${graph} log=${log}"
  tmux new-session -d -s "$session" \
    "bash '$0' worker '$graph' 2>&1 | tee '$log'; status=\${PIPESTATUS[0]}; echo; echo '[exit status]' \$status; exec bash"
}

launch_seed_session() {
  local seed="$1"
  local session="${SEED_GPU_PREFIX}_seed${seed}"
  local log="logs/${session}.log"

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "[skip] tmux session exists: $session"
    return
  fi

  echo "[launch] session=${session} seed=${seed} log=${log}"
  tmux new-session -d -s "$session" \
    "bash '$0' seed-worker '$seed' 2>&1 | tee '$log'; status=\${PIPESTATUS[0]}; echo; echo '[exit status]' \$status; exec bash"
}

smoke() {
  cd "$ROOT_DIR"
  source start.sh
  export CUDA_VISIBLE_DEVICES="$GPU"
  export OMP_NUM_THREADS="$CPU_THREADS_PER_JOB"
  export MKL_NUM_THREADS="$CPU_THREADS_PER_JOB"
  export OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_JOB"
  export NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_JOB"

  local old_exp_name="$EXP_NAME"
  EXP_NAME="smoke_high_dim_coupled_v1_gpu3"
  ${PYTHON_BIN} $(common_args chain 1) --trial-index 0 --max-query-iters 1 --theta-only-extra-epochs 0
  EXP_NAME="$old_exp_name"
}

case "$MODE" in
  smoke)
    smoke
    ;;
  launch)
    cd "$ROOT_DIR"
    launch_graph_session chain
    launch_graph_session square
    ;;
  launch-seeds)
    cd "$ROOT_DIR"
    for seed in "${SEEDS[@]}"; do
      launch_seed_session "$seed"
    done
    ;;
  worker)
    run_graph_sequence "$2"
    ;;
  seed-worker)
    run_seed_session "$2"
    ;;
  graph-seed-worker)
    run_single_graph_seed "$2" "$3" "$4"
    ;;
  *)
    echo "usage: $0 [smoke|launch|launch-seeds|worker GRAPH|seed-worker SEED|graph-seed-worker GRAPH SEED GPU]" >&2
    exit 2
    ;;
esac
