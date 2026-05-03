#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-plan}"

ROOT_DIR="${ROOT_DIR:-/home/adiba.ejaz/NCMCounterfactuals}"
EXP_ROOT="${EXP_ROOT:-paper_enum_high_dim_baseline}"
DATA_ROOT="${DATA_ROOT:-out/high_dim_small_graphs}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CPU_CAP_PREFIX="${CPU_CAP_PREFIX:-/local/eb/bin/cap -c 25 --}"

GRAPHS=(${GRAPHS:-chain backdoor square four_clique})
GPUS=(${GPUS:-0 1 2 3})
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-8}"

N_SAMPLES="${N_SAMPLES:-100000}"
DIM="${DIM:-8}"
U_SIZE="${U_SIZE:-8}"
LR="${LR:-4e-3}"
DATA_BS="${DATA_BS:-1000}"
NCM_BS="${NCM_BS:-1000}"
ENUM_SAMPLE_K="${ENUM_SAMPLE_K:-3}"
ENUM_SAMPLE_SEED="${ENUM_SAMPLE_SEED:-0}"
ID_RERUNS="${ID_RERUNS:-3}"
MAX_EPOCHS="${MAX_EPOCHS:-500}"
EVAL_N="${EVAL_N:-1000000}"

trials_for_graph() {
  local graph="$1"
  case "$graph" in
    four_clique)
      printf "%s" "${FOUR_CLIQUE_TRIALS:-0 1}"
      ;;
    *)
      printf "%s" "${TRIALS:-0 1 2 3 4}"
      ;;
  esac
}

extra_args_for_graph() {
  local graph="$1"
  case "$graph" in
    square|four_clique)
      printf "%s" " --var-dim W=8 --var-dim Z=8"
      ;;
    *)
      printf "%s" ""
      ;;
  esac
}

trial_args_for_graph() {
  local graph="$1"
  local args=""
  local trial
  for trial in $(trials_for_graph "$graph"); do
    args="${args} --trial-index ${trial}"
  done
  printf "%s" "$args"
}

command_for_graph() {
  local graph="$1"
  local equiv_file="dat/cg/${graph}_equiv.cg"

  printf "%s" "${CPU_CAP_PREFIX} ${PYTHON_BIN} -m src.enumeration_experiment ${EXP_ROOT} \
    --graph ${graph} \
    --equiv-class-file ${equiv_file} \
    --reuse-data-root ${DATA_ROOT} \
    --bound-query \
    --bound-treatment X \
    --bound-outcome Y \
    --bound-outcome-value 1 \
    --bound-treatment-value 0 \
    --n-samples ${N_SAMPLES} \
    --dim ${DIM} \
    --u-size ${U_SIZE} \
    --lr ${LR} \
    --data-bs ${DATA_BS} \
    --ncm-bs ${NCM_BS} \
    --enum-sample-k ${ENUM_SAMPLE_K} \
    --enum-sample-seed ${ENUM_SAMPLE_SEED} \
    --id-reruns ${ID_RERUNS} \
    --max-epochs ${MAX_EPOCHS} \
    --eval-n ${EVAL_N} \
    --gpu 0$(trial_args_for_graph "$graph")$(extra_args_for_graph "$graph")"
}

gpu_for_index() {
  local index="$1"
  printf "%s" "${GPUS[$((index % ${#GPUS[@]}))]}"
}

session_for_graph() {
  local graph="$1"
  local gpu="$2"
  printf "enum_hd_k%s_max%s_%s_gpu%s" "${ENUM_SAMPLE_K}" "${MAX_EPOCHS}" "${graph}" "${gpu}"
}

plan() {
  local index=0
  local graph gpu session

  echo "High-dimensional enumeration baseline plan"
  echo "EXP_ROOT=${EXP_ROOT}"
  echo "DATA_ROOT=${DATA_ROOT}"
  echo "N_SAMPLES=${N_SAMPLES} DIM=${DIM} U_SIZE=${U_SIZE}"
  echo "CPU_CAP_PREFIX=${CPU_CAP_PREFIX}"
  echo "ENUM_SAMPLE_K=${ENUM_SAMPLE_K} ENUM_SAMPLE_SEED=${ENUM_SAMPLE_SEED}"
  echo "ID_RERUNS=${ID_RERUNS}"
  echo "MAX_EPOCHS=${MAX_EPOCHS} DATA_BS=${DATA_BS} NCM_BS=${NCM_BS} EVAL_N=${EVAL_N}"
  echo

  for graph in "${GRAPHS[@]}"; do
    gpu="$(gpu_for_index "$index")"
    session="$(session_for_graph "$graph" "$gpu")"
    echo "----- ${graph} -----"
    echo "gpu=${gpu} session=${session} trials=$(trials_for_graph "$graph")"
    command_for_graph "$graph"
    echo
    echo
    index=$((index + 1))
  done
}

launch() {
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux not found" >&2
    exit 1
  fi

  cd "$ROOT_DIR"
  mkdir -p logs

  local index=0
  local graph gpu session log cmd equiv_file
  for graph in "${GRAPHS[@]}"; do
    equiv_file="dat/cg/${graph}_equiv.cg"
    if [[ ! -f "$equiv_file" ]]; then
      echo "[missing] ${ROOT_DIR}/${equiv_file}" >&2
      exit 1
    fi

    gpu="$(gpu_for_index "$index")"
    session="$(session_for_graph "$graph" "$gpu")"
    log="logs/${session}.log"
    cmd="$(command_for_graph "$graph")"

    if tmux has-session -t "$session" 2>/dev/null; then
      echo "[skip] tmux session exists: $session"
      index=$((index + 1))
      continue
    fi

    echo "[launch] graph=${graph} gpu=${gpu} session=${session} log=${log}"
    tmux new-session -d -s "$session" \
      "cd ${ROOT_DIR} && source start.sh && export CUDA_VISIBLE_DEVICES=${gpu}; export OMP_NUM_THREADS=${CPU_THREADS_PER_JOB} MKL_NUM_THREADS=${CPU_THREADS_PER_JOB} OPENBLAS_NUM_THREADS=${CPU_THREADS_PER_JOB} NUMEXPR_NUM_THREADS=${CPU_THREADS_PER_JOB}; ${cmd} 2>&1 | tee ${log}; status=\${PIPESTATUS[0]}; echo; echo '[exit status]' \$status; exec bash"
    index=$((index + 1))
  done
}

case "$MODE" in
  plan)
    plan
    ;;
  launch)
    launch
    ;;
  *)
    echo "usage: $0 [plan|launch]" >&2
    exit 2
    ;;
esac
