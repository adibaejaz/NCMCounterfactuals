#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-plan}"

ROOT_DIR="${ROOT_DIR:-/home/adiba.ejaz/NCMCounterfactuals}"
EXP_ROOT="${EXP_ROOT:-paper_enum_sachs_joint_obs_baseline}"
DATA_ROOT="${DATA_ROOT:-out/paper_bound_datasets_sachs_joint_obs}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CPU_CAP_PREFIX="${CPU_CAP_PREFIX:-/local/eb/bin/cap -m 10 -s 0 -c 5 --}"

GPU="${GPU:-5}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-5}"
SESSION="${SESSION:-enum_sachs_joint_gpu5}"
LOG="${LOG:-logs/${SESSION}.log}"

N_SAMPLES="${N_SAMPLES:-100000}"
DIM="${DIM:-1}"
LR="${LR:-4e-3}"
DATA_BS="${DATA_BS:-1000}"
NCM_BS="${NCM_BS:-1000}"
ENUM_SAMPLE_K="${ENUM_SAMPLE_K:-3}"
ENUM_SAMPLE_SEED="${ENUM_SAMPLE_SEED:-0}"
ID_RERUNS="${ID_RERUNS:-3}"
MAX_EPOCHS="${MAX_EPOCHS:-500}"
EVAL_N="${EVAL_N:-1000000}"
TRIALS=(${TRIALS:-0 1 2 3 4})
BOUND_DO_JSON='{"PKA":0,"PKC":0}'

trial_args() {
  local args=""
  local trial
  for trial in "${TRIALS[@]}"; do
    args="${args} --trial-index ${trial}"
  done
  printf "%s" "$args"
}

command_for_sachs() {
  printf "%s" "${CPU_CAP_PREFIX} ${PYTHON_BIN} -m src.enumeration_experiment ${EXP_ROOT} \
    --graph sachs \
    --equiv-class-file dat/cg/sachs_equiv.cg \
    --reuse-data-root ${DATA_ROOT} \
    --bound-query \
    --bound-outcome Akt \
    --bound-outcome-value 1 \
    --bound-do-json '${BOUND_DO_JSON}' \
    --n-samples ${N_SAMPLES} \
    --dim ${DIM} \
    --lr ${LR} \
    --data-bs ${DATA_BS} \
    --ncm-bs ${NCM_BS} \
    --enum-sample-k ${ENUM_SAMPLE_K} \
    --enum-sample-seed ${ENUM_SAMPLE_SEED} \
    --id-reruns ${ID_RERUNS} \
    --max-epochs ${MAX_EPOCHS} \
    --eval-n ${EVAL_N} \
    --gpu 0$(trial_args)"
}

plan() {
  echo "Sachs joint-observation enumeration baseline plan"
  echo "EXP_ROOT=${EXP_ROOT}"
  echo "DATA_ROOT=${DATA_ROOT}"
  echo "GPU=${GPU}"
  echo "CPU_CAP_PREFIX=${CPU_CAP_PREFIX}"
  echo "CPU_THREADS_PER_JOB=${CPU_THREADS_PER_JOB}"
  echo "N_SAMPLES=${N_SAMPLES} DIM=${DIM}"
  echo "ENUM_SAMPLE_K=${ENUM_SAMPLE_K} ENUM_SAMPLE_SEED=${ENUM_SAMPLE_SEED}"
  echo "ID_RERUNS=${ID_RERUNS}"
  echo "MAX_EPOCHS=${MAX_EPOCHS} DATA_BS=${DATA_BS} NCM_BS=${NCM_BS} EVAL_N=${EVAL_N}"
  echo "TRIALS=${TRIALS[*]}"
  echo "SESSION=${SESSION}"
  echo "LOG=${LOG}"
  echo
  command_for_sachs
  echo
}

launch() {
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux not found" >&2
    exit 1
  fi

  cd "$ROOT_DIR"
  mkdir -p logs

  if [[ ! -f dat/cg/sachs.cg ]]; then
    echo "[missing] ${ROOT_DIR}/dat/cg/sachs.cg" >&2
    exit 1
  fi
  if [[ ! -f dat/cg/sachs_equiv.cg ]]; then
    echo "[missing] ${ROOT_DIR}/dat/cg/sachs_equiv.cg" >&2
    exit 1
  fi
  if [[ ! -d "$DATA_ROOT" ]]; then
    echo "[missing] ${ROOT_DIR}/${DATA_ROOT}" >&2
    exit 1
  fi
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[skip] tmux session exists: $SESSION"
    return
  fi

  local cmd
  cmd="$(command_for_sachs)"

  echo "[launch] session=${SESSION} gpu=${GPU} log=${LOG}"
  tmux new-session -d -s "$SESSION" \
    "cd ${ROOT_DIR} && source start.sh && export CUDA_VISIBLE_DEVICES=${GPU}; export OMP_NUM_THREADS=${CPU_THREADS_PER_JOB} MKL_NUM_THREADS=${CPU_THREADS_PER_JOB} OPENBLAS_NUM_THREADS=${CPU_THREADS_PER_JOB} NUMEXPR_NUM_THREADS=${CPU_THREADS_PER_JOB}; ${cmd} 2>&1 | tee ${LOG}; status=\${PIPESTATUS[0]}; echo; echo '[exit status]' \$status; exec bash"
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
