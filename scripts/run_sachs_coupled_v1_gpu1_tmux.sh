#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-plan}"

ROOT_DIR="${ROOT_DIR:-/home/adiba.ejaz/NCMCounterfactuals}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EXP_NAME="${EXP_NAME:-overnight_runs/coupling/paper_bound_masked_ff_overnight_coupled_v1_querymask_mfit1_1x1_uniform}"
DATA_ROOT="${DATA_ROOT:-out/paper_bound_datasets_sachs_joint_obs}"
SESSION="${SESSION:-sachs_coupled_v1_gpu6_trial0_rerun1}"
LOG="${LOG:-logs/${SESSION}.log}"

GPU="${GPU:-1}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-8}"
N_SAMPLES="${N_SAMPLES:-100000}"
DIM="${DIM:-1}"
LR="${LR:-4e-3}"
THETA_LR="${THETA_LR:-4e-3}"
MASK_LR="${MASK_LR:-0.1}"
DATA_BS="${DATA_BS:-1000}"
NCM_BS="${NCM_BS:-1000}"
ID_RERUNS="${ID_RERUNS:-1}"
TRIALS=(${TRIALS:-0})
MAX_QUERY_ITERS="${MAX_QUERY_ITERS:-1000}"
THETA_ONLY_EXTRA_EPOCHS="${THETA_ONLY_EXTRA_EPOCHS:-50}"
SELECTION_QUERY_LAMBDA="${SELECTION_QUERY_LAMBDA:-1e-4}"
TRAIN_SEED_OFFSET="${TRAIN_SEED_OFFSET:-1}"
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
  printf "%s" "${PYTHON_BIN} src/masked_experiment.py ${EXP_NAME} \
    --bound-query \
    --bound-outcome Akt \
    --bound-outcome-value 1 \
    --bound-do-json '${BOUND_DO_JSON}' \
    --graph sachs \
    --reuse-data-root ${DATA_ROOT} \
    --n-samples ${N_SAMPLES} \
    --dim ${DIM} \
    --lr ${LR} \
    --theta-lr ${THETA_LR} \
    --mask-lr ${MASK_LR} \
    --mask-fit-loss-weight 1.0 \
    --gpu 0 \
    --data-bs ${DATA_BS} \
    --ncm-bs ${NCM_BS} \
    --mask-mode multiply \
    --learn-mask \
    --cycle-lambda 0.1 \
    --cycle-penalty notears \
    --mask-init-mode uniform \
    --mask-init-value 0.5 \
    --mask-init-low 0.1 \
    --mask-init-high 0.9 \
    --mask-equiv-class-file dat/cg/sachs_equiv.cg \
    --mask-l1-lambda 0 \
    --alt-opt \
    --theta-steps-per-mask 1 \
    --mask-steps-per-theta 1 \
    --theta-only-extra-epochs ${THETA_ONLY_EXTRA_EPOCHS} \
    --no-theta-only-final-query-reg \
    --max-lambda 1e-3 \
    --min-lambda 1e-5 \
    --selection-query-lambda ${SELECTION_QUERY_LAMBDA} \
    --query-update-target mask \
    --train-seed-offset ${TRAIN_SEED_OFFSET} \
    --id-reruns ${ID_RERUNS} \
    --max-query-iters ${MAX_QUERY_ITERS}$(trial_args)"
}

plan() {
  echo "Sachs coupled-v1 masked FF-NCM plan"
  echo "EXP_NAME=${EXP_NAME}"
  echo "DATA_ROOT=${DATA_ROOT}"
  echo "GPU=${GPU}"
  echo "CPU_THREADS_PER_JOB=${CPU_THREADS_PER_JOB}"
  echo "N_SAMPLES=${N_SAMPLES} DIM=${DIM}"
  echo "ID_RERUNS=${ID_RERUNS}"
  echo "TRIALS=${TRIALS[*]}"
  echo "TRAIN_SEED_OFFSET=${TRAIN_SEED_OFFSET}"
  echo "MAX_QUERY_ITERS=${MAX_QUERY_ITERS}"
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
