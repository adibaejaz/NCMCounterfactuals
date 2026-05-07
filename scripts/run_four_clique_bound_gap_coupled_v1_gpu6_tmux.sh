#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-plan}"

ROOT_DIR="${ROOT_DIR:-/home/adiba.ejaz/NCMCounterfactuals}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EXP_NAME="${EXP_NAME:-overnight_runs/coupling/paper_bound_masked_ff_four_clique_bound_gap_only_coupled_v1_querymask_mfit1_1x1_uniform}"
DATA_ROOT="${DATA_ROOT:-out/paper_bound_datasets_four_clique_bound_gap_only}"
SESSION_PREFIX="${SESSION_PREFIX:-fc_bg_coupled_v1}"
GPU="${GPU:-6}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-8}"
N_TRIALS="${N_TRIALS:-5}"
OFFSETS=(${OFFSETS:-1 2 3})

command_for_offset() {
  local offset="$1"

  printf "%s" "${PYTHON_BIN} src/masked_experiment.py ${EXP_NAME} \
    --bound-query \
    --bound-treatment X \
    --bound-outcome Y \
    --bound-outcome-value 1 \
    --bound-treatment-value 0 \
    --graph four_clique \
    --reuse-data-root ${DATA_ROOT} \
    --n-samples 10000 \
    --n-trials ${N_TRIALS} \
    --dim 1 \
    --lr 4e-3 \
    --theta-lr 4e-3 \
    --mask-lr 0.1 \
    --mask-fit-loss-weight 1.0 \
    --gpu 0 \
    --mask-mode multiply \
    --learn-mask \
    --cycle-lambda 0.1 \
    --cycle-penalty notears \
    --mask-init-mode uniform \
    --mask-init-value 0.5 \
    --mask-init-low 0.1 \
    --mask-init-high 0.9 \
    --mask-equiv-class-file dat/cg/four_clique_equiv.cg \
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
    --train-seed-offset ${offset} \
    --max-query-iters 1000"
}

plan() {
  echo "Four-clique bound-gap-only coupled-v1 masked FF-NCM plan"
  echo "EXP_NAME=${EXP_NAME}"
  echo "DATA_ROOT=${DATA_ROOT}"
  echo "GPU=${GPU}"
  echo "CPU_THREADS_PER_JOB=${CPU_THREADS_PER_JOB}"
  echo "N_TRIALS=${N_TRIALS}"
  echo "OFFSETS=${OFFSETS[*]}"
  echo
  local offset
  for offset in "${OFFSETS[@]}"; do
    echo "SESSION=${SESSION_PREFIX}_s${offset}_gpu${GPU}"
    command_for_offset "$offset"
    echo
    echo
  done
}

launch() {
  cd "$ROOT_DIR"
  mkdir -p logs

  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux not found" >&2
    exit 1
  fi
  if [[ ! -f dat/cg/four_clique.cg ]]; then
    echo "[missing] ${ROOT_DIR}/dat/cg/four_clique.cg" >&2
    exit 1
  fi
  if [[ ! -f dat/cg/four_clique_equiv.cg ]]; then
    echo "[missing] ${ROOT_DIR}/dat/cg/four_clique_equiv.cg" >&2
    exit 1
  fi
  if [[ ! -d "$DATA_ROOT" ]]; then
    echo "[missing] ${ROOT_DIR}/${DATA_ROOT}" >&2
    exit 1
  fi

  local offset session log cmd
  for offset in "${OFFSETS[@]}"; do
    session="${SESSION_PREFIX}_s${offset}_gpu${GPU}"
    log="logs/${session}.log"

    if tmux has-session -t "$session" 2>/dev/null; then
      echo "[skip] tmux session exists: $session"
      continue
    fi

    cmd="$(command_for_offset "$offset")"
    echo "[launch] session=${session} gpu=${GPU} log=${log}"
    tmux new-session -d -s "$session" \
      "cd ${ROOT_DIR} && source start.sh && export CUDA_VISIBLE_DEVICES=${GPU}; export OMP_NUM_THREADS=${CPU_THREADS_PER_JOB} MKL_NUM_THREADS=${CPU_THREADS_PER_JOB} OPENBLAS_NUM_THREADS=${CPU_THREADS_PER_JOB} NUMEXPR_NUM_THREADS=${CPU_THREADS_PER_JOB}; ${cmd} 2>&1 | tee ${log}; status=\${PIPESTATUS[0]}; echo; echo '[exit status]' \$status; exec bash"
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
