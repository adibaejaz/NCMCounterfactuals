#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/adiba.ejaz/NCMCounterfactuals}"
EXP_ROOT="${EXP_ROOT:-paper_enum_baseline}"
DATA_ROOT="${DATA_ROOT:-out/paper_bound_datasets_adjustment_gap}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

GPU="${GPU:-0}"
N_TRIALS="${N_TRIALS:-5}"
N_SAMPLES="${N_SAMPLES:-10000}"
DIM="${DIM:-1}"
LR="${LR:-4e-3}"
DATA_BS="${DATA_BS:-1000}"
NCM_BS="${NCM_BS:-1000}"
ENUM_SAMPLE_K="${ENUM_SAMPLE_K:-3}"
ENUM_SAMPLE_SEED="${ENUM_SAMPLE_SEED:-0}"
ID_RERUNS="${ID_RERUNS:-3}"
MAX_EPOCHS="${MAX_EPOCHS:-500}"
EVAL_N="${EVAL_N:-1000000}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found" >&2
  exit 1
fi

launch_graph() {
  local graph="$1"
  local session_name="enum_k${ENUM_SAMPLE_K}_max${MAX_EPOCHS}_${graph}_gpu${GPU}"
  local equiv_file="dat/cg/${graph}_equiv.cg"
  local exp_name="${EXP_ROOT}"

  if [[ ! -f "${ROOT_DIR}/${equiv_file}" ]]; then
    echo "[missing] ${ROOT_DIR}/${equiv_file}" >&2
    return 1
  fi

  if tmux has-session -t "${session_name}" 2>/dev/null; then
    echo "[skip] tmux session exists: ${session_name}"
    return
  fi

  local cmd
  cmd="$PYTHON_BIN -m src.enumeration_experiment ${exp_name} \
    --graph ${graph} \
    --equiv-class-file ${equiv_file} \
    --reuse-data-root ${DATA_ROOT} \
    --bound-query \
    --bound-treatment X \
    --bound-outcome Y \
    --bound-outcome-value 1 \
    --bound-treatment-value 0 \
    --n-samples ${N_SAMPLES} \
    --n-trials ${N_TRIALS} \
    --dim ${DIM} \
    --lr ${LR} \
    --data-bs ${DATA_BS} \
    --ncm-bs ${NCM_BS} \
    --enum-sample-k ${ENUM_SAMPLE_K} \
    --enum-sample-seed ${ENUM_SAMPLE_SEED} \
    --id-reruns ${ID_RERUNS} \
    --max-epochs ${MAX_EPOCHS} \
    --eval-n ${EVAL_N} \
    --gpu 0"

  echo "[launch] graph=${graph} gpu=${GPU} trials=${N_TRIALS} sampled_dags=${ENUM_SAMPLE_K} reruns=${ID_RERUNS} max_epochs=${MAX_EPOCHS} session=${session_name}"
  tmux new-session -d -s "${session_name}" \
    "cd ${ROOT_DIR} && source start.sh && export CUDA_VISIBLE_DEVICES=${GPU}; ${cmd}; status=\$?; echo; echo '[exit status]' \$status; exec bash"
}

launch_graph chain
launch_graph backdoor

echo
echo "Launched enum baseline sessions on GPU ${GPU}:"
echo "  enum_k${ENUM_SAMPLE_K}_max${MAX_EPOCHS}_chain_gpu${GPU}"
echo "  enum_k${ENUM_SAMPLE_K}_max${MAX_EPOCHS}_backdoor_gpu${GPU}"
