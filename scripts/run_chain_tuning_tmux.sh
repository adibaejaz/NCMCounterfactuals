#!/usr/bin/env bash
set -euo pipefail

# Small pilot launcher for tuning the fit-vs-acyclicity tradeoff on the chain graph.
#
# Default grid:
# - trial_index in {0, 1, 2} via --n-trials 3
# - baseline lr in {1e-3, 4e-3}
# - masked:
#     notears x lr in {1e-3, 4e-3} x lambda in {0.05, 1}
#     dagma   x lr in {1e-3, 4e-3} x lambda in {0.05, 0.1} with dagma-s=2
#
# Total default tmux sessions:
# - configs: 10
# - each config runs 3 trial indices internally

EXP_ROOT="${1:-chain_tuning}"
N_SAMPLES="${N_SAMPLES:-1000}"
N_TRIALS="${N_TRIALS:-3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
LRS=(${LRS:-1e-3 4e-3})
NOTEARS_LAMBDAS=(${NOTEARS_LAMBDAS:-0.05 1 0.1})
DAGMA_LAMBDAS=(${DAGMA_LAMBDAS:-0.05 0.1})
DAGMA_S="${DAGMA_S:-2}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found" >&2
  exit 1
fi

job_index=0

launch_run() {
  local session_name="$1"
  local gpu="$2"
  local cmd="$3"

  if tmux has-session -t "$session_name" 2>/dev/null; then
    echo "[skip] tmux session exists: $session_name"
    return
  fi

  echo "[launch] gpu=$gpu session=$session_name"
  tmux new-session -d -s "$session_name" \
    "cd /home/adiba.ejaz/NCMCounterfactuals && source start.sh && CUDA_VISIBLE_DEVICES=$gpu $cmd; status=\$?; echo; echo '[exit status]' \$status; exec bash"
}

for lr in "${LRS[@]}"; do
  gpu="${GPUS[$((job_index % ${#GPUS[@]}))]}"
  lr_tag="${lr//./p}"
  session="chain_tune_base_lr${lr_tag}"
  cmd="$PYTHON_BIN -m src.main ${EXP_ROOT}/baseline_lr${lr_tag} divergence --graph chain --n-samples ${N_SAMPLES} --n-trials ${N_TRIALS} -d 1 --query-track ate --lr ${lr} --gpu 0"
  launch_run "$session" "$gpu" "$cmd"
  job_index=$((job_index + 1))

  for cycle_lambda in "${NOTEARS_LAMBDAS[@]}"; do
    gpu="${GPUS[$((job_index % ${#GPUS[@]}))]}"
    cycle_tag="${cycle_lambda/./p}"
    session="chain_tune_notears_c${cycle_tag}_lr${lr_tag}"
    cmd="$PYTHON_BIN -m src.masked_experiment ${EXP_ROOT}/notears_cycle${cycle_tag}_lr${lr_tag} --graph chain --n-samples ${N_SAMPLES} --n-trials ${N_TRIALS} -d 1 --query-track ate --lr ${lr} --gpu 0 --mask-mode gate --learn-mask --cycle-lambda ${cycle_lambda} --cycle-penalty notears --mask-init-value 0.5 --mask-fixed-zero 'Z->Y' --mask-fixed-zero 'Y->Z' --mask-l1-lambda 0"
    launch_run "$session" "$gpu" "$cmd"
    job_index=$((job_index + 1))
  done

  for cycle_lambda in "${DAGMA_LAMBDAS[@]}"; do
    gpu="${GPUS[$((job_index % ${#GPUS[@]}))]}"
    cycle_tag="${cycle_lambda/./p}"
    session="chain_tune_dagma_c${cycle_tag}_lr${lr_tag}"
    cmd="$PYTHON_BIN -m src.masked_experiment ${EXP_ROOT}/dagma_cycle${cycle_tag}_lr${lr_tag} --graph chain --n-samples ${N_SAMPLES} --n-trials ${N_TRIALS} -d 1 --query-track ate --lr ${lr} --gpu 0 --mask-mode gate --learn-mask --cycle-lambda ${cycle_lambda} --cycle-penalty dagma --dagma-s ${DAGMA_S} --mask-init-value 0.1 --mask-fixed-zero 'Z->Y' --mask-fixed-zero 'Y->Z' --mask-l1-lambda 0"
    launch_run "$session" "$gpu" "$cmd"
    job_index=$((job_index + 1))
  done
done

echo
echo "Launched $job_index tuning configs across ${#GPUS[@]} GPUs."
echo "Each config runs ${N_TRIALS} trial indices internally."
echo "Use 'tmux ls' to inspect sessions."
echo "Recommended metrics to compare: total_dat_KL, total_true_KL, dag_h, and the learned mask."
