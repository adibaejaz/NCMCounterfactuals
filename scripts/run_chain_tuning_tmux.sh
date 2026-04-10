#!/usr/bin/env bash
set -euo pipefail

# Small pilot launcher for tuning the fit-vs-acyclicity tradeoff on the chain graph.
#
# This intentionally focuses on the main open question:
# how much cycle penalty can we apply before data fit degrades too much?
#
# Default grid:
# - lr in {1e-3, 4e-3}
# - masked configs:
#     gate x {0.01, 0.05, 0.1}
#
# Total default runs:
# - baseline: 2
# - masked: 2 * 5 = 10
# - total: 12

EXP_ROOT="${1:-chain_tuning}"
N_SAMPLES="${N_SAMPLES:-1000}"
TRIAL_INDEX="${TRIAL_INDEX:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
LRS=(${LRS:-1e-3 4e-3})

MASK_CONFIGS=(
  "gate 0.01"
  "gate 0.05"
  "gate 0.1"
)

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
  cmd="$PYTHON_BIN -m src.main ${EXP_ROOT}/baseline_lr${lr_tag} divergence --graph chain --n-samples ${N_SAMPLES} --n-trials 1 -d 1 --query-track ate --lr ${lr}"
  launch_run "$session" "$gpu" "$cmd"
  job_index=$((job_index + 1))

  for cfg in "${MASK_CONFIGS[@]}"; do
    read -r mask_mode cycle_lambda <<< "$cfg"
    gpu="${GPUS[$((job_index % ${#GPUS[@]}))]}"
    cycle_tag="${cycle_lambda/./p}"
    session="chain_tune_${mask_mode}_c${cycle_tag}_lr${lr_tag}"
    cmd="$PYTHON_BIN -m src.masked_experiment ${EXP_ROOT}/${mask_mode}_cycle${cycle_tag}_lr${lr_tag} --graph chain --n-samples ${N_SAMPLES} --n-trials 1 -d 1 --query-track ate --lr ${lr} --mask-mode ${mask_mode} --learn-mask --cycle-lambda ${cycle_lambda}"
    launch_run "$session" "$gpu" "$cmd"
    job_index=$((job_index + 1))
  done
done

echo
echo "Launched $job_index tuning runs across ${#GPUS[@]} GPUs."
echo "Use 'tmux ls' to inspect sessions."
echo "Recommended metrics to compare: total_dat_KL, total_true_KL, dag_h, ATE error, and the learned mask."
