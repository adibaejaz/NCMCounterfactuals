#!/usr/bin/env bash
set -euo pipefail

# Launch the chain fit experiment as one tmux session per run.
#
# Runs:
# - 5 baseline FF-NCM runs
# - 5 trials x 3 mask modes x 2 cycle penalties = 30 masked runs
# Total = 35 tmux sessions
#
# GPU assignment is round-robin over GPUs 0-7, which yields:
# - GPUs 0,1,2: 5 runs each
# - GPUs 3,4,5,6,7: 4 runs each

EXP_ROOT="${1:-chain_fit_compare}"
N_SAMPLES="${N_SAMPLES:-10000}"
N_TRIALS="${N_TRIALS:-5}"
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
PYTHON_BIN="${PYTHON_BIN:-python}"

MASK_MODES=(threshold gate multiply)
CYCLE_LAMBDAS=(0.0 0.1)

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

for trial in $(seq 0 $((N_TRIALS - 1))); do
  gpu="${GPUS[$((job_index % ${#GPUS[@]}))]}"
  session="chain_base_t${trial}"
  cmd="$PYTHON_BIN -m src.main ${EXP_ROOT}/baseline divergence --graph chain --n-samples ${N_SAMPLES} --n-trials 1 -d 1 --query-track ate"
  launch_run "$session" "$gpu" "$cmd"
  job_index=$((job_index + 1))

  for mask_mode in "${MASK_MODES[@]}"; do
    for cycle_lambda in "${CYCLE_LAMBDAS[@]}"; do
      gpu="${GPUS[$((job_index % ${#GPUS[@]}))]}"
      cycle_tag="${cycle_lambda/./p}"
      session="chain_${mask_mode}_c${cycle_tag}_t${trial}"
      cmd="$PYTHON_BIN -m src.masked_experiment ${EXP_ROOT}/masked_${mask_mode}_cycle${cycle_tag} --graph chain --n-samples ${N_SAMPLES} --n-trials 1 -d 1 --query-track ate --mask-mode ${mask_mode} --learn-mask --cycle-lambda ${cycle_lambda} --mask-fixed-zero Z->Y --mask-fixed-zero Y->Z"
      launch_run "$session" "$gpu" "$cmd"
      job_index=$((job_index + 1))
    done
  done
done

echo
echo "Launched $job_index runs across ${#GPUS[@]} GPUs."
echo "Use 'tmux ls' to inspect sessions."
