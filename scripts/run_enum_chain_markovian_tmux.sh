#!/usr/bin/env bash
set -euo pipefail

EXP_ROOT="${1:-enum_baseline/markovian/chain}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPUS=(${GPUS:-0 1 2 3 5 6 7})
TRIALS=(${TRIALS:-0 1 2 3 4 5 6 7 8 9 10 12 41 56 69 87})
SEED_OFFSETS=(${SEED_OFFSETS:-0 1 2})
N_SAMPLES="${N_SAMPLES:-10000}"
DIM="${DIM:-1}"
LR="${LR:-4e-3}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found" >&2
  exit 1
fi

job_index=0

launch_run() {
  local session_name="$1"
  local gpu="$2"
  local trial_index="$3"
  local seed_offset="$4"

  if tmux has-session -t "$session_name" 2>/dev/null; then
    echo "[skip] tmux session exists: $session_name"
    return
  fi

  local cmd="$PYTHON_BIN -m src.enumeration_experiment ${EXP_ROOT} --graph chain --equiv-class-file dat/cg/chain_y_x_z_equiv.cg --bound-query --bound-treatment Z --bound-outcome Y --bound-treatment-value 0 --n-samples ${N_SAMPLES} --dim ${DIM} --lr ${LR} --data-bs 1000 --ncm-bs 1000 --regions 20 --gen-bs 10000 --id-reruns 1 --train-seed-offset ${seed_offset} --trial-index ${trial_index} --gpu 0"

  echo "[launch] gpu=$gpu trial=$trial_index seed_offset=$seed_offset session=$session_name"
  tmux new-session -d -s "$session_name" \
    "cd /home/adiba.ejaz/NCMCounterfactuals && source start.sh && CUDA_VISIBLE_DEVICES=$gpu $cmd; status=\$?; echo; echo '[exit status]' \$status; exec bash"
}

for seed_offset in "${SEED_OFFSETS[@]}"; do
  for trial_index in "${TRIALS[@]}"; do
    gpu="${GPUS[$((job_index % ${#GPUS[@]}))]}"
    session="enum_chain_s${seed_offset}_t${trial_index}_g${gpu}"
    launch_run "$session" "$gpu" "$trial_index" "$seed_offset"
    job_index=$((job_index + 1))
  done
done

echo
echo "Launched $job_index enum baseline jobs across GPUs: ${GPUS[*]}"
