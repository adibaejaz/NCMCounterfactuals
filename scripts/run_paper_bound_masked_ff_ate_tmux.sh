#!/usr/bin/env bash
set -euo pipefail

# Launch masked FF-NCM direct P(Y=1 | do(X=0)) bound runs on generated paper datasets.
#
# Usage:
#   bash scripts/run_paper_bound_masked_ff_ate_tmux.sh nonchain
#   bash scripts/run_paper_bound_masked_ff_ate_tmux.sh chain
#   bash scripts/run_paper_bound_masked_ff_ate_tmux.sh main3
#   bash scripts/run_paper_bound_masked_ff_ate_tmux.sh four_clique
#   bash scripts/run_paper_bound_masked_ff_ate_tmux.sh all
#
# The graph-specific dataset is selected by masked_experiment.py through:
#   --reuse-data-root out/paper_bound_datasets_adjustment_gap
#   --graph <graph>
#   --n-samples 10000 --dim 1 --trial-index / --n-trials

MODE="${1:-all}"

ROOT_DIR="${ROOT_DIR:-/home/adiba.ejaz/NCMCounterfactuals}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EXP_NAME="${EXP_NAME:-paper_bound_masked_ff_py1_dox0_alt1x1_masklr1e-2}"
SESSION_PREFIX="${SESSION_PREFIX:-pbpy1x0mlr1e2}"
GPUS=(${GPUS:-0 1 2 3})
OFFSETS=(${OFFSETS:-1 2 3})
MAX_SESSIONS_PER_GPU="${MAX_SESSIONS_PER_GPU:-2}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-8}"
N_TRIALS="${N_TRIALS:-5}"

for gpu in "${GPUS[@]}"; do
  case "$gpu" in
    0|1|2|3)
      ;;
    *)
      echo "unsupported GPU for this script: $gpu" >&2
      echo "allowed GPUs: 0 1 2 3" >&2
      exit 2
      ;;
  esac
done

if ! [[ "$MAX_SESSIONS_PER_GPU" =~ ^[0-9]+$ ]] || [[ "$MAX_SESSIONS_PER_GPU" -lt 1 ]]; then
  echo "MAX_SESSIONS_PER_GPU must be a positive integer" >&2
  exit 2
fi

if ! [[ "$CPU_THREADS_PER_JOB" =~ ^[0-9]+$ ]] || [[ "$CPU_THREADS_PER_JOB" -lt 1 ]]; then
  echo "CPU_THREADS_PER_JOB must be a positive integer" >&2
  exit 2
fi

if ! [[ "$N_TRIALS" =~ ^[0-9]+$ ]] || [[ "$N_TRIALS" -lt 1 ]]; then
  echo "N_TRIALS must be a positive integer" >&2
  exit 2
fi

case "$MODE" in
  all)
    GRAPHS=(chain backdoor square four_clique)
    ;;
  main3)
    GRAPHS=(chain backdoor square)
    ;;
  nonchain)
    GRAPHS=(backdoor square four_clique)
    ;;
  chain)
    GRAPHS=(chain)
    ;;
  four_clique)
    GRAPHS=(four_clique)
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    echo "expected one of: all, main3, nonchain, chain, four_clique" >&2
    exit 2
    ;;
esac

cd "$ROOT_DIR"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found" >&2
  exit 1
fi

fixed_edge_args() {
  local graph="$1"
  case "$graph" in
    chain)
      printf "%s" "--mask-fixed-zero 'X->Y' --mask-fixed-zero 'Y->X'"
      ;;
    square)
      printf "%s" "--mask-fixed-zero 'Y->W' --mask-fixed-zero 'Z->W' --mask-fixed-zero 'X->Y' --mask-fixed-zero 'Y->X' --mask-fixed-zero 'W->Z' --mask-fixed-one 'Z->Y' --mask-fixed-one 'W->Y' --mask-non-collider 'W,X,Z'"
      ;;
    *)
      printf "%s" ""
      ;;
  esac
}

base_cmd() {
  local graph="$1"
  local offset="$2"

  printf "%s src/masked_experiment.py %s \
    --bound-query \
    --bound-treatment X \
    --bound-outcome Y \
    --bound-outcome-value 1 \
    --bound-treatment-value 0 \
    --graph %s \
    --reuse-data-root out/paper_bound_datasets_adjustment_gap \
    --n-samples 10000 \
    --n-trials %s \
    --dim 1 \
    --lr 4e-3 \
    --theta-lr 4e-3 \
    --mask-lr 0.01 \
    --gpu 0 \
    --mask-mode multiply \
    --learn-mask \
    --cycle-lambda 0.1 \
    --cycle-penalty notears \
    --mask-init-mode constant \
    --mask-init-value 0.5 \
    %s \
    --mask-l1-lambda 0 \
    --alt-opt \
    --theta-steps-per-mask 1 \
    --mask-steps-per-theta 1 \
    --theta-only-extra-epochs 50 \
    --no-theta-only-final-query-reg \
    --max-lambda 1e-2 \
    --min-lambda 1e-4 \
    --train-seed-offset %s \
    --max-query-iters 1000" \
    "$PYTHON_BIN" "$EXP_NAME" "$graph" "$N_TRIALS" "$(fixed_edge_args "$graph")" "$offset"
}

launch_tmux() {
  local session_name="$1"
  local gpu="$2"
  local cmd="$3"

  if tmux has-session -t "$session_name" 2>/dev/null; then
    echo "[skip] tmux session exists: $session_name"
    return
  fi

  echo "[launch] gpu=$gpu session=$session_name"
  tmux new-session -d -s "$session_name" \
    "cd ${ROOT_DIR} && source start.sh && OMP_NUM_THREADS=${CPU_THREADS_PER_JOB} MKL_NUM_THREADS=${CPU_THREADS_PER_JOB} OPENBLAS_NUM_THREADS=${CPU_THREADS_PER_JOB} NUMEXPR_NUM_THREADS=${CPU_THREADS_PER_JOB} CUDA_VISIBLE_DEVICES=${gpu} ${cmd}; status=\$?; echo; echo '[exit status]' \$status; exec bash"
}

job_index=0
launched_count=0
declare -A gpu_session_counts=()
for gpu in "${GPUS[@]}"; do
  gpu_session_counts["$gpu"]=0
done

for graph in "${GRAPHS[@]}"; do
  for offset in "${OFFSETS[@]}"; do
    gpu="${GPUS[$((job_index % ${#GPUS[@]}))]}"
    session="${SESSION_PREFIX}_${graph}_s${offset}_gpu${gpu}"
    if [[ "${gpu_session_counts[$gpu]}" -ge "$MAX_SESSIONS_PER_GPU" ]]; then
      echo "[skip] gpu=$gpu reached MAX_SESSIONS_PER_GPU=$MAX_SESSIONS_PER_GPU session=$session"
    else
      cmd="$(base_cmd "$graph" "$offset")"
      launch_tmux "$session" "$gpu" "$cmd"
      gpu_session_counts["$gpu"]=$((gpu_session_counts["$gpu"] + 1))
      launched_count=$((launched_count + 1))
    fi
    job_index=$((job_index + 1))
  done
done

echo
echo "Launched up to $launched_count of $job_index tmux sessions."
echo "Experiment: $EXP_NAME"
echo "Mode: $MODE"
echo "Graphs: ${GRAPHS[*]}"
echo "Offsets: ${OFFSETS[*]}"
echo "Trials per session: $N_TRIALS"
echo "GPUs: ${GPUS[*]}"
echo "Max sessions per GPU: $MAX_SESSIONS_PER_GPU"
echo "CPU threads per new job: $CPU_THREADS_PER_JOB"
for gpu in "${GPUS[@]}"; do
  echo "GPU $gpu sessions launched: ${gpu_session_counts[$gpu]}"
done
echo "Use 'tmux ls' to inspect sessions."
