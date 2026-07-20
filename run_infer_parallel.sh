#!/usr/bin/env bash
# run_infer_parallel.sh — skeleton: infer several models in parallel, one model per GPU.
#
# Why one model per process: passing many models to a single `infer.py` call loads them
# sequentially in one process and can OOM (vllm doesn't always release the GPU between
# models). A fresh process per model is the only guaranteed GPU reclaim — so we fan out,
# pinning each model to a GPU via CUDA_VISIBLE_DEVICES and running at most one model per GPU.
#
# Edit the three lists below, then: bash run_infer_parallel.sh
set -euo pipefail

# ── edit these ────────────────────────────────────────────────────────────────
GPUS=(0 1)                                   # GPU ids to use (one model runs per GPU at a time)
MODELS=(qwen3vl-4b qwen3.5-4b qwen3vl-8b)    # model tags from models.yaml
BENCHMARKS=spatialscore                      # comma-list, or 'all'
EXTRA_ARGS=""                                # e.g. "--max-new-tokens 1024"
LOGDIR=logs                                  # per-model stdout/stderr go here
# ────────────────────────────────────────────────────────────────────────────

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p "$LOGDIR"

declare -A GPU_PID=()   # gpu id -> pid currently running on it (unset = free)

# Wait until at least one GPU is free, reaping finished jobs; echoes a free GPU id.
free_gpu() {
  while :; do
    for g in "${GPUS[@]}"; do
      local pid="${GPU_PID[$g]:-}"
      if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
        unset "GPU_PID[$g]"
        echo "$g"; return
      fi
    done
    wait -n   # block until any background job exits, then re-scan (bash 4.3+)
  done
}

for model in "${MODELS[@]}"; do
  gpu="$(free_gpu)"
  echo "[launch] $model on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" \
    python infer.py --benchmarks "$BENCHMARKS" --models "$model" $EXTRA_ARGS \
    >"$LOGDIR/infer_${model}.log" 2>&1 &
  GPU_PID[$gpu]=$!
done

wait   # let the last in-flight models finish
echo "[done] all models finished — see $LOGDIR/ for logs"
