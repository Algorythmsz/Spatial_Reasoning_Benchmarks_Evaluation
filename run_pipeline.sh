#!/usr/bin/env bash
# run_pipeline.sh — prepare -> infer -> score -> table, end to end.
#
# A thin wrapper around data_preparation.py / infer.py / evaluate.py / make_table.py.
# Nothing here is required: every step is one plain python command and you can run them
# by hand (see README). This just chains them, one model at a time, with the conda-env
# switch SpatialScore needs.
#
#   ./run_pipeline.sh -m qwen3vl-4b -b multihopspatial
#   ./run_pipeline.sh -m qwen3vl-4b,qwen3vl-8b -b refspatial_expand,multihopspatial
#   ./run_pipeline.sh -m qwen3vl-4b -b spatialscore --infer-env infer-env --score-env score-env
#   ./run_pipeline.sh -m qwen3vl-4b -b multihopspatial --skip-infer     # re-score existing preds
#
# Storage: export HF_HOME / BENCH_DATA_DIR / POST_CRISP_ROOT first if you don't want
# weights, data and outputs on the home disk (see README step b/c).

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODELS=""
BENCHMARKS=""
INFER_ENV=""                 # empty -> use whatever env is already active
SCORE_ENV=""                 # empty -> same env as inference (fine unless bench=spatialscore)
MAX_NEW_TOKENS=512
DO_PREPARE=1 DO_INFER=1 DO_SCORE=1 DO_TABLE=1

usage() {
    awk 'NR==1 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "${BASH_SOURCE[0]}"
    cat <<'EOF'

Options:
  -m, --models TAGS        comma-separated tags from models.yaml, or "all"   (required)
  -b, --benchmarks NAMES   comma-separated benchmark names, or "all"         (required)
      --infer-env NAME     conda env for prepare/infer (default: current env)
      --score-env NAME     conda env for scoring       (default: --infer-env)
      --max-new-tokens N   generation budget (default: 512)
      --skip-prepare | --skip-infer | --skip-score | --skip-table
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--models)       MODELS="$2"; shift 2 ;;
        -b|--benchmarks)   BENCHMARKS="$2"; shift 2 ;;
        --infer-env)       INFER_ENV="$2"; shift 2 ;;
        --score-env)       SCORE_ENV="$2"; shift 2 ;;
        --max-new-tokens)  MAX_NEW_TOKENS="$2"; shift 2 ;;
        --skip-prepare)    DO_PREPARE=0; shift ;;
        --skip-infer)      DO_INFER=0; shift ;;
        --skip-score)      DO_SCORE=0; shift ;;
        --skip-table)      DO_TABLE=0; shift ;;
        -h|--help)         usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "$MODELS"     ]] || { echo "error: -m/--models is required" >&2; usage >&2; exit 2; }
[[ -n "$BENCHMARKS" ]] || { echo "error: -b/--benchmarks is required" >&2; usage >&2; exit 2; }
: "${SCORE_ENV:=$INFER_ENV}"

# conda activate only works in a script after sourcing the hook
if [[ -n "$INFER_ENV$SCORE_ENV" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi
use_env() { [[ -n "$1" ]] && conda activate "$1"; :; }

cd "$REPO"
IFS=',' read -ra BENCH_LIST <<< "$BENCHMARKS"
IFS=',' read -ra MODEL_LIST <<< "$MODELS"
[[ "$BENCHMARKS" == "all" ]] && BENCH_LIST=(all)      # let the python entry points expand "all"

step() { echo; echo "=== [$(date +%H:%M:%S)] $* ==="; }

for bench in "${BENCH_LIST[@]}"; do
    # 1) prepare — downloads raw data + builds the ms-swift jsonl. Idempotent: re-running
    #    an already-prepared benchmark just prints "already present" and returns.
    if (( DO_PREPARE )); then
        step "prepare $bench"
        use_env "$INFER_ENV"
        python data_preparation.py "$bench"
    fi

    # 2) infer — ONE MODEL PER PROCESS on purpose. vLLM doesn't always release the GPU
    #    between models in the same process, so a multi-model run can OOM midway.
    #    Already-finished (model, benchmark) pairs are skipped by infer.py itself.
    if (( DO_INFER )); then
        use_env "$INFER_ENV"
        for model in "${MODEL_LIST[@]}"; do
            step "infer $bench / $model"
            python infer.py --benchmarks "$bench" --models "$model" \
                            --max-new-tokens "$MAX_NEW_TOKENS"
        done
    fi

    # 3) score — SpatialScore runs an LLM judge and needs the scoring env; every other
    #    benchmark scores in the inference env. Scoring is cheap, so all models at once.
    if (( DO_SCORE )); then
        step "score $bench / $MODELS"
        if [[ "$bench" == "spatialscore" || "$bench" == "all" ]]; then
            use_env "$SCORE_ENV"
        else
            use_env "$INFER_ENV"
        fi
        python evaluate.py --benchmarks "$bench" --models "$MODELS"
    fi

    # 4) table — leaderboard across every model scored so far, not just this run's.
    if (( DO_TABLE )) && [[ "$bench" != "all" ]]; then
        step "table $bench"
        python make_table.py --bench "$bench"
    fi
done

echo
echo "=== [$(date +%H:%M:%S)] pipeline done: [$BENCHMARKS] x [$MODELS] ==="
