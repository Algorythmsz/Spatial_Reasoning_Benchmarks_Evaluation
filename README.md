# Spatial Reasoning Benchmark harness

Run multimodal (Qwen-VL etc.) models on spatial-reasoning benchmarks and score them,
in three separate steps:

| step | script | enviornment | what it does |
|---|---|---|---|
| 1. prepare | `data_preparation.py` | inference env | download raw data (HF) → build a model-agnostic ms-swift jsonl |
| 2. infer | `infer.py` | inference env | `swift infer` each model over that jsonl → predictions + a `done.flag` |
| 3. score | `evaluate.py` | inference env (scoring env for SpatialScore) | predictions → run the scorer → `metrics.json` |
- 2 environments are needed: the inference env, plus a scoring env for SpatialScore's LLM judge.
- **Benchmarks:** `spatialscore`, `multihopspatial`, `refspatial_expand` (in `benchmarks/`).
  *(`spatialscore` and `refspatial_expand` are wired end-to-end. `multihopspatial` runs
  inference but its scorer is not implemented yet — reshape/score raise NotImplementedError.)*
- **Models:** listed in `models.yaml` — edit that file to add/remove models.

---

## ⚙️ Setup - for one time

### a. Conda environments

The **inference** env covers most of the pipeline. A separate **scoring** env is needed
only to score **SpatialScore** (its LLM-judge stage). Everything else — including scoring
`multihopspatial`/`refspatial_expand` — runs in the inference env.
**Name them whatever you like** - just activate the right one:

| env (any name) | used by | must contain |
|---|---|---|
| inference | steps 1 & 2, **and** scoring `multihopspatial` / `refspatial_expand` | `ms-swift`, `vllm`, `huggingface_hub`, `datasets` |
| scoring | step 3 for **SpatialScore only** | `vllm` + judge LLM (`openai/gpt-oss-20b`), `tqdm`, and the vendored scorer's deps: `torch`, `torchvision`, `matplotlib`, `numpy`, `pillow` |

Per-env dependency lists live in `requirements/`.

#### How to build an inference env

```bash
# 1. fresh env
conda create -n <inference-env> python=3.11 -y
conda activate <inference-env>

# 2. install (torch/transformers come via ms-swift + vllm)
pip install -r requirements/infer.txt

# 3. verify
swift infer --help >/dev/null && echo "swift OK"
python -c "import vllm; print('vllm', vllm.__version__)"

# 4. lock the exact versions you ended up with
pip freeze > requirements/infer.lock.txt
```

#### How to build a scoring env

Python 3.10 + CUDA 12.8. `requirements/score.txt` pins the working set
(vllm 0.11 + torch 2.8+cu128 + the vendored scorer's deps) and includes the cu128 index.
You only need this env to score SpatialScore (see note above).

```bash
# 1. fresh env
conda create -n <scoring-env> python=3.10 -y
conda activate <scoring-env>

# 2. install (torch/torchvision come from the cu128 index in the file)
pip install -r requirements/score.txt

# 3. verify
python -c "import vllm, torch, torchvision, matplotlib, openai_harmony; print('torch', torch.__version__)"

# 4. lock
pip freeze > requirements/score.lock.txt
```

### b. Designate where to cache the model weights from Hugging Face (Optional)

`models.yaml` uses HF repo ids. Point HF at where you want them cached:

```bash
export HF_HOME=<hf-cache-dir>   # download destination for model weights
```

Put this in your shell profile (`~/.bashrc`) or the env's activation hook
(`$CONDA_PREFIX/etc/conda/activate.d/`) so you don't retype it.

### c. Designate where to download/store data/outputs (Optional)

By default benchmark data and outputs land **inside the repo** (home disk). SpatialScore
alone is ~15.8 GB — if home is tight, point these at a big disk:

```bash
export BENCH_DATA_DIR=<where-to-download-the-data>/post_crisp/data   # raw + preprocessed benchmark data
export POST_CRISP_ROOT=<where-to-store-the-results>/post_crisp       # cache / preds / results
```


---

## 📁 Prepare data

Downloads each benchmark's raw data (into `benchmarks/data/<name>/`) and builds the
ms-swift input jsonl. 

```bash
conda activate <inference-env>
python data_preparation.py spatialscore     # or: multihopspatial | refspatial_expand | all
```

---

## 🤖 Run inference

Runs `swift infer` for each model over the prepared jsonl. Predictions go to
`preds/<model-tag>/<benchmark>.jsonl`, plus a `done.flag` marking a clean finish.
Per-model image budget (`min/max_pixels`) and `enable_thinking` from `models.yaml`
are injected automatically. We disabled Qwen3.5 models' thinking mode by default (You can change at the `models.yaml`). Already-finished (model, benchmark) pairs are skipped.

```bash
conda activate <inference-env>
python infer.py --benchmarks spatialscore --models qwen3vl-4b,qwen3.5-4b
```

Options: `--benchmarks all` (every benchmark), `--models all` (every model in `models.yaml`), `--max-new-tokens N` (default 512).

Passing multiple models in one call loads them sequentially in the same process, which can
OOM — vllm doesn't always fully release the GPU between models. Prefer **one model per run**
(a fresh process is the only guaranteed reclaim). To run several at once, launch each as its
own job in parallel.

---

## 📈 Scoring

**SpatialScore needs the scoring env** (LLM judge); the other benchmarks score in the
inference env. Activate the right one, then run — scoring reshapes the predictions and runs
the scorer; results land in
`results/<model-tag>/<benchmark>/` (`all_results.json`, `summary_report.json`, `metrics.json`).

```bash
conda activate <scoring-env>        # SpatialScore; use the inference env for the other two
python evaluate.py --benchmarks spatialscore --models qwen3vl-4b,qwen3.5-4b
```

SpatialScore's scorer lives at `benchmarks/scorers/spatialscore/evaluate_results.py` — a
local copy of the official SpatialScore code, vendored here so scoring is self-contained.

Optional:

```bash
export SS_NO_LLM=1     # rule-only: skip the Stage-2 judge LLM (no GPU)
```
Other optional overrides: `SS_LLM_PATH` (judge, default `openai/gpt-oss-20b`),
`SS_SCORER` (a different scorer checkout), `SS_TP_SIZE` / `SS_GPU_MEM` (judge vllm knobs).

---

## 📊 Make a table for results

`make_table.py` scans `results/<model-tag>/<benchmark>/metrics.json` and prints an
accuracy leaderboard (sorted by overall). Scored models appear; ones without a
`metrics.json` (OOM'd or failed) are listed as skipped.

It reads the **same** `POST_CRISP_ROOT` / `RESULTS_DIR` env vars as the rest of the
pipeline (Step c), so point it at wherever your results live. If you redirected outputs
off the home disk, either prefix each run or `export` it once for the session:

```bash
export POST_CRISP_ROOT=<where-you-stored-the-results>/post_crisp        # same value as Step c; skip if already exported this session
python make_table.py --bench spatialscore                              # overall accuracy, all scored models
python make_table.py --bench spatialscore --breakdown category         # add per-category columns
python make_table.py --bench spatialscore --csv spatialscore.csv       # also write a CSV
```

`--breakdown` accepts `category`, `task`, `sub_task`, or `source_dataset`.

---

## Full example from scratch (qwen3vl-4b + qwen3.5-4b on SpatialScore)

```bash
# a) build the envs (once, ever — see Step a for details/notes)
conda create -n infer-env python=3.11 -y
conda activate infer-env
pip install -r requirements/infer.txt            # ms-swift + vllm inference stack

conda create -n score-env python=3.10 -y         # only needed to score SpatialScore
conda activate score-env
pip install -r requirements/score.txt            # judge LLM + cu128 torch/torchvision/matplotlib

# b/c) env vars (optional — all have defaults; put in ~/.bashrc to skip retyping)
export HF_HOME=<hf-cache-dir>                                       # model cache; omit for ~/.cache/huggingface
export BENCH_DATA_DIR=<data-disk>/post_crisp/data                   # benchmark data; omit to keep in-repo
export POST_CRISP_ROOT=<results-disk>/post_crisp                    # preds/results/table; omit to keep in-repo

# 1) prepare
conda activate infer-env
python data_preparation.py spatialscore

# 2) infer  (one model per run is safest; loads sequentially otherwise — see Run inference)
python infer.py --benchmarks spatialscore --models qwen3vl-4b,qwen3.5-4b

# 3) score  (SpatialScore needs the scoring env; the other benches score in infer-env)
conda activate score-env
python evaluate.py --benchmarks spatialscore --models qwen3vl-4b,qwen3.5-4b
cat results/qwen3.5-4b/spatialscore/summary_report.json

# 4) collect all scored models into a leaderboard
#    (same session as above → env vars still apply; new shell → re-export step b/c first)
python make_table.py --bench spatialscore
```

## Where things land

```
benchmarks/data/<name>/        raw data + <name>.jsonl (prepared input)
preds/<model-tag>/<name>.jsonl predictions (+ <name>.done.json flag)
results/<model-tag>/<name>/     all_results.json, summary_report.json, metrics.json
table/<name>.csv                make_table.py --csv output (leaderboards)
```

Paths are overridable via `CACHE_DIR` / `PREDS_DIR` / `RESULTS_DIR` / `TABLE_DIR` /
`POST_CRISP_ROOT` (see `benchmarks/base.py`; `TABLE_DIR` in `make_table.py`).
