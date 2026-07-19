# POST_CRISP — spatial benchmark harness

Run multimodal (Qwen-VL etc.) models on spatial-reasoning benchmarks and score them,
in three separate steps:

| step | script | enviornment | what it does |
|---|---|---|---|
| 1. prepare | `data_preparation.py` | inference env | download raw data (HF) → build a model-agnostic ms-swift jsonl |
| 2. infer | `infer.py` | inference env | `swift infer` each model over that jsonl → predictions + a `done.flag` |
| 3. score | `evaluate.py` | scorer env (gpt-oss is needed at SpatialScore) | predictions → run the scorer → `metrics.json` |
- 2 environments are needed. One for 'ms-swift' and one for evaluating SpatialScore.
- **Benchmarks:** `spatialscore`, `multihopspatial`, `refspatial_expand` (in `benchmarks/`).
  *(`spatialscore` and `refspatial_expand` are wired end-to-end. `multihopspatial` runs
  inference but its scorer is not implemented yet — reshape/score raise NotImplementedError.)*
- **Models:** listed in `models.yaml` — edit that file to add/remove models.

---

## Step 0 — one-time setup

### 0a. Conda environments

You need one **inference** env and one **scoring** env (that has each benchmark's deps).
**Name them whatever you like** — the scripts run in whatever conda env is active and
don't check names. Just make sure you activate an env that has the right deps for the
step you're running:

| env (any name) | used by | must contain |
|---|---|---|
| inference | steps 1 & 2 | `ms-swift`, `vllm`, `huggingface_hub`, `datasets` |
| scoring | step 3 | `vllm` + judge LLM (`openai/gpt-oss-20b`), `tqdm`, and the vendored scorer's deps: `torch`, `torchvision`, `matplotlib`, `numpy`, `pillow` |

One scoring env covers **all** benchmarks: spatialscore drives the heavy list above, and
refspatial_expand/multihopspatial only need `numpy`/`pillow`/`PyYAML`, which that list
already includes — so the spatialscore env is a superset and there's no need to split it.

Per-env dependency lists live in `requirements/`.

#### How to build the inference env

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

#### How to build the scoring env

Python 3.10 + CUDA 12.8. `requirements/score.txt` pins the working set
(vllm 0.11 + torch 2.8+cu128 + the vendored scorer's deps) and includes the cu128 index.
This single env scores every benchmark (see note above).

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

### 0b. Designate where to download the model weights from Huggingface

`models.yaml` uses HF repo ids. Two things make them resolve correctly:

```bash
export USE_HF=1                
export HF_HOME=<where-you-want-to-download>  # where to download the model
```

Put these in your shell profile (`~/.bashrc`) or the env's activation hook
(`$CONDA_PREFIX/etc/conda/activate.d/`) so you don't retype them — **not** in this repo.
Without them, a repo id re-downloads (e.g. a 27B model) instead of using your cache.

### 0c. Designate where to download/store data/outputs

By default benchmark data and outputs land **inside the repo** (home disk). SpatialScore
alone is ~15.8 GB — if home is tight, point these at a big disk:

```bash
export BENCH_DATA_DIR=<where-you-want-to-download>/post_crisp/data   # raw + preprocessed benchmark data
export POST_CRISP_ROOT=<where-you-want-to-store-the-results>/post_crisp       # cache / preds / results
```

`BENCH_DATA_DIR` is separate from `POST_CRISP_ROOT` — set **both**, or benchmark data
still defaults into the repo. (Fine-grained overrides: `CACHE_DIR` / `PREDS_DIR` /
`RESULTS_DIR`; see `benchmarks/base.py`.)

---

## Step 1 — prepare data

Downloads each benchmark's raw data (into `benchmarks/data/<name>/`) and builds the
ms-swift input jsonl. Idempotent — re-running only regenerates when data/prompts change.

```bash
conda activate <inference-env>
python data_preparation.py spatialscore     # or: multihopspatial | refspatial_expand | all
```

---

## Step 2 — run inference

Runs `swift infer` for each model over the prepared jsonl. Predictions go to
`preds/<model-tag>/<benchmark>.jsonl`, plus a `done.flag` marking a clean finish.
Per-model image budget (`min/max_pixels`) and `enable_thinking` from `models.yaml`
are injected automatically. We disabled Qwen3.5 models' thinking mode by default (You can change at the `models.yaml`). Already-finished (model, benchmark) pairs are skipped.

```bash
conda activate <inference-env>
export USE_HF=1                          # (+ HF_HOME if not default)
python infer.py --benchmarks spatialscore --models qwen3vl-4b,qwen3.5-4b
```

Options: `--all` (every benchmark), `--max-new-tokens N` (default 512).

---

## Step 3 — score

Activate an env that has the benchmark's scoring deps (Step 0a) **first**, then run.
Scoring reshapes the predictions and runs the scorer; results land in
`results/<model-tag>/<benchmark>/` (`all_results.json`, `summary_report.json`, `metrics.json`).

```bash
conda activate <your scoring env>
python evaluate.py --benchmarks spatialscore --models qwen3vl-4b,qwen3.5-4b
```

Evaluation code for SpatialScore is `scorers/spatialscore/evaluate_reesults.py`. The code is originally from official SpatialScore code base, but we locally copied it to this code base.

Optional:

```bash
export SS_NO_LLM=1     # rule-only: skip the Stage-2 judge LLM (no GPU)
```
Other optional overrides: `SS_LLM_PATH` (judge, default `openai/gpt-oss-20b`),
`SS_SCORER` (a different scorer checkout), `SS_TP_SIZE` / `SS_GPU_MEM` (judge vllm knobs).

---

## Step 4 — collect results into a table

`make_table.py` scans `results/<model-tag>/<benchmark>/metrics.json` and prints an
accuracy leaderboard (sorted by overall). Only models whose scoring finished appear;
scored-but-incomplete models are listed as skipped, not silently dropped.

It reads the **same** `POST_CRISP_ROOT` / `RESULTS_DIR` env vars as the rest of the
pipeline (Step 0c), so point it at wherever your results live. If you redirected outputs
off the home disk, either prefix each run or `export` it once for the session:

```bash
export POST_CRISP_ROOT=<where-you-stored-the-results>/post_crisp   # same value you used in Step 0c
python make_table.py                           # overall accuracy, all scored models
python make_table.py --breakdown category      # add per-category columns
python make_table.py --csv spatialscore.csv    # also write a CSV
```

`--breakdown` accepts `category`, `task`, `sub_task`, or `source_dataset`.
A `--csv` given as a **bare filename** is written under `POST_CRISP_ROOT/table/`
(e.g. `table/spatialscore.csv`); pass an absolute path to write elsewhere.
(`RESULTS_DIR` / `TABLE_DIR` override those two locations directly.)

---

## Full example (qwen3vl-4b + qwen3.5-4b on spatialscore)

```bash
# 0a) build the two conda envs (once, ever — see Step 0a for details/notes)
conda create -n infer-env python=3.11 -y
conda activate infer-env
pip install -r requirements/infer.txt            # ms-swift + vllm inference stack

conda create -n score-env python=3.10 -y
conda activate score-env
pip install -r requirements/score.txt            # single scoring env (all benches; cu128 torch)

# 0b/0c) env vars (set once per session; put in ~/.bashrc to skip retyping)
export USE_HF=1                                   # use HuggingFace, not ModelScope
export HF_HOME=<your HF cache>                    # where models download/cache; omit for ~/.cache/huggingface
export BENCH_DATA_DIR=<big-disk>/post_crisp/data  # where benchmark data downloads; omit to keep in-repo
export POST_CRISP_ROOT=<big-disk>/post_crisp      # where preds/results/table land; omit to keep in-repo
# (HF_HOME / BENCH_DATA_DIR / POST_CRISP_ROOT are only needed if the home disk is tight;
#  with room to spare, `export USE_HF=1` alone is enough — the rest fall back to defaults.)

# 1) prepare
conda activate infer-env
python data_preparation.py spatialscore

# 2) infer
python infer.py --benchmarks spatialscore --models qwen3vl-4b,qwen3.5-4b

# 3) score  (single scoring env — has vllm + torch/torchvision/matplotlib for every bench)
conda activate score-env
python evaluate.py --benchmarks spatialscore --models qwen3vl-4b,qwen3.5-4b
cat results/qwen3.5-4b/spatialscore/summary_report.json

# 4) collect all scored models into a leaderboard
#    (same session as above → env vars still apply; new shell → re-export step 0 first)
python make_table.py
```

### One-shot (single machine, no SLURM)

`run_spatialscore.sh` chains all four steps for spatialscore — it activates the
inference env for prepare+infer, switches to the scoring env for evaluate, then
writes the leaderboard to `table/spatialscore.csv`:

```bash
./run_spatialscore.sh qwen3vl-4b,qwen3.5-4b
```

Defaults match a specific box; override from the environment for yours:

```bash
INFER_ENV=<inference-env> SCORE_ENV=<scoring-env> \
POST_CRISP_ROOT=<big-disk>/post_crisp HF_HOME=<your HF cache> \
./run_spatialscore.sh qwen3vl-4b
```

`INFER_ENV`/`SCORE_ENV` name the two conda envs; storage vars (Step 0) fall back to
those defaults only if not already exported. On a cluster, prefer the SLURM scripts
(inference+scoring); run `make_table.py` afterwards.

---

## Where things land

```
benchmarks/data/<name>/        raw data + <name>.jsonl (prepared input)
preds/<model-tag>/<name>.jsonl predictions (+ <name>.done.json flag)
results/<model-tag>/<name>/     all_results.json, summary_report.json, metrics.json
table/<name>.csv                make_table.py --csv output (leaderboards)
```

Paths are overridable via `CACHE_DIR` / `PREDS_DIR` / `RESULTS_DIR` / `TABLE_DIR` /
`POST_CRISP_ROOT` (see `benchmarks/base.py`; `TABLE_DIR` in `make_table.py`).
