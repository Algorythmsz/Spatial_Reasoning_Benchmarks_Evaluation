# POST_CRISP — spatial benchmark harness

Run multimodal (Qwen-VL etc.) models on spatial-reasoning benchmarks and score them,
in three separate steps:

| step | script | runs in | what it does |
|---|---|---|---|
| 1. prepare | `data_preparation.py` | inference env | download raw data (HF) → build a model-agnostic ms-swift jsonl |
| 2. infer | `test.py` | inference env | `swift infer` each model over that jsonl → predictions + a `done.flag` |
| 3. score | `evaluate.py` | an env with the scorer's deps | predictions → run the scorer → `metrics.json` |

- **Benchmarks:** `spatialscore`, `multihopspatial`, `refspatial_expand` (in `benchmarks/`).
  *(Only `spatialscore` is fully wired end-to-end today; the other two still need their
  reshape/score implemented.)*
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
| scoring — spatialscore | step 3 | `vllm` + judge LLM (`openai/gpt-oss-20b`), `tqdm`, and the vendored scorer's deps: `torch`, `torchvision`, `matplotlib`, `numpy`, `pillow` |
| scoring — refspatial_expand | step 3 | `pillow`, `numpy` |
| scoring — multihopspatial | step 3 | `numpy` |

Per-env dependency lists live in `requirements/`.

#### Rebuild the inference env from scratch

The inference env is **vllm-based** (`models.yaml` uses `backend: vllm`). Build it clean
in a fresh env so vllm can pin a consistent `torch` — **do not** `pip install vllm` on top
of an existing torch/sglang stack; the CUDA/torch versions conflict and break both.

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

Notes:
- **Qwen3.5** (`model_type: qwen3_5`) is new — it needs a recent `transformers`, and your
  vllm build must recognize `qwen3_5`. If `swift infer` can't load it, `pip install -U
  transformers` and/or a newer vllm. Verify on a Qwen3.5 model before a big run.
- Prefer **sglang** instead of vllm? Keep the sglang stack and set each model's
  `backend: sglang` in `models.yaml` (test.py passes it straight to `--infer_backend`).

#### Rebuild the scoring env from scratch (spatialscore)

Python 3.10 + CUDA 12.8. `requirements/score-spatialscore.txt` pins the working set
(vllm 0.11 + torch 2.8+cu128 + the vendored scorer's deps) and includes the cu128 index.

```bash
# 1. fresh env
conda create -n <scoring-env> python=3.10 -y
conda activate <scoring-env>

# 2. install (torch/torchvision come from the cu128 index in the file)
pip install -r requirements/score-spatialscore.txt

# 3. verify
python -c "import vllm, torch, torchvision, matplotlib, openai_harmony; print('torch', torch.__version__)"

# 4. lock
pip freeze > requirements/score-spatialscore.lock.txt
```

Notes:
- The Stage-2 judge is `openai/gpt-oss-20b` (resolved from the HF cache; `SS_LLM_PATH`
  to override). It needs `openai-harmony` (in the requirements) for its chat format.
- Skip the judge entirely with `SS_NO_LLM=1` (rule-only, no GPU) — but the scorer still
  imports vllm at module top, so vllm must be installed regardless.

`requirements/score-multihopspatial.txt` and `score-refspatial.txt` cover those benches'
(not-yet-implemented) scorers.

### 0b. Point at your HuggingFace cache (once)

`models.yaml` uses HF repo ids. Two things make them resolve correctly:

```bash
export USE_HF=1                 # ms-swift would use ModelScope otherwise
export HF_HOME=<your HF cache>  # the dir that contains hub/models--* ; skip if it's the default ~/.cache/huggingface
```

Put these in your shell profile (`~/.bashrc`) or the env's activation hook
(`$CONDA_PREFIX/etc/conda/activate.d/`) so you don't retype them — **not** in this repo.
Without them, a repo id re-downloads (e.g. a 27B model) instead of using your cache.

### 0c. Redirect data/outputs off the home disk (if it's small/full)

By default benchmark data and outputs land **inside the repo** (home disk). SpatialScore
alone is ~15.8 GB — if home is tight, point these at a big disk:

```bash
export BENCH_DATA_DIR=<big-disk>/post_crisp/data   # raw + preprocessed benchmark data
export POST_CRISP_ROOT=<big-disk>/post_crisp       # cache / preds / results
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

⚠️ SpatialScore images are a ~15.8 GB zip — the first run takes a while.

---

## Step 2 — run inference

Runs `swift infer` for each model over the prepared jsonl. Predictions go to
`preds/<model-tag>/<benchmark>.jsonl`, plus a `done.flag` marking a clean finish.
Per-model image budget (`min/max_pixels`) and `enable_thinking` from `models.yaml`
are injected automatically. Already-finished (model, benchmark) pairs are skipped.

```bash
conda activate <inference-env>
export USE_HF=1                          # (+ HF_HOME if not default)
python test.py --benchmarks spatialscore --models qwen3vl-4b,qwen3.5-4b
```

Options: `--all` (every benchmark), `--max-new-tokens N` (default 512).

---

## Step 3 — score

Activate an env that has the benchmark's scoring deps (Step 0a) **first**, then run.
Scoring reshapes the predictions and runs the scorer; results land in
`results/<model-tag>/<benchmark>/` (`all_results.json`, `summary_report.json`, `metrics.json`).

```bash
conda activate <your scoring env>        # any env that has the deps (Step 0a)
python evaluate.py --benchmarks spatialscore --models qwen3vl-4b,qwen3.5-4b
```

Scoring is **self-contained** for spatialscore (the scorer is vendored under
`benchmarks/scorers/spatialscore/`) — no exports needed. Optional:

```bash
export SS_NO_LLM=1     # rule-only: skip the Stage-2 judge LLM (no GPU)
```
Other optional overrides: `SS_LLM_PATH` (judge, default `openai/gpt-oss-20b`),
`SS_SCORER` (a different scorer checkout), `SS_TP_SIZE` / `SS_GPU_MEM` (judge vllm knobs).

If benches need **different** deps, activate an env with one set, score those benches,
then switch envs and re-run for the rest. (If the active env is missing a scorer's deps,
that benchmark just fails with an import error — activate the right env and re-run.)

---

## Full example (qwen3vl-4b + qwen3.5-4b on spatialscore)

```bash
# 1) prepare
conda activate <inference-env>
export USE_HF=1 HF_HOME=<your HF cache>
python data_preparation.py spatialscore

# 2) infer
python test.py --benchmarks spatialscore --models qwen3vl-4b,qwen3.5-4b

# 3) score  (activate an env that has vllm + torch/torchvision/matplotlib)
conda activate <scoring-env>
python evaluate.py --benchmarks spatialscore --models qwen3vl-4b,qwen3.5-4b
cat results/qwen3.5-4b/spatialscore/summary_report.json
```

---

## Where things land

```
benchmarks/data/<name>/        raw data + <name>.jsonl (prepared input)
preds/<model-tag>/<name>.jsonl predictions (+ <name>.done.json flag)
results/<model-tag>/<name>/     all_results.json, summary_report.json, metrics.json
```

Paths are overridable via `CACHE_DIR` / `PREDS_DIR` / `RESULTS_DIR` / `POST_CRISP_ROOT`
(see `benchmarks/base.py`).
