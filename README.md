# Spatial Reasoning Benchmark Interface for POST_CRISP

Run multimodal (Qwen-VL etc.) models on spatial-reasoning benchmarks and score them,
in three separate steps:

| step | script | enviornment | what it does |
|---|---|---|---|
| 1. prepare | `data_preparation.py` | inference env | download raw data (HF) → build a ms-swift jsonl |
| 2. infer | `infer.py` | inference env | `swift infer` each model over that jsonl → predictions + a `done.flag` |
| 3. score | `evaluate.py` | inference env (scoring env for SpatialScore) | predictions → run the scorer → `metrics.json` |
- 2 environments are needed: the inference env, plus a scoring env for SpatialScore's LLM judge.
- **Benchmarks:** `spatialscore`, `multihopspatial`, `refspatial_expand` (in `benchmarks/`).
  *(all three are wired end-to-end.)*
- **Models:** listed in `models.yaml` — edit that file to add/remove models.

---

## 🧩 Benchmarks

Three spatial-reasoning benchmarks, each with an interactive **viewer** to browse its samples (image + question + ground truth). Scoring for all three is described under [Scoring](#-scoring).

### `spatialscore` — [🔎 Viewer](https://algorythmsz.github.io/SpatialScore_Viewer/)

- **Task:** broad multimodal spatial VQA — a large aggregate benchmark spanning many sub-tasks (depth/height ordering, distance, counting, pose, object localization, …) drawn from several source datasets (CV-Bench, etc.).
- **Input:** one image (or video frames) + a question. For multi-choice, the options are folded into the question text as `(A) … / (B) …` so the prompt is self-contained.
- **Output:** free text — a **letter** for multi-choice, **yes/no** for judgement, or a **number/short phrase** for open-ended (e.g. a distance in meters).
- **Metrics:** overall accuracy + breakdowns by `category` / `task` / `sub_task` / `source_dataset` (rule-based + an LLM judge for open-ended).
- **Source:** [haoningwu/SpatialScore](https://huggingface.co/datasets/haoningwu/SpatialScore) (~15.8 GB images).

### `multihopspatial` — [🔎 Viewer](https://algorythmsz.github.io/MultihopSpatial_Viewer/)

- **Task:** multi-hop spatial reasoning — answer a multiple-choice question that requires chaining several spatial relations, **and** localize the referenced object with a bounding box.
- **Input:** one image + a question that already contains the `(a)…(d)` choices and a bbox request; a system prompt pins the output format.
- **Output:** an answer line + a box, e.g. `Answer: (a) chair` and `bbox_2d` in **normalized `[0,1]` xyxy**.
- **Metrics:** **MCQ accuracy**, **Acc@50IoU** (MCQ-correct AND bbox IoU ≥ 0.5), **Avg IoU** (mean IoU over MCQ-correct samples).
- **Source:** [etri-vilab/MultihopSpatial](https://huggingface.co/datasets/etri-vilab/MultihopSpatial) (4500 questions).

### `refspatial_expand` — [🔎 Viewer](https://algorythmsz.github.io/RefSpatial-Expand-bench_Viewer/)

- **Task:** referring spatial grounding — given an object + a spatial instruction, **POINT** at the target location(s). Two subsets: `Location` and `Placement`.
- **Input:** one RGB image + a referring prompt (the grounding prompt is chosen per model at inference — see the [parse-function table](#refspatial_expand-point-parser---parse-function)).
- **Output:** one or more points as text, e.g. `[(0.25, 0.40)]` (Qwen emits a 0–1000 space and/or JSON `point_2d`).
- **Metrics:** fraction of predicted points that land inside the ground-truth mask, aggregated overall + by subset / step / category.
- **Source:** [JingkunAn/RefSpatial-Expand-Bench](https://huggingface.co/datasets/JingkunAn/RefSpatial-Expand-Bench).

---

## ⚙️ Setup - for one time

### a. Conda environments

The **inference env** covers most of the pipeline. A separate **scoring env** is needed to score **SpatialScore** (its LLM-judge stage). Everything else — including scoring `multihopspatial`/`refspatial_expand` — runs in the inference env.

| env (any name) | used by | must contain |
|---|---|---|
| inference | inference **and** scoring `multihopspatial` / `refspatial_expand` | `ms-swift`, `vllm`, `huggingface_hub`, `datasets` |
| scoring | scoring **SpatialScore only** | `vllm` + judge LLM (`openai/gpt-oss-20b`), `tqdm`,`torch`, `torchvision`, `matplotlib`, `numpy`, `pillow` |

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
(vllm 0.11 + torch 2.8+cu128 + scorer's deps) and includes the cu128 index.
You only need this env to score SpatialScore.

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

### c. Designate where to download and store data / outputs (Optional)

By default benchmark data and outputs land **inside the repo** (home disk). SpatialScore
alone is ~15.8 GB — if home is tight, point these at a big disk:

```bash
export BENCH_DATA_DIR=<where-to-download-the-data>/post_crisp/data   # raw + preprocessed benchmark data
export POST_CRISP_ROOT=<where-to-store-the-results>/post_crisp       # cache / preds / results
```


---

## 📁 Prepare data

Downloads each benchmark's raw data (into `benchmarks/data/<name>/`) and builds the
ms-swift style input jsonl. 

```bash
conda activate <inference-env>
python data_preparation.py spatialscore     # or: multihopspatial | refspatial_expand | all
```

---

## 🤖 Run inference

Runs `swift infer` for each model over the prepared jsonl. Predictions go to
`preds/<model-tag>/<benchmark>.jsonl`, plus a `done.flag` marking a clean finish.
We disabled Qwen3.5 models' thinking mode by default (You can change at the `models.yaml`). Already-finished (model, benchmark) pairs are skipped.

```bash
conda activate <inference-env> 
python infer.py --benchmarks spatialscore --models qwen3vl-4b,qwen3.5-4b
```

Options: `--benchmarks all` (every benchmark), `--models all` (every model in `models.yaml`), `--max-new-tokens N` (default 512).

Passing multiple models in one call loads them sequentially in the same process, which can
cause Out-of-Memory (OOM) — vllm doesn't always fully release the GPU between models. Prefer **one model per run**. To run several at once, launch each as its
own job in parallel.

---

## 📈 Scoring

**SpatialScore needs the scoring env** (LLM judge); the other benchmarks could be scored in the **inference env**. Activate the right one, then run. Results land in
`results/<model-tag>/<benchmark>/` (`all_results.json`, `summary_report.json`, `metrics.json`).

```bash
conda activate <scoring-env>        # SpatialScore; use the inference env for the other two
python evaluate.py --benchmarks spatialscore --models qwen3vl-4b,qwen3.5-4b
```

SpatialScore's scorer lives at `benchmarks/scorers/spatialscore/evaluate_results.py` — a local copy of the official SpatialScore code, vendored here so scoring is self-contained.

Optional:

```bash
export SS_NO_LLM=1     # rule-only: skip the Stage-2 judge LLM (no GPU)
```
Other optional overrides: `SS_LLM_PATH` (judge, default `openai/gpt-oss-20b`), `SS_SCORER` (a different scorer checkout), `SS_TP_SIZE` / `SS_GPU_MEM` (judge vllm knobs).

### `refspatial_expand`: point parser (`--parse-function`)

`refspatial_expand` is a point-in-mask task: the model POINTs at a target and the scorer counts how many predicted points land inside the GT mask. **Different models write coordinates in different formats**, so the scorer must know how to read them. `--parse-function` is **optional**:

- **Omit it → auto best-of.** The scorer scores with the model's parser (per-model dispatch ported from the official evaluator: `molmo`→`xml`, `gemini`→`json`, `robobrain`→`absolute`, `qwen`→`qwen1000`, else `normalized`) **and** with `normalized` (`text2pts`), then **reports whichever scores higher overall**. This is robust to a wrong auto-pick or a model that emits an unexpected format. Both candidates' full results are persisted (`metrics_<name>.json` / `summary_report_<name>.json`).
- **Pass it → force** that parser for every model (the first of a comma list is primary, no best-of).

| `--parse-function` | how it reads coords | official RefSpatial models that use it |
|---|---|---|
| `normalized` | `(x, y)` tuples; **float** coords scale by `(W, H)`, **int** coords are absolute original pixels | RoboPoint, Claude, GPT4O, RoboRefer |
| `absolute`   | `(x, y)` tuples taken as raw pixels, **no scaling** | RoboBrain, Qwen *(official)* |
| `xml`        | Molmo `x1="..." y1="..."` attributes (percent `/100`) | Molmo |
| `json`       | ```` ```json ```` block of `[{"point": [y, x]}]` (norm1000 `/1000`) | Gemini |
| `qwen1000`   | norm1000 magnitude rule: `v` in `[0,1]` → `v·dim`, `\|v\|>1` → `v/1000·dim`; parses **both** `(x, y)` tuples and JSON `point_2d` | **Qwen3-VL / Qwen3.5 (this repo — recommended)** |

> **Why `qwen1000` for Qwen and not `absolute`?** Qwen3-VL / Qwen3.5 emit points in a 0–1000 normalized space (ms-swift's `norm_bbox: norm1000`). The official evaluator scores Qwen with `absolute`, which reads those as raw pixels and under-reproduces badly (acc ~0.05–0.10). `qwen1000` interprets them correctly.

```bash
conda activate <inference-env>
python evaluate.py --benchmarks refspatial_expand --models qwen3vl-4b --parse-function qwen1000
```

**Score under several parsers at once** — comma-separate them. The **first is primary** (feeds the canonical `metrics.json` / the leaderboard / the per-question `accuracy` field). Every parser you list also gets its **own standalone file pair**, name-suffixed — `metrics_<name>.json` and `summary_report_<name>.json` — plus a per-question `accuracy_<name>` in `all_results.json`, so you can compare with no re-run:

```bash
python evaluate.py --benchmarks refspatial_expand --models qwen3vl-4b --parse-function qwen1000,normalized
# -> metrics.json (=qwen1000, primary) + metrics_qwen1000.json + metrics_normalized.json
#    + summary_report.json (=qwen1000) + summary_report_qwen1000.json + summary_report_normalized.json
```

(A single `--parse-function` writes only the plain `metrics.json` / `summary_report.json`, exactly as before — no suffixed files.)

### `multihopspatial`: scoring knobs (optional)

`multihopspatial` is an MCQ + bounding-box task, scored rule-based in the inference env (no judge). Metrics: **MCQ accuracy**, **Acc@50IoU** (MCQ-correct AND bbox IoU ≥ threshold), **Avg IoU** (mean IoU over MCQ-correct samples). Optional env knobs:

```bash
export MHS_IOU_THR=0.5   # IoU threshold for Acc@IoU (default 0.5)
export MHS_STRICT=1      # reject coord-scale / xywh rescue heuristics (stricter bbox parsing)
```

---

## 📊 Make a table for results

`make_table.py` scans `results/<model-tag>/<benchmark>/metrics.json` and prints an accuracy leaderboard (sorted by overall). Scored models appear; ones without a `metrics.json` are listed as skipped.

It reads the **same** `POST_CRISP_ROOT` / `RESULTS_DIR` env vars as the rest of the pipeline, so point it at wherever your results live. If you redirected outputs
off the home disk, either prefix each run or `export` it once for the session:

```bash
export POST_CRISP_ROOT=<where-you-stored-the-results>/post_crisp        # skip if already exported this session
python make_table.py --bench spatialscore --breakdown category --csv spatialscore.csv       # write a CSV file of results on SpatialScore with a per-category score
```

`--breakdown` accepts `category`, `task`, `sub_task`, or `source_dataset`.

---

## 🖼️ Full example from scratch (qwen3vl-4b on SpatialScore)

```bash
# a) build the envs (once, ever — see Step a for details/notes)
conda create -n infer-env python=3.11 -y
conda activate infer-env
pip install -r requirements/infer.txt

conda create -n score-env python=3.10 -y         # needed to score SpatialScore
conda activate score-env
pip install -r requirements/score.txt

# b/c) env vars (optional; put in ~/.bashrc to skip retyping)
export HF_HOME=<hf-cache-dir>                                       # model cache; can skip if you want to store it in ~/.cache/huggingface
export BENCH_DATA_DIR=<data-disk>/post_crisp/data                   # benchmark data; can skip if you want to keep it in-repo
export POST_CRISP_ROOT=<results-disk>/post_crisp                    # preds/results/table; can skip if you want to keep it in-repo

# 1) prepare
conda activate infer-env
python data_preparation.py spatialscore

# 2) infer  (one model per run is safest; loads sequentially otherwise — see Run inference)
python infer.py --benchmarks spatialscore --models qwen3vl-4b

# 3) score  (SpatialScore needs the scoring env; the other benches score in infer-env)
conda activate score-env
python evaluate.py --benchmarks spatialscore --models qwen3vl-4b
cat results/qwen3vl-4b/spatialscore/summary_report.json

# 4) collect all scored models into a leaderboard
#    (same session as above → env vars still apply; new shell → re-export step b/c first)
python make_table.py --bench spatialscore
```

## 🖼️ Full example from scratch (qwen3vl-4b on MultihopSpatial)

MultihopSpatial scores entirely in the **inference env** (rule-based MCQ + bbox IoU — no judge, no scoring env).

```bash
# a) build the inference env (once, ever — see Step a). No scoring env needed here.
conda create -n infer-env python=3.11 -y
conda activate infer-env
pip install -r requirements/infer.txt

# b/c) env vars (optional; put in ~/.bashrc to skip retyping)
export HF_HOME=<hf-cache-dir>                                       # model cache
export BENCH_DATA_DIR=<data-disk>/post_crisp/data                   # benchmark data
export POST_CRISP_ROOT=<results-disk>/post_crisp                    # preds/results/table

# 1) prepare  (downloads etri-vilab/MultihopSpatial: 4500 questions + COCO-style images)
python data_preparation.py multihopspatial

# 2) infer  (one model per run is safest)
python infer.py --benchmarks multihopspatial --models qwen3vl-4b

# 3) score  (same env; optional: export MHS_IOU_THR / MHS_STRICT before this)
python evaluate.py --benchmarks multihopspatial --models qwen3vl-4b
cat results/qwen3vl-4b/multihopspatial/summary_report.json     # MCQ acc, Acc@50IoU, Avg IoU

# 4) leaderboard
python make_table.py --bench multihopspatial
```

## 🖼️ Full example from scratch (qwen3vl-4b on RefSpatial-Expand)

RefSpatial-Expand also scores in the **inference env** (point-in-mask, PIL + numpy — no judge). Scoring **auto-picks the point parser per model** (Qwen → `qwen1000`); pass `--parse-function` only to override (see [point parser](#refspatial_expand-point-parser---parse-function)).

```bash
# a) build the inference env (once, ever — see Step a). No scoring env needed here.
conda create -n infer-env python=3.11 -y
conda activate infer-env
pip install -r requirements/infer.txt

# b/c) env vars (optional; put in ~/.bashrc to skip retyping)
export HF_HOME=<hf-cache-dir>                                       # model cache
export BENCH_DATA_DIR=<data-disk>/post_crisp/data                   # benchmark data
export POST_CRISP_ROOT=<results-disk>/post_crisp                    # preds/results/table

# 1) prepare  (downloads JingkunAn/RefSpatial-Expand-Bench: Location/ + Placement/ image+mask)
python data_preparation.py refspatial_expand

# 2) infer  (per-model grounding prompt is baked in automatically; one model per run)
python infer.py --benchmarks refspatial_expand --models qwen3vl-4b

# 3) score  (--parse-function REQUIRED; qwen1000 for Qwen. Add ,normalized to also see the official reading)
python evaluate.py --benchmarks refspatial_expand --models qwen3vl-4b --parse-function qwen1000
cat results/qwen3vl-4b/refspatial_expand/summary_report.json   # overall + by subset/step/category

# 4) leaderboard
python make_table.py --bench refspatial_expand
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
