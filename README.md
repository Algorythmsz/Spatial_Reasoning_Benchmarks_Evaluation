# Spatial Reasoning Benchmark Interface for POST_CRISP

Run multimodal (Qwen-VL etc.) models on spatial-reasoning benchmarks and score them,
in three separate steps:

| step | script | environment | what it does |
|---|---|---|---|
| 1. prepare | `data_preparation.py` | inference env | download raw data (HF) → build a ms-swift jsonl |
| 2. infer | `infer.py` | inference env | `swift infer` each model over that jsonl → predictions + a `done.flag` |
| 3. score | `evaluate.py` | inference env (scoring env for SpatialScore) | predictions → run the scorer → `metrics.json` |
- 2 environments are needed: the inference env, plus a scoring env for SpatialScore's LLM judge.
- **Benchmarks:** `spatialscore`, `multihopspatial`, `refspatial_bench`, `refspatial_expand` (in `benchmarks/`).
  *(all four are wired end-to-end.)*
- **Models:** listed in `models.yaml` — edit that file to add/remove models.

---

## 🧩 Benchmarks

Four spatial-reasoning benchmarks. Three of them have an interactive **viewer** to browse the samples (image + question + ground truth); `refspatial_bench` has none. Scoring for all of them is described under [Scoring](#-scoring).

### `spatialscore` — [🔎 Viewer](https://algorythmsz.github.io/SpatialScore_Viewer/)

- **Task:** broad multimodal spatial VQA — a large aggregate benchmark spanning many sub-tasks (depth/height ordering, distance, counting, pose, object localization, …) drawn from several source datasets (CV-Bench, etc.).
- **Input:** one image (or video frames) + a question. For multi-choice, the options are folded into the question text as `(A) … / (B) …` so the prompt is self-contained.
- **Output:** free text — a **letter** for multi-choice, **yes/no** for judgement, or a **number/short phrase** for open-ended (e.g. a distance in meters).
- **Metrics:** overall accuracy + breakdowns by `category` / `task` / `sub_task` / `source_dataset` (rule-based + an LLM judge for open-ended).
- **Source:** [haoningwu/SpatialScore](https://huggingface.co/datasets/haoningwu/SpatialScore) (~15.8 GB images).

### `multihopspatial` — [🔎 Viewer](https://algorythmsz.github.io/MultihopSpatial_Viewer/)

- **Task:** multi-hop spatial reasoning — answer a multiple-choice question that requires chaining several spatial relations, **and** localize the referenced object with a bounding box.
- **Input:** one image + a question that already contains the `(a)…(d)` choices and a bbox request, followed by the output-format block.
- **Output:** an answer line + a box, e.g. `Answer: (a) chair` and `Bounding Box: [x1, y1, x2, y2]`.
- **Metrics:** **MCQ accuracy**, **Acc@50IoU** (MCQ-correct AND bbox IoU ≥ 0.5), **Avg IoU** (mean IoU over MCQ-correct samples).
- **Source:** [etri-vilab/MultihopSpatial](https://huggingface.co/datasets/etri-vilab/MultihopSpatial) (4500 questions).
- Runs the authors' own evaluator, vendored and driven on ms-swift — including its sampled
  decoding and retry rounds, so **scores vary run to run**. See
  [multihopspatial](#multihopspatial--the-official-evaluator-on-ms-swift).

### `refspatial_bench` — *(no viewer)*

- **Task:** referring spatial grounding — given an object + a spatial instruction, **POINT** at the target location(s). Three subsets: `Location`, `Placement`, `Unseen`.
- **Input:** one RGB image + a referring prompt (the grounding prompt is chosen per model at inference).
- **Output:** one or more points as text, e.g. `[(0.25, 0.40)]` (Qwen emits a 0–1000 space and/or JSON `point_2d`).
- **Metrics:** fraction of predicted points that land inside the ground-truth mask, aggregated overall + by subset / step / category.
- **Source:** [BAAI/RefSpatial-Bench](https://huggingface.co/datasets/BAAI/RefSpatial-Bench) (the benchmark released with [RoboRefer](https://github.com/Zhoues/RoboRefer)).

### `refspatial_expand` — [🔎 Viewer](https://algorythmsz.github.io/RefSpatial-Expand-bench_Viewer/)

- **Task:** same point-in-mask grounding task as `refspatial_bench`, on the expanded release. Two subsets: `Location` (241) and `Placement` (200); no `Unseen`.
- **Input:** one RGB image + a referring prompt (the grounding prompt is chosen per model at inference).
- **Output:** one or more points as text, e.g. `[(0.25, 0.40)]` (Qwen emits a 0–1000 space and/or JSON `point_2d`).
- **Metrics:** fraction of predicted points that land inside the ground-truth mask, aggregated overall + by subset / step / category.
- **Source:** [JingkunAn/RefSpatial-Expand-Bench](https://huggingface.co/datasets/JingkunAn/RefSpatial-Expand-Bench). No published baselines for this release.

---

## ⚙️ Setup - for one time

### a. Conda environments

The **inference env** covers most of the pipeline. A separate **scoring env** is needed to score **SpatialScore** (its LLM-judge stage). Everything else — including scoring `multihopspatial` / `refspatial_bench` / `refspatial_expand` — runs in the inference env.

| env (any name) | used by | must contain |
|---|---|---|
| inference | inference **and** scoring `multihopspatial` / `refspatial_bench` / `refspatial_expand` | `ms-swift`, `vllm`, `huggingface_hub`, `datasets` |
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
python data_preparation.py spatialscore     # or: multihopspatial | refspatial_bench | refspatial_expand | all
```

The accepted names are the registered adapters, so anything in `benchmarks/` works — plus `all`.

---

## 🤖 Run inference

Runs `swift infer` for each model over the prepared jsonl. Predictions go to
`preds/<model-tag>/<benchmark>.jsonl`, plus a `<benchmark>.done.json` flag recording the expected
sample count — a (model, benchmark) pair counts as finished only when the flag exists **and** the
prediction line count matches, so a truncated run re-runs instead of being silently skipped.
We disabled Qwen3.5 models' thinking mode by default (You can change at the `models.yaml`). Already-finished (model, benchmark) pairs are skipped.

```bash
conda activate <inference-env> 
python infer.py --benchmarks spatialscore --models qwen3vl-4b,qwen3.5-4b
```

Options: `--benchmarks all` (every benchmark), `--models all` (every model in `models.yaml`), `--max-new-tokens N` (omit to use each benchmark's own default — 512 normally, upstream's 8192 for multihopspatial).

Passing multiple models in one call loads them sequentially in the same process, which can
cause Out-of-Memory (OOM) — vllm doesn't always fully release the GPU between models. Prefer **one model per run**. To run several at once, launch each as its
own job in parallel.

---

## 📈 Scoring
### SpatialScore ##
**SpatialScore needs the scoring env** (LLM judge); the other three benchmarks score in the **inference env**. Activate the right one, then run. Results land in
`results/<model-tag>/<benchmark>/` (`all_results.json`, `summary_report.json`, `metrics.json`).

```bash
conda activate <scoring-env>        # SpatialScore; use the inference env for the others
python evaluate.py --benchmarks spatialscore --models qwen3vl-4b,qwen3.5-4b
```

SpatialScore is scored by the official upstream scorer, vendored unmodified under
`benchmarks/vendor/spatialscore/` (see the README there)

Optional:

```bash
export SS_NO_LLM=1     # rule-only: skip the Stage-2 judge LLM
```
Other optional overrides: `SS_LLM_PATH` (judge, default `openai/gpt-oss-20b`), `SS_SCORER` (a different scorer checkout), `SS_TP_SIZE` / `SS_GPU_MEM` (judge vllm knobs).

### RefSpatial_bench & RefSpatial_expand ###
The RefSpatial family asks the model to output 2D point coordinates `[x, y]`. The official code
pairs a different prompt with a different parsing function per model, and Qwen is not covered
correctly by that mapping. The official prompt asks for coordinates between 0 and 1, but Qwen
answers in the 0–1000 space it was trained to emit for grounding, while the parser paired with Qwen reads those numbers as raw pixels.  So we added a parsing/normalizing function, `qwen1000`, that rescales
them before the point-in-mask check. It is selected automatically for Qwen models.

### multihopspatial — the official evaluator, on ms-swift ###

The authors' harness is public, so the whole protocol is theirs.
[`benchmarks/vendor/multihopspatial/`](benchmarks/vendor/multihopspatial/) holds a
byte-identical copy of `eval/benchmark_qwen_vllm.py`; prompt, retry rounds, answer/bbox
parsing, coordinate scaling, IoU and the hop × view table all run upstream's code.

The only thing we change is the inference backend. Upstream reaches vLLM through two
module-level names, so `swift_backend.py` rebinds them to `swift.VllmEngine` /
`swift.RequestConfig` rather than editing the file. See the
[README there](benchmarks/vendor/multihopspatial/README.md).

**This benchmark's generation settings are upstream's, not the repo's** — deliberately, since
matching them is the point:

| | multihopspatial | our other benches |
|---|---|---|
| `min/max_pixels` | **not set** (model default) | pinned to the SpatialScore protocol |
| decoding | the checkpoint's `generation_config.json` (Qwen3-VL-4B: t=0.7) | greedy |
| `max_new_tokens` | 8192 | 512 |
| retry | up to 3 rounds if the answer or bbox won't parse | none |

So **scores here vary run to run**, and retry-until-valid means a score is "best of ≤3
attempts filtered by format validity", which flatters models with messy output. `--greedy`
buys determinism back but turns the retry rounds off (a retry at temperature 0 reproduces
the same string), and is then no longer the paper's protocol.
---

## 📊 Make a table for results

`make_table.py` scans `results/<model-tag>/<benchmark>/metrics.json` and prints an accuracy leaderboard (sorted by overall). Scored models appear; ones without a `metrics.json` are listed as skipped.

It reads the **same** `POST_CRISP_ROOT` / `RESULTS_DIR` env vars as the rest of the pipeline, so point it at wherever your results live. If you redirected outputs
off the home disk, either prefix each run or `export` it once for the session:

```bash
export POST_CRISP_ROOT=<where-you-stored-the-results>/post_crisp        # skip if already exported this session
python make_table.py --bench spatialscore --breakdown category --csv spatialscore.csv       # write a CSV file of results on SpatialScore with a per-category score
```

`--breakdown` accepts `category`, `task`, `sub_task`, or `source_dataset` — which of those a
benchmark actually fills depends on its scorer (SpatialScore: all four; MultihopSpatial:
`category`=hop, `sub_task`=view, `task`=hop × view; RefSpatial: subset / step / category).

`--metrics` picks which numbers each column shows (default `accuracy`, comma-separate for more).
Only `multihopspatial` emits more than accuracy today (`acc@50iou`, `avg_iou`); a metric a scorer
doesn't emit renders as `-`:

```bash
python make_table.py --bench multihopspatial --breakdown task --metrics accuracy,acc@50iou
```

---

## 🧷 Notes when running several models

- **One model per `infer.py` process.** vLLM doesn't reliably release the GPU between models
  loaded in the same process, so passing several `--models` at once can OOM midway. Scoring has
  no such problem and takes them all in one call.
- **Re-running is safe.** Preparation skips already-downloaded data, and `infer.py` skips
  (model, benchmark) pairs that already finished — a crashed run resumes where it stopped. To
  re-score existing predictions after a scorer change, just run `evaluate.py` again.

> `slurm/` holds the batch-job wrappers used on the cluster this was developed on. They call the
> exact same entry points and are **not needed** to run anything here — ignore that directory
> unless you're on SLURM, in which case start at `slurm/config.sh` (it hardcodes our paths and
> conda env names, so edit it first).

---

## 🏋️ Training (SFT)

`train/` fine-tunes a base VLM on the MultihopSpatial **train** split, to get an upper-bound
reference for that benchmark. Two steps, both in the inference env:

```bash
python train/prepare_mhs_sft.py                  # multihop_train_6791.json -> ms-swift SFT jsonl
python train/train.py --tuner-type lora          # or: --tuner-type full   (--dry-run to inspect kwargs)
```

- Trains on `multihop_train_6791.json`, **disjoint from the 4500-question eval set** — no leakage.
- Prompt and target format both come from the vendored official evaluator (`build_prompt`, and
  `Answer: (b) …` + `Bounding Box: [x1, y1, x2, y2]`), so training inputs match evaluation inputs
  and a checkpoint is directly evaluable through the normal `infer.py` / `evaluate.py` path.
- Same `min/max_pixels` as inference, so image-token counts match between SFT and eval.
- LoRA fits on one 48 GB card; `--tuner-type full` on a 9B needs DeepSpeed ZeRO-3 across
  multiple GPUs (launch it under `torchrun --nproc_per_node=<N>`).
- Checkpoints land in `$POST_CRISP_ROOT/sft/mhs-<base>-<tuner>/`. To evaluate one, add it to
  `models.yaml` with `model_type: qwen3_vl` (FT configs otherwise fail auto-detection), then run
  it through the normal pipeline.

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

MultihopSpatial runs the vendored **official evaluator** on ms-swift, entirely in the inference env (no judge, no scoring env). Inference and scoring both come from upstream's code — see [multihopspatial](#multihopspatial--the-official-evaluator-on-ms-swift).

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

# 2) infer  (upstream protocol: sampled decoding + up to 3 retry rounds. One model per run.
#    --test-samples N for a smoke test; --greedy for determinism, which disables retries.)
python infer.py --benchmarks multihopspatial --models qwen3vl-4b

# 3) score  (same env; metrics come from upstream's calculate_full_metrics)
python evaluate.py --benchmarks multihopspatial --models qwen3vl-4b
cat results/qwen3vl-4b/multihopspatial/summary_report.json     # Acc, Acc@50IoU, avg IoU + paper_table

# 4) leaderboard  (add the paper's hop x view cells)
python make_table.py --bench multihopspatial
python make_table.py --bench multihopspatial --breakdown task --metrics accuracy,acc@50iou
```

## 🖼️ Full example from scratch (qwen3vl-4b on RefSpatial-Expand)

RefSpatial-Expand also scores in the **inference env** (point-in-mask, PIL + numpy — no judge). Swap `refspatial_expand` → `refspatial_bench` below for the original three-subset benchmark; every step is the same.

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

# 3) score  (the point parser is auto-picked from the model; `evaluate.py --help` to override)
python evaluate.py --benchmarks refspatial_expand --models qwen3vl-4b
cat results/qwen3vl-4b/refspatial_expand/summary_report.json   # overall + by subset/step/category

# 4) leaderboard
python make_table.py --bench refspatial_expand
```

## Where things land

```
benchmarks/data/<name>/         raw data + <name>.jsonl (prepared input)   [BENCH_DATA_DIR]
preds/<model-tag>/<name>.jsonl  predictions (+ <name>.done.json flag)      [PREDS_DIR]
results/<model-tag>/<name>/     all_results.json, summary_report.json, metrics.json  [RESULTS_DIR]
table/<name>.csv                make_table.py --csv output (leaderboards)  [TABLE_DIR]
sft/mhs-<base>-<tuner>/         train/train.py checkpoints
```

`benchmarks/vendor/` holds byte-identical copies of the benchmarks' official code (SpatialScore's scorer, MultihopSpatial's whole evaluator) — read-only, each with its own README.

Everything except `benchmarks/data/` (which follows `BENCH_DATA_DIR`) sits under
`POST_CRISP_ROOT`, and each row is individually overridable via the env var in brackets
(see `benchmarks/base.py`; `TABLE_DIR` in `make_table.py`).
