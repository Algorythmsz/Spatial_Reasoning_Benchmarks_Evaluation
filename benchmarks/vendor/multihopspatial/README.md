# Vendored: official MultihopSpatial evaluator

**byte-identical copies** of `eval/*.py` from
[youngwanLEE/multihopspatial](https://github.com/youngwanLEE/multihopspatial) @ `ab711b8`,
one per model family. Upstream's Apache-2.0 `LICENSE` sits next to them. **Treat them as
read-only** — every part of this protocol that we previously guessed at turned out to differ
from the real thing.

    benchmark_qwen_vllm.py    benchmark_gpt.py    benchmark_claude.py    benchmark_gemini.py

(Upstream also has `benchmark_qwen.py`, a plain-transformers variant of the Qwen script. Not
vendored: our inference backend is vLLM, so `benchmark_qwen_vllm.py` is the matching one.)

`benchmarks/multihopspatial.py` imports the pieces that define what a score means and uses
them unchanged:

| imported | what it fixes |
|---|---|
| `build_prompt` | the prompt text and layout (question first, then the format block) |
| `parse_response` | answer letter + bbox, including the per-box `any(v>1) -> /1000` scale rule |
| `calculate_iou` | COCO xywh GT -> pixel-space IoU |
| `compute_score` | MCQ correctness |
| `calculate_full_metrics` | Acc / Acc@50IoU / avg IoU, and their denominators |

## The evaluators are not interchangeable — routing

Upstream ships one script per model family, and the differences are not cosmetic:

| script | prompt asks for | parser |
|---|---|---|
| `benchmark_qwen*.py` *(vendored here)* | bare `Bounding Box: [x1, y1, x2, y2]`, **no range instruction** | per-box `any(v > 1) -> /1000` rescue (Qwen answers in its native 0-1000 space) |
| `benchmark_gpt.py`, `benchmark_claude.py` | `{"bbox_2d": [...]}`, **"Use NORMALIZED (0.0 to 1.0)"** | none — the 0-1 instruction is trusted |
| `benchmark_gemini.py` | `{"bbox_2d": [y1, x1, y2, x2]}` | axis order swapped |

Note the prompt printed in the paper (`bbox_2d` + "Use NORMALIZED coordinates (0.0 to 1.0)"
+ an example) is the **GPT/Claude** one, not Qwen's. Reading the paper alone would give you
the wrong prompt for a Qwen run.

Scoring a GPT-style model with the Qwen parser would divide its 0-1 coordinates by 1000 the
moment one exceeded 1, and a Gemini-style model would come out axis-swapped — wrong numbers,
no error. So `benchmarks/multihopspatial.py` routes:

- `family_for(model)` matches the tag+path against `gemini | claude | gpt | qwen`; anything
  else is **refused**, not guessed at
- the resolved family is recorded in each sample's `meta`, so `reshape()` parses a response
  back with the same module that wrote its prompt
- `upstream(family)` imports **lazily** — each script pulls in its own SDK (vllm / openai /
  google-genai / anthropic) and only the family in use has to be installed
- `MODEL_SPECIFIC_PROMPT = True`, since the prompt now varies per model

To add a family: vendor its script here and add it to `FAMILY_MODULES`.

`compute_score` and the metric definitions (Acc / Acc@50IoU / avg IoU over MCQ-correct) are
identical across all the scripts; only `benchmark_qwen_vllm.py` factors them into
`calculate_full_metrics`, so that one computes the metrics whichever family generated the
predictions.

## What is NOT reproduced: the retry loop

Upstream re-generates any response whose answer or bbox fails to parse, up to 3 rounds
(`run_benchmark`). We do not, and we do not call `run_benchmark` at all — inference goes
through this repo's normal `infer.py` path.

Reproducing it would mean handing the generation loop to upstream's code, which needs a
custom inference path, an engine shim to put ms-swift behind upstream's vLLM calls, and
sampled decoding (a retry at temperature 0 regenerates the same string, so the loop only
works with sampling — and then scores vary run to run). That was judged too much machinery
for a bound measured at **1.8% of responses** (80/4500 on qwen3vl-4b).

**So our numbers are a slight under-estimate versus the paper's protocol**: an unparseable
response is scored as-is instead of being retried. Worth stating whenever they are compared
to published numbers.

## Settings that still follow upstream

`MultihopSpatialAdapter.INFER_DEFAULTS` keeps two of upstream's choices, because they change
what the model sees rather than how it is scored:

- **image resolution unpinned** — upstream never sets min/max_pixels; our other benchmarks
  pin them to the SpatialScore protocol
- **`max_new_tokens` 8192** — this repo's default is 512

Not followed: upstream reads the checkpoint's `generation_config.json` (sampled decoding);
we generate greedily, like every other benchmark here, so runs are reproducible.

## Re-vendoring

```bash
git clone --depth 1 https://github.com/youngwanLEE/multihopspatial
cp multihopspatial/eval/benchmark_qwen_vllm.py .
```

Then check that the five functions above kept their signatures — `benchmarks/multihopspatial.py`
calls them directly.
