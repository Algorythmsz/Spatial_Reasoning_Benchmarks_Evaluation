# Vendored: official MultihopSpatial evaluator

`benchmark_qwen_vllm.py` is a **byte-identical copy** of `eval/benchmark_qwen_vllm.py` from
[youngwanLEE/multihopspatial](https://github.com/youngwanLEE/multihopspatial) @ `ab711b8`.
Upstream's Apache-2.0 `LICENSE` sits next to it. **Treat it as read-only** — every part of
this protocol that we previously guessed at turned out to differ from the real thing.

`benchmarks/multihopspatial.py` imports the pieces that define what a score means and uses
them unchanged:

| imported | what it fixes |
|---|---|
| `build_prompt` | the prompt text and layout (question first, then the format block) |
| `parse_response` | answer letter + bbox, including the per-box `any(v>1) -> /1000` scale rule |
| `calculate_iou` | COCO xywh GT -> pixel-space IoU |
| `compute_score` | MCQ correctness |
| `calculate_full_metrics` | Acc / Acc@50IoU / avg IoU, and their denominators |

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
