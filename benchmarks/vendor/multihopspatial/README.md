# Vendored: official MultihopSpatial evaluator

`benchmark_qwen_vllm.py` is a **byte-identical copy** of `eval/benchmark_qwen_vllm.py` from
[youngwanLEE/multihopspatial](https://github.com/youngwanLEE/multihopspatial) @ `ab711b8`.
Upstream's Apache-2.0 `LICENSE` sits next to it.

It is the whole protocol: prompt, retry rounds, answer/bbox parsing, coordinate scaling,
IoU, and the hop × view metric table. **Treat it as read-only** — every part of this that
we previously guessed at turned out to differ from the real thing.

## How ms-swift gets plugged in

`swift_backend.py` (ours) does not edit the file. Upstream reaches vLLM through exactly two
module-level names, so we rebind them:

```python
upstream.LLM = swift_backend.LLM                  # wraps swift.VllmEngine
upstream.SamplingParams = swift_backend.SamplingParams   # -> swift.RequestConfig
```

`swift_backend.install()` does this and returns the module. It also converts upstream's
OpenAI-style message parts into `swift.InferRequest` (`<image>` placeholders + an images
list) and adapts the response objects back to `output.outputs[0].text`.

Two things the official script has no notion of, because it is Qwen3-VL-only, are injected
via `swift_backend.configure()`:

| | why |
|---|---|
| `model_type` | fine-tuned checkpoints match several ms-swift types; auto-detection fails |
| `enable_thinking` | a `Template` ctor arg; Qwen3.5 needs `False` or it emits reasoning traces |

## Protocol settings that differ from our other benchmarks

Deliberate — these are upstream's, and matching them is the point:

| | here | our other benches |
|---|---|---|
| `min/max_pixels` | **not set** (model default) | pinned to the SpatialScore protocol |
| decoding | the checkpoint's `generation_config.json` (Qwen3-VL-4B: t=0.7) | greedy |
| `max_new_tokens` | 8192 | 512 |
| `max_model_len` | 32768 | 86016 |
| retry | up to 3 rounds on an unparseable answer or bbox | none |

Consequences worth remembering:

- **Scores vary run to run.** Sampling is on, and retries change batch composition, so a
  fixed seed does not fully pin the result. `--greedy` restores determinism but disables
  the retry rounds (a retry at temperature 0 regenerates the same string), so it is no
  longer the paper's protocol.
- **Retry is a selection procedure.** Only invalid responses are resampled, and a retry
  regenerates the whole response — the answer letter included. Scores are therefore "best
  of ≤3 attempts filtered by format validity", which favours models with messy output
  formats. That is upstream's design, not a bug, but it is not sampling accuracy.
- The model path must be a **local directory**: upstream only reads
  `generation_config.json` when `os.path.isdir(model_path)`, so an HF repo id silently
  downgrades sampling to greedy. `benchmarks/multihopspatial.py::_local_model_dir`
  resolves this before the call.

## Re-vendoring

```bash
git clone --depth 1 https://github.com/youngwanLEE/multihopspatial
cp multihopspatial/eval/benchmark_qwen_vllm.py .
```

Then re-check that `swift_backend.py` still only needs `LLM` / `SamplingParams` rebound,
and that `run_benchmark(...)`'s signature and `calculate_full_metrics`'s output keys are
unchanged — `benchmarks/multihopspatial.py` calls both.
