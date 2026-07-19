#!/usr/bin/env python
"""infer.py — inference runner (ms-swift).

Runs in the inference env, driving the INFERENCE half of the pipeline. For each
(benchmark, model):
  1) adapter.preprocess()   — ensure the model-agnostic ms-swift jsonl exists (idempotent)
  2) resolve model path     — HF repo id / local dir / repo-id+subfolder (SFT ckpts)
  3) infer_main()           — run over the jsonl IN-PROCESS (no CLI/subprocess), writing
                              the preds jsonl to preds_path(model)
  4) adapter.mark_done()    — write done.flag with the expected sample count

Inference calls ms-swift as a LIBRARY: `swift.pipelines.infer_main(InferArguments(...))`
— exactly the code path the `swift infer` CLI runs, so the preds jsonl schema is
identical (reshape/scoring unchanged). No `swift` binary / subprocess is spawned.

Per-model settings from models.yaml are injected here (NOT in the adapter, which stays
model-agnostic): min/max_pixels (SpatialScore test_qwen protocol) via MIN_PIXELS/MAX_PIXELS
env + max_pixels arg, enable_thinking, backend via infer_backend.
`remove_unused_columns=False` keeps our id/meta columns in the preds (reshape needs them).

⚠️ In-process means no per-model GPU isolation: running several models in one invocation
loads each into the same process. We do best-effort teardown (gc + empty_cache + vllm
parallel-state destroy) between models, but vllm doesn't always fully release. If a
multi-model run OOMs, run one model per invocation (each process starts/exits clean).

HF cache: repo ids need `USE_HF=1` (else ms-swift hits ModelScope) and, if your cache
isn't in the default location, `HF_HOME` (see README). infer.py sets USE_HF if unset.

Usage:
    conda activate <swift env>
    python infer.py --benchmarks spatialscore --models qwen3.5-27b,qwen3vl-8b
    python infer.py --all --models qwen3.5-27b
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from benchmarks import base  # importing the package registers all adapters

MODELS_YAML = Path(os.environ.get("MODELS_YAML", Path(__file__).resolve().parent / "models.yaml"))


# ── models.yaml -> [Model] ───────────────────────────────────────────────────
def load_models(tags: list[str] | None = None) -> list[base.Model]:
    import yaml

    spec = yaml.safe_load(MODELS_YAML.read_text())
    models = [base.Model.from_dict(m) for m in spec.get("models", [])]
    if tags:                                                   # keep only requested tags, preserving request order
        by_tag = {m.tag: m for m in models}
        missing = [t for t in tags if t not in by_tag]
        if missing:
            raise SystemExit(f"unknown model tag(s): {missing}. known: {sorted(by_tag)}")
        models = [by_tag[t] for t in tags]
    return models


# ── resolve a model to a concrete path swift can load ────────────────────────
def resolve_model_path(model: base.Model) -> str:
    """HF repo id / local dir pass through; repo-id + subfolder is downloaded to a local dir."""
    if not model.subfolder:
        return model.path                                     # repo id (USE_HF resolves) or absolute local path
    # subfolder lives inside an HF repo (e.g. SFT ckpts under haoningwu/SpatialScore) ->
    # fetch just that subfolder and hand swift the concrete checkpoint dir.
    from huggingface_hub import snapshot_download             # lazy (the inference env has it)
    root = snapshot_download(model.path, allow_patterns=[f"{model.subfolder}/*"])
    ckpt = Path(root) / model.subfolder
    if not (ckpt / "config.json").exists():
        raise FileNotFoundError(f"resolved subfolder has no config.json: {ckpt}")
    return str(ckpt)


# ── best-effort GPU release between in-process models ────────────────────────
def _release_gpu() -> None:
    """Free GPU memory after an infer_main() call so the next model can load.

    In-process runs don't get the clean slate a per-model subprocess did. This is
    best-effort: it drops references, collects, empties the torch cache, and tears down
    vllm's parallel state. vllm may still not release everything (CUDA context / graph
    memory); if a multi-model run OOMs, run one model per invocation. All imports are
    lazy + guarded so this is a no-op when torch/vllm aren't the active stack.
    """
    import gc

    gc.collect()
    try:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        pass                                                 # not vllm / already torn down
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass                                                 # no torch / no CUDA


# ── run inference for one (adapter, model) ───────────────────────────────────
def run_infer(adapter: base.BenchmarkAdapter, model: base.Model, max_new_tokens: int) -> None:
    val = adapter.preprocess()                                # model-agnostic ms-swift jsonl (idempotent)
    preds = adapter.preds_path(model)
    preds.parent.mkdir(parents=True, exist_ok=True)
    if preds.exists():                                        # start clean so the line count reflects this run
        preds.unlink()

    model_path = resolve_model_path(model)

    # min/max_pixels are Qwen-VL smart_resize bounds (test_qwen protocol) read from the
    # environment by the template at inference time — no InferArguments field for min, so
    # env is the channel for both. We're in-process, so mutate os.environ (restored in the
    # finally below) instead of handing a copied env to a child.
    os.environ.setdefault("USE_HF", "1")                 
    saved_env = {k: os.environ.get(k) for k in ("MIN_PIXELS", "MAX_PIXELS")}
    if model.min_pixels is not None:
        os.environ["MIN_PIXELS"] = str(model.min_pixels)
    if model.max_pixels is not None:
        os.environ["MAX_PIXELS"] = str(model.max_pixels)     # also passed as max_pixels arg below (same value)

    # Build InferArguments == the flags the `swift infer` CLI would have parsed. Field
    # names match the CLI flags 1:1, so the run (and preds schema) is identical.
    kwargs = dict(
        model=model_path,
        infer_backend=model.backend,                         # vllm | pt
        val_dataset=[str(val)],                              # list form, like the CLI
        result_path=str(preds),
        remove_unused_columns=False,                         # ★ keep id/meta columns for reshape/scoring
        max_new_tokens=max_new_tokens,
        temperature=0.0,                                     # greedy (matches test_qwen)
        use_hf=True,                                         # HF hub/cache, not ModelScope
        vllm_max_num_seqs=128,                               # 256->128: cap concurrent seqs -> lower host-RAM peak
        write_batch_size=200,                                # 1000->200: smaller per-shard decode -> lower host-RAM peak
    )
    if model.model_type is not None:                         # FT ckpts (etri/sft) match multiple swift types -> force it
        kwargs["model_type"] = model.model_type
    if model.max_pixels is not None:
        kwargs["max_pixels"] = model.max_pixels              # CLI upper bound (env covers the lower bound)
    if model.enable_thinking is not None:                    # e.g. Qwen3.5 -> False for a direct, parseable answer
        kwargs["enable_thinking"] = model.enable_thinking
    if model.vllm_max_model_len is not None:                 # cap KV cache: model config default (e.g. 262144) OOMs
        kwargs["vllm_max_model_len"] = model.vllm_max_model_len
    tp = model.vllm_tensor_parallel_size
    if tp is None:                                           # else split across every GPU SLURM gave us
        tp = len([d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()])
    if tp and tp > 1:
        kwargs["vllm_tensor_parallel_size"] = tp

    # Lazy import: swift + vllm are heavy, and only the inference env has them.
    from swift.arguments import InferArguments
    from swift.pipelines import infer_main

    print(f"[infer] infer {adapter.name}/{model.tag}: infer_main({kwargs})")
    try:
        infer_main(InferArguments(**kwargs))                 # raises on failure -> mark_done skipped (crash-safe)
    finally:
        for name, prev in saved_env.items():                 # restore pixel env so the next model isn't polluted
            if prev is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = prev
        _release_gpu()

    n = _count_lines(val)                                     # expected sample count = input rows
    got = _count_lines(preds) if preds.exists() else 0
    if got != n:                                             # surface a short/absent output before marking done
        raise RuntimeError(f"{adapter.name}/{model.tag}: preds has {got} lines, expected {n}")
    adapter.mark_done(model, n)                               # done.flag -> evaluate.py's is_complete gate
    print(f"[infer] done {adapter.name}/{model.tag}: {n} preds -> {preds}")


def _count_lines(p: Path) -> int:
    with open(p, "rb") as f:
        return sum(1 for _ in f)


# ── entry point ──────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="Run inference (ms-swift) over models.yaml.")
    ap.add_argument("--benchmarks", help="comma-separated bench names")
    ap.add_argument("--all", action="store_true", help="run every registered bench")
    ap.add_argument("--models", help="comma-separated model tags from models.yaml")
    ap.add_argument("--max-new-tokens", type=int, default=512, help="generation budget (default 512)")
    args = ap.parse_args()

    names = args.benchmarks.split(",") if args.benchmarks else None
    adapters = base.resolve(names, args.all)                  # errors if neither --benchmarks nor --all given
    if not args.models:
        raise SystemExit("specify --models (comma-separated tags from models.yaml).")
    models = load_models(args.models.split(","))

    failures: list[str] = []
    for model in models:
        for adapter in adapters:
            if adapter.is_complete(model):                   # already inferred cleanly -> skip (resume-friendly)
                print(f"[infer] skip {adapter.name}/{model.tag}: already complete")
                continue
            try:
                run_infer(adapter, model, args.max_new_tokens)
            except Exception as e:
                print(f"[infer] FAIL {adapter.name}/{model.tag}: {type(e).__name__}: {e}")
                failures.append(f"{adapter.name}/{model.tag}")

    if failures:
        print(f"[infer] failures: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
