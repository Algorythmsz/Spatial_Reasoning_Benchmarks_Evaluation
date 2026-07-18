#!/usr/bin/env python
"""test.py — inference runner (ms-swift).

Drives the INFERENCE half of the pipeline (env1). For each (benchmark, model):
  1) adapter.preprocess()   — ensure the model-agnostic ms-swift jsonl exists (idempotent)
  2) resolve model path     — HF repo id / local dir / repo-id+subfolder (SFT ckpts)
  3) swift infer            — run over the jsonl, writing preds jsonl to preds_path(model)
  4) adapter.mark_done()    — write done.flag with the expected sample count

Per-model settings from models.yaml are injected here (NOT in the adapter, which stays
model-agnostic): min/max_pixels (SpatialScore test_qwen protocol) via MIN_PIXELS/MAX_PIXELS
env + --max_pixels, enable_thinking via --enable_thinking, backend via --infer_backend.
`--remove_unused_columns false` keeps our id/meta columns in the preds (reshape needs them).

HF cache: repo ids need `USE_HF=1` (else ms-swift hits ModelScope) and, if your cache
isn't in the default location, `HF_HOME` (see README). test.py sets USE_HF if unset.

Usage:
    conda activate <swift env>
    python test.py --benchmarks spatialscore --models qwen3.5-27b,qwen3vl-8b
    python test.py --all --models qwen3.5-27b
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from benchmarks import base  # importing the package registers all adapters

MODELS_YAML = Path(os.environ.get("MODELS_YAML", Path(__file__).resolve().parent / "models.yaml"))
SWIFT_BIN = os.environ.get("SWIFT_BIN", "swift")  # override if `swift` isn't on PATH


# ── models.yaml -> [Model] ───────────────────────────────────────────────────
def load_models(tags: list[str] | None = None) -> list[base.Model]:
    import yaml  # lazy

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
    from huggingface_hub import snapshot_download             # lazy (env1 has it)
    root = snapshot_download(model.path, allow_patterns=[f"{model.subfolder}/*"])
    ckpt = Path(root) / model.subfolder
    if not (ckpt / "config.json").exists():
        raise FileNotFoundError(f"resolved subfolder has no config.json: {ckpt}")
    return str(ckpt)


# ── run inference for one (adapter, model) ───────────────────────────────────
def run_infer(adapter: base.BenchmarkAdapter, model: base.Model, max_new_tokens: int) -> None:
    val = adapter.preprocess()                                # model-agnostic ms-swift jsonl (idempotent)
    preds = adapter.preds_path(model)                         # where swift writes results for this (model, bench)
    preds.parent.mkdir(parents=True, exist_ok=True)
    if preds.exists():                                        # start clean so the line count reflects this run
        preds.unlink()

    model_path = resolve_model_path(model)                    # concrete path/id for --model

    env = os.environ.copy()                                   # child inherits our env...
    env.setdefault("USE_HF", "1")                            # ...ensure HF hub (not ModelScope) for repo ids
    if model.min_pixels is not None:                          # Qwen-VL smart_resize bounds (test_qwen protocol);
        env["MIN_PIXELS"] = str(model.min_pixels)            # min has no CLI flag -> env is the only channel
    if model.max_pixels is not None:
        env["MAX_PIXELS"] = str(model.max_pixels)            # also passed as --max_pixels below (same value)

    cmd = [
        SWIFT_BIN, "infer",
        "--model", model_path,                               # repo id / local dir
        "--infer_backend", model.backend,                    # vllm | pt
        "--val_dataset", str(val),                           # our preprocessed jsonl
        "--result_path", str(preds),                         # preds jsonl output
        "--remove_unused_columns", "false",                  # ★ keep id/meta columns for reshape/scoring
        "--max_new_tokens", str(max_new_tokens),             # generation budget
        "--temperature", "0",                                # greedy (matches test_qwen)
        "--use_hf", "true",                                  # resolve repo ids from the HF hub/cache
    ]
    if model.max_pixels is not None:
        cmd += ["--max_pixels", str(model.max_pixels)]       # CLI upper bound (env covers the lower bound)
    if model.enable_thinking is not None:                    # e.g. Qwen3.5 -> false for a direct, parseable answer
        cmd += ["--enable_thinking", str(model.enable_thinking).lower()]
    if model.vllm_max_model_len is not None:                 # cap KV cache: model config default (e.g. 262144) OOMs
        cmd += ["--vllm_max_model_len", str(model.vllm_max_model_len)]
    # tensor parallel: honor an explicit models.yaml value, else split across every
    # GPU SLURM gave us (CUDA_VISIBLE_DEVICES). Without this a 2-GPU job loads on ONE
    # card (TP=1), starving the KV cache -> big models OOM even at a sane max_model_len.
    tp = model.vllm_tensor_parallel_size
    if tp is None:
        tp = len([d for d in env.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()])
    if tp and tp > 1:
        cmd += ["--vllm_tensor_parallel_size", str(tp)]

    print(f"[test] infer {adapter.name}/{model.tag}: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)                 # raises on non-zero -> mark_done skipped (crash-safe)

    n = _count_lines(val)                                     # expected sample count = input rows
    got = _count_lines(preds) if preds.exists() else 0
    if got != n:                                             # surface a short/absent output before marking done
        raise RuntimeError(f"{adapter.name}/{model.tag}: preds has {got} lines, expected {n}")
    adapter.mark_done(model, n)                               # done.flag -> evaluate.py's is_complete gate
    print(f"[test] done {adapter.name}/{model.tag}: {n} preds -> {preds}")


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
    adapters = base.resolve(names, args.all)                  # selected adapters (or error if none specified)
    if not args.models:
        raise SystemExit("specify --models (comma-separated tags from models.yaml).")
    models = load_models(args.models.split(","))             # selected models

    failures: list[str] = []
    for model in models:
        for adapter in adapters:
            if adapter.is_complete(model):                   # already inferred cleanly -> skip (resume-friendly)
                print(f"[test] skip {adapter.name}/{model.tag}: already complete")
                continue
            try:
                run_infer(adapter, model, args.max_new_tokens)
            except Exception as e:
                print(f"[test] FAIL {adapter.name}/{model.tag}: {type(e).__name__}: {e}")
                failures.append(f"{adapter.name}/{model.tag}")

    if failures:
        print(f"[test] failures: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
