from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from benchmarks import base  # importing the package registers all adapters

# ── resolve a model to a concrete path swift can load ────────────────────────
def resolve_model_path(model: base.Model) -> str:
    if not model.subfolder:
        return model.path                                     # repo id (USE_HF resolves) or absolute local path
    
    from huggingface_hub import snapshot_download             
    root = snapshot_download(model.path, allow_patterns=[f"{model.subfolder}/*"])
    ckpt = Path(root) / model.subfolder
    if not (ckpt / "config.json").exists():
        raise FileNotFoundError(f"resolved subfolder has no config.json: {ckpt}")
    return str(ckpt)


# ── best-effort GPU release between in-process models ────────────────────────
def _release_gpu() -> None:
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
def run_infer(adapter: base.BenchmarkAdapter, model: base.Model, max_new_tokens: int | None) -> None:
    val = adapter.preprocess(model)                           # ensure the input jsonl (per-model when the bench bakes a model-specific prompt)
    preds = adapter.preds_path(model)
    preds.parent.mkdir(parents=True, exist_ok=True)
    if preds.exists():
        preds.unlink()

    model_path = resolve_model_path(model)

    os.environ.setdefault("USE_HF", "1")                     # HF hub/cache
    saved_env = {k: os.environ.get(k) for k in ("MIN_PIXELS", "MAX_PIXELS")}
    restore_env = lambda: [os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
                           for k, v in saved_env.items()]

    # Per-benchmark inference settings (base.BenchmarkAdapter.INFER_DEFAULTS). A benchmark
    # whose official harness generates differently from this repo's defaults declares it
    # there, so the numbers stay comparable to the harness we reproduce.
    defaults = getattr(adapter, "INFER_DEFAULTS", {}) or {}
    max_new_tokens = max_new_tokens or defaults.get("max_new_tokens") or 512
    pin_pixels = defaults.get("pin_pixels", True)
    # A bench whose official protocol specifies its own image budget overrides models.yaml.
    min_px = defaults.get("min_pixels", model.min_pixels)
    max_px = defaults.get("max_pixels", model.max_pixels)

    if pin_pixels:                                           # ms-swift reads these from the env
        if min_px is not None:
            os.environ["MIN_PIXELS"] = str(min_px)
        if max_px is not None:
            os.environ["MAX_PIXELS"] = str(max_px)            # also passed as max_pixels arg below
    else:                                                    # bench pins nothing -> model default
        os.environ.pop("MIN_PIXELS", None)
        os.environ.pop("MAX_PIXELS", None)

    kwargs = dict(
        model=model_path,
        infer_backend=model.backend,                         # vllm | pt
        val_dataset=[str(val)],
        result_path=str(preds),
        remove_unused_columns=False,                         # keep id/meta columns for reshape/scoring
        max_new_tokens=max_new_tokens,
        temperature=0.0,                                     # greedy (matches test_qwen)
        use_hf=True,                                         # HF hub/cache
        vllm_max_num_seqs=128,                               # 256->128: cap concurrent seqs -> lower host-RAM peak
        write_batch_size=200,                                # 1000->200: smaller per-shard decode -> lower host-RAM peak
    )
    if model.model_type is not None:                         # FT ckpts (etri/sft) match multiple swift types -> force it
        kwargs["model_type"] = model.model_type
    if pin_pixels and max_px is not None:
        kwargs["max_pixels"] = max_px                        # upper bound (env covers the lower bound)
    if model.enable_thinking is not None:                    # e.g. Qwen3.5 -> False for a direct, parseable answer
        kwargs["enable_thinking"] = model.enable_thinking
    if model.vllm_max_model_len is not None:                 # cap KV cache: model config default (e.g. 262144) OOMs
        kwargs["vllm_max_model_len"] = model.vllm_max_model_len
    if model.vllm_engine_kwargs:                             # raw EngineArgs pass-through (see models.yaml)
        kwargs["vllm_engine_kwargs"] = model.vllm_engine_kwargs
    tp = model.vllm_tensor_parallel_size
    if tp is None:                                           # else split across every visible GPU (CUDA_VISIBLE_DEVICES)
        tp = len([d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()])
    if tp and tp > 1:
        kwargs["vllm_tensor_parallel_size"] = tp

    from swift.arguments import InferArguments
    from swift.pipelines import infer_main

    print(f"[infer] infer {adapter.name}/{model.tag}: infer_main({kwargs})")
    try:
        infer_main(InferArguments(**kwargs))                 # inference code! Raises on failure -> mark_done skipped (crash-safe)
    finally:
        restore_env()                                        # restore pixel env so the next model isn't polluted
        _release_gpu()                                       # reclaim GPU before the next model loads

    n = _count_lines(val)                                     # expected sample count = input rows
    got = _count_lines(preds) if preds.exists() else 0
    if got != n:                                             # surface a short/absent output before marking done
        raise RuntimeError(f"{adapter.name}/{model.tag}: preds has {got} lines, expected {n}")

    # What actually produced these predictions. Written into done.json so a reader (and
    # is_complete's stale-input guard) can tell which protocol the numbers came from.
    run_config = {
        "benchmark": adapter.name,
        "model_tag": model.tag,
        "model_path": model_path,                             # resolved (subfolder ckpts included)
        "model_type": model.model_type,
        "backend": model.backend,
        "enable_thinking": model.enable_thinking,
        "max_new_tokens": max_new_tokens,
        "temperature": 0.0,                                   # greedy, every bench
        "pin_pixels": pin_pixels,
        "min_pixels": min_px if pin_pixels else None,             # None -> model default
        "max_pixels": max_px if pin_pixels else None,
        "vllm_max_model_len": model.vllm_max_model_len,
        "vllm_tensor_parallel_size": tp,
        "vllm_engine_kwargs": model.vllm_engine_kwargs,
        "val_dataset": str(val),
        "input_fingerprint": adapter.input_fingerprint(model),  # md5 of the input records
        "versions": _versions(),
    }
    adapter.mark_done(model, n, run_config)                   # done.flag -> evaluate.py's is_complete gate
    print(f"[infer] done {adapter.name}/{model.tag}: {n} preds -> {preds}")


def _count_lines(p: Path) -> int:
    with open(p, "rb") as f:
        return sum(1 for _ in f)


def _versions() -> dict[str, str]:
    """Library versions that can move the numbers. Best-effort: never fail a finished run."""
    out = {}
    for pkg in ("ms-swift", "vllm", "transformers", "torch"):
        try:
            from importlib.metadata import version
            out[pkg] = version(pkg)
        except Exception:
            pass
    return out


# ── entry point ──────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="Run inference (ms-swift) over models.yaml.")
    ap.add_argument("--benchmarks", help="comma-separated bench names, or 'all' for every bench")
    ap.add_argument("--models", help="comma-separated model tags from models.yaml, or 'all' for every model")
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="generation budget. Omit to use each benchmark's own default "
                         "(512 on the infer_main path, upstream's 8192 for multihopspatial).")
    args = ap.parse_args()

    names = args.benchmarks.split(",") if args.benchmarks else None
    adapters = base.resolve(names)                            # errors if --benchmarks not given
    if not args.models:
        raise SystemExit("specify --models (comma-separated tags from models.yaml, or 'all').")
    tags = None if args.models == "all" else args.models.split(",")  # 'all' -> every model in models.yaml
    models = base.load_models(tags)

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
