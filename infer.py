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


# ── PCIe P2P guard ───────────────────────────────────────────────────────────
def _has_nvlink() -> bool:
    """True if any two visible GPUs are NVLink-connected (`nvidia-smi topo -m` shows NV*)."""
    import subprocess

    try:
        out = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True,
                             text=True, timeout=10)
        if out.returncode != 0:
            return False
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        keep = {int(x) for x in visible.split(",") if x.strip().isdigit()} if visible else None
        lines = out.stdout.splitlines()
        start = next((i for i, ln in enumerate(lines) if "GPU0" in ln), None)
        if start is None:
            return False
        for ln in lines[start + 1:]:
            if not ln.startswith("GPU"):
                break
            cells = ln.split()
            try:
                gpu = int(cells[0].replace("GPU", ""))
            except ValueError:
                continue
            if keep is not None and gpu not in keep:
                continue
            if any(c.startswith("NV") for c in cells[1:]):
                return True
    except Exception:
        pass
    return False


def _guard_nccl_p2p() -> None:
    """Mirrors the official MultihopSpatial evaluator: with no NVLink, vLLM's
    tensor-parallel transport can hang during NCCL setup. Must run before vllm is imported."""
    if "NCCL_P2P_DISABLE" not in os.environ and not _has_nvlink():
        os.environ["NCCL_P2P_DISABLE"] = "1"
        print("[infer] no NVLink detected -> NCCL_P2P_DISABLE=1 (avoids PCIe P2P hangs)")


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
def run_infer(adapter: base.BenchmarkAdapter, model: base.Model, max_new_tokens: int,
              **infer_opts) -> None:
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

    # CUSTOM_INFER: the bench owns its whole inference step. Needed when the protocol has
    # to inspect a response and RE-GENERATE it (MultihopSpatial's retry rounds), which the
    # args-driven infer_main path below cannot express. The adapter owns its generation
    # settings too, so infer.py's defaults deliberately do NOT apply here — including the
    # pixel bounds, which are CLEARED so the model's own default resolution is used (the
    # official MultihopSpatial evaluator never pins them; pinning would change image-token
    # counts and stop the run being a reproduction). ms-swift reads these from the env, so
    # leaving a stale value set would silently apply it.
    if getattr(adapter, "CUSTOM_INFER", False):
        os.environ.pop("MIN_PIXELS", None)
        os.environ.pop("MAX_PIXELS", None)
        _guard_nccl_p2p()                                    # must precede any vllm import
        try:
            n = adapter.run_inference(model, model_path, max_new_tokens, **infer_opts)
        finally:
            restore_env()
            _release_gpu()
        adapter.mark_done(model, n)
        print(f"[infer] done {adapter.name}/{model.tag}: {n} preds -> {preds}")
        return

    if model.min_pixels is not None:
        os.environ["MIN_PIXELS"] = str(model.min_pixels)
    if model.max_pixels is not None:
        os.environ["MAX_PIXELS"] = str(model.max_pixels)     # also passed as max_pixels arg below (same value)

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
    if model.max_pixels is not None:
        kwargs["max_pixels"] = model.max_pixels              # upper bound (env covers the lower bound)
    if model.enable_thinking is not None:                    # e.g. Qwen3.5 -> False for a direct, parseable answer
        kwargs["enable_thinking"] = model.enable_thinking
    if model.vllm_max_model_len is not None:                 # cap KV cache: model config default (e.g. 262144) OOMs
        kwargs["vllm_max_model_len"] = model.vllm_max_model_len
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
    adapter.mark_done(model, n)                               # done.flag -> evaluate.py's is_complete gate
    print(f"[infer] done {adapter.name}/{model.tag}: {n} preds -> {preds}")


def _count_lines(p: Path) -> int:
    with open(p, "rb") as f:
        return sum(1 for _ in f)


# ── entry point ──────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="Run inference (ms-swift) over models.yaml.")
    ap.add_argument("--benchmarks", help="comma-separated bench names, or 'all' for every bench")
    ap.add_argument("--models", help="comma-separated model tags from models.yaml, or 'all' for every model")
    ap.add_argument("--max-new-tokens", type=int, default=512, help="generation budget (default 512)")
    # CUSTOM_INFER knobs — forwarded to adapter.run_inference(); benches on the infer_main
    # path ignore them entirely. Today only multihopspatial (the vendored official evaluator).
    ap.add_argument("--greedy", action="store_true",
                    help="force temperature=0; deterministic, but disables retry rounds")
    ap.add_argument("--temperature", type=float, default=None,
                    help="explicit temperature (default: the checkpoint's generation_config.json)")
    ap.add_argument("--max-retries", type=int, default=None,
                    help="rounds of re-generation for invalid outputs (default: the protocol's 3)")
    ap.add_argument("--seed", type=int, default=None, help="sampling seed")
    ap.add_argument("--test-samples", type=int, default=None,
                    help="only run the first N samples (smoke test)")
    args = ap.parse_args()
    infer_opts = {k: v for k, v in (("greedy", args.greedy), ("temperature", args.temperature),
                                    ("max_retries", args.max_retries), ("seed", args.seed),
                                    ("test_samples", args.test_samples))
                  if v not in (None, False)}

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
                run_infer(adapter, model, args.max_new_tokens, **infer_opts)
            except Exception as e:
                print(f"[infer] FAIL {adapter.name}/{model.tag}: {type(e).__name__}: {e}")
                failures.append(f"{adapter.name}/{model.tag}")

    if failures:
        print(f"[infer] failures: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
