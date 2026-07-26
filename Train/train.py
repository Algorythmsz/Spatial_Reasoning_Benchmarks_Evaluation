#!/usr/bin/env python
"""Train/train.py — SFT driver (in-process ms-swift) for the MultihopSpatial upper-bound.

Fine-tunes a base VLM on the LEAKAGE-FREE MHS train split (Train/prepare_mhs_sft.py output),
in the exact output format the multihopspatial scorer parses, so the resulting checkpoint is
directly evaluable with the normal infer.py / evaluate.py --benchmarks multihopspatial path.
Supports LoRA and full FT (tuner_type). Mirrors infer.py's in-process swift usage:
    infer.py:  infer_main(InferArguments(**kwargs))
    here:      sft_main(SftArguments(**kwargs))      # ms-swift 4.4.1: swift.arguments/pipelines

Usage (inference env; set POST_CRISP_ROOT / BENCH_DATA_DIR / HF_HOME):
    python Train/train.py --tuner-type lora
    python Train/train.py --tuner-type full          # 9B full FT -> DeepSpeed ZeRO-3 (+offload)
    python Train/train.py --tuner-type lora --dry-run   # print kwargs, don't train
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = "multihopspatial/mhs_sft_train.jsonl"     # relative to BENCH_DATA_DIR

# SpatialScore test_qwen protocol pixel bounds — match inference so image-token counts stay
# consistent between SFT and eval (infer.py sets the same via MIN/MAX_PIXELS env + max_pixels).
MIN_PIXELS = 200704       # 256*28*28
MAX_PIXELS = 2007040      # 2560*28*28


def main() -> int:
    ap = argparse.ArgumentParser(description="SFT a VLM on the MHS train split (ms-swift, in-process).")
    ap.add_argument("--base", default="Qwen/Qwen3.5-9B", help="base model (HF id or local path)")
    ap.add_argument("--model-type", default=None, help="force ms-swift model_type (else auto-detect)")
    ap.add_argument("--tuner-type", choices=["lora", "full"], required=True)
    ap.add_argument("--dataset", default=None,
                    help="SFT jsonl (default: <BENCH_DATA_DIR>/multihopspatial/mhs_sft_train.jsonl)")
    ap.add_argument("--output-dir", default=None,
                    help="ckpt dir (default: <POST_CRISP_ROOT>/sft/mhs-<base>-<tuner>)")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=None, help="default: 1e-4 (lora) / 1e-5 (full)")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--deepspeed", default=None,
                    help="zero2 | zero3 | zero3-offload (default: zero3-offload for full, none for lora)")
    ap.add_argument("--dry-run", action="store_true", help="print kwargs and exit (no training)")
    args = ap.parse_args()

    bench_data = os.environ.get("BENCH_DATA_DIR", str(REPO / "benchmarks" / "data"))
    dataset = args.dataset or os.path.join(bench_data, DEFAULT_DATASET)
    if not Path(dataset).exists():
        raise SystemExit(f"dataset not found: {dataset}\n  run: python Train/prepare_mhs_sft.py")

    root = os.environ.get("POST_CRISP_ROOT", str(REPO))
    tag = args.base.rstrip("/").split("/")[-1].lower()
    output_dir = args.output_dir or os.path.join(root, "sft", f"mhs-{tag}-{args.tuner_type}")

    lr = args.lr if args.lr is not None else (1e-4 if args.tuner_type == "lora" else 1e-5)
    # 9B full FT on 2x48GB needs params+optimizer offloaded to CPU RAM (node has ~245G) -> zero3-offload.
    ds = args.deepspeed or ("zero3-offload" if args.tuner_type == "full" else None)

    # pixel bounds via env (template reads these) + max_pixels kwarg, same as infer.py
    os.environ.setdefault("USE_HF", "1")
    os.environ["MIN_PIXELS"] = str(MIN_PIXELS)
    os.environ["MAX_PIXELS"] = str(MAX_PIXELS)

    kwargs = dict(
        model=args.base,
        tuner_type=args.tuner_type,
        dataset=[dataset],
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        max_length=args.max_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=lr,
        gradient_checkpointing=True,
        warmup_ratio=0.03,
        logging_steps=5,
        save_strategy="epoch",
        max_pixels=MAX_PIXELS,
        use_hf=True,
        dataloader_num_workers=4,
        dataset_num_proc=4,
    )
    if args.tuner_type == "lora":
        kwargs.update(lora_rank=16, lora_alpha=32)          # target_modules: swift default (all-linear); freeze_vit default True
    if args.model_type:
        kwargs["model_type"] = args.model_type
    if ds:
        kwargs["deepspeed"] = ds

    print(f"[train] MHS SFT | base={args.base} tuner={args.tuner_type} deepspeed={ds} epochs={args.epochs}")
    print(f"[train] dataset={dataset}")
    print(f"[train] output_dir={output_dir}")
    print(f"[train] kwargs={kwargs}")
    if args.dry_run:
        return 0

    os.makedirs(output_dir, exist_ok=True)
    from swift.arguments import SftArguments
    from swift.pipelines import sft_main

    sft_main(SftArguments(**kwargs))                        # training! raises on failure
    print(f"[train] done -> {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
