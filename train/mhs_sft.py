#!/usr/bin/env python
"""train/mhs_sft.py — SFT driver (in-process ms-swift) for the MultihopSpatial upper-bound.

Fine-tunes a base VLM on the LEAKAGE-FREE MHS train split (train/prepare_mhs_sft.py output),
in the exact output format the multihopspatial scorer parses, so the resulting checkpoint is
directly evaluable with the normal infer.py / evaluate.py --benchmarks multihopspatial path.
Supports LoRA and full FT (tuner_type). Mirrors infer.py's in-process swift usage:
    infer.py:  infer_main(InferArguments(**kwargs))
    here:      sft_main(SftArguments(**kwargs))      # ms-swift 4.4.1: swift.arguments/pipelines

Usage (inference env; set POST_CRISP_ROOT / BENCH_DATA_DIR / HF_HOME):
    python train/mhs_sft.py --tuner-type lora
    python train/mhs_sft.py --tuner-type full        # full FT -> DeepSpeed ZeRO-3 (+offload)
    python train/mhs_sft.py --tuner-type lora --dry-run   # print kwargs, don't train
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = "multihopspatial/mhs_sft_train.jsonl"     # relative to BENCH_DATA_DIR

# The authors' own training bounds, from their train/train_grpo.sh:
#     --image_min_pixels $((256 * 32 * 32))   --image_max_pixels $((1280 * 32 * 32))
# 32 = patch 16 x merge 2, so these are a floor of 256 and a ceiling of 1280 image tokens.
# NOTE their evaluation script pins nothing (our INFER_DEFAULTS mirrors that with
# pin_pixels=False), so training and eval see different resolutions upstream too — we follow
# each side as published rather than making them agree.
MIN_PIXELS = 256 * 32 * 32     # 262144  -> 256 image tokens
MAX_PIXELS = 1280 * 32 * 32    # 1310720 -> 1280 image tokens


def main() -> int:
    ap = argparse.ArgumentParser(description="SFT a VLM on the MHS train split (ms-swift, in-process).")
    ap.add_argument("--base", default="Qwen/Qwen3.5-9B", help="base model (HF id or local path)")
    ap.add_argument("--model-type", default=None, help="force ms-swift model_type (else auto-detect)")
    ap.add_argument("--tuner-type", choices=["lora", "full"], required=True)
    ap.add_argument("--dataset", default=None,
                    help="SFT jsonl (default: <BENCH_DATA_DIR>/multihopspatial/mhs_sft_train.jsonl)")
    ap.add_argument("--output-dir", default=None,
                    help="ckpt dir (default: <POST_CRISP_ROOT>/sft/mhs-<base>-<tuner>)")
    ap.add_argument("--epochs", type=float, default=10.0,   # authors' train_grpo.sh
                    help="authors use 10; a 1-GPU run will not finish 10 in one 6h job")
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=None,
                    help="default: 5e-5 (lora, the authors' value) / 1e-5 (full)")
    ap.add_argument("--batch-size", type=int, default=1)
    # authors: global batch 128 across 8 GPUs. On one GPU the same global batch means
    # accumulating all 128 — matched here, at the cost of 8x the wall-clock per step.
    ap.add_argument("--grad-accum", type=int, default=128)
    ap.add_argument("--deepspeed", default=None,
                    help="zero2 | zero3 | zero3-offload (default: zero3-offload for full, none for lora)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the newest checkpoint in output_dir if one exists. The "
                         "authors train 10 epochs; one 6h job on a single GPU does not get "
                         "there, so the same job is re-submitted until it does.")
    ap.add_argument("--dry-run", action="store_true", help="print kwargs and exit (no training)")
    args = ap.parse_args()

    bench_data = os.environ.get("BENCH_DATA_DIR", str(REPO / "benchmarks" / "data"))
    dataset = args.dataset or os.path.join(bench_data, DEFAULT_DATASET)
    if not Path(dataset).exists():
        raise SystemExit(f"dataset not found: {dataset}\n  run: python train/prepare_mhs_sft.py")

    root = os.environ.get("POST_CRISP_ROOT", str(REPO))
    tag = args.base.rstrip("/").split("/")[-1].lower()
    output_dir = args.output_dir or os.path.join(root, "sft", f"mhs-{tag}-{args.tuner_type}")

    lr = args.lr if args.lr is not None else (5e-5 if args.tuner_type == "lora" else 1e-5)
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
        add_version=False,          # stable output_dir, so --resume can find the checkpoints
    )
    if args.tuner_type == "lora":
        # authors' LoRA shape (train_grpo.sh): r=64, alpha=64, dropout 0.05, adapters on the
        # language model only. Their lora_namespan_exclude=['visual','lm_head','embed_tokens']
        # is what swift gets from freeze_vit/freeze_aligner + all-linear targeting.
        kwargs.update(lora_rank=64, lora_alpha=64, lora_dropout=0.05,
                      freeze_vit=True, freeze_aligner=True)
    if args.model_type:
        kwargs["model_type"] = args.model_type
    if ds:
        kwargs["deepspeed"] = ds

    if args.resume:
        ckpts = sorted(Path(output_dir).glob("**/checkpoint-*"),
                       key=lambda p: int(p.name.split("-")[-1]))
        if ckpts:
            kwargs["resume_from_checkpoint"] = str(ckpts[-1])
            print(f"[train] resuming from {ckpts[-1]}")
        else:
            print(f"[train] --resume given but no checkpoint under {output_dir}; starting fresh")

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
