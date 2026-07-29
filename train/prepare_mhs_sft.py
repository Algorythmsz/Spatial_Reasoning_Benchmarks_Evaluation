#!/usr/bin/env python
"""train/prepare_mhs_sft.py — build the MultihopSpatial SFT training set (ms-swift format).

Trains on the DISJOINT train split (multihop_train_6791.json), NOT the eval set
(multihop_test_4500.json) -> no leakage. Each record's assistant target is written in the
EXACT format the official evaluator parses:

    Answer: (b) green round container
    Bounding Box: [x1, y1, x2, y2]                       # NORMALIZED xyxy in [0,1]

The train GT bbox is COCO-style [x, y, w, h] in PIXELS + an "image_resolution" (WxH), so we
convert it to normalized xyxy here (the frame the evaluator scores in). Output is one
ms-swift SFT record per line: {"messages": [user, assistant], "images": [abs_path]}.

The PROMPT comes from the vendored official evaluator (build_prompt), so training inputs
match evaluation inputs exactly — if upstream's prompt changes, this follows automatically.
Run in the inference env (huggingface_hub); set BENCH_DATA_DIR to /data2.

Usage:
    BENCH_DATA_DIR=/data2/seungwon/post_crisp/data python train/prepare_mhs_sft.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root -> import benchmarks/

from benchmarks.multihopspatial import (
    MultihopSpatialAdapter,
    upstream,
    DEFAULT_FAMILY,
    HF_REPO,
    IMAGES_SUBDIR,
)

TRAIN_JSON = "data/multihop_train_6791.json"   # disjoint from the eval multihop_test_4500.json
OUT_NAME = "mhs_sft_train.jsonl"


def _norm_xyxy(bbox, wh_str: str) -> list[float]:
    """GT bbox [x, y, w, h] px  +  image_resolution 'WxH'  ->  normalized xyxy [0,1]."""
    if isinstance(bbox, str):
        bbox = json.loads(bbox)
    x, y, w, h = (float(v) for v in bbox)
    W, H = (float(v) for v in wh_str.lower().split("x"))
    return [round(x / W, 4), round(y / H, 4), round((x + w) / W, 4), round((y + h) / H, 4)]


def main() -> int:
    from huggingface_hub import hf_hub_download, snapshot_download

    adapter = MultihopSpatialAdapter()
    root = adapter.data_dir
    root.mkdir(parents=True, exist_ok=True)

    # 1) train json (the adapter downloads only the TEST split; grab TRAIN here)
    tj = root / TRAIN_JSON
    if not tj.exists():
        print(f"[sft-prep] downloading {TRAIN_JSON} ...")
        hf_hub_download(HF_REPO, TRAIN_JSON, repo_type="dataset", local_dir=str(root))

    # 2) images (shared COCO pool; usually already present from `data_preparation.py multihopspatial`)
    imgs = root / IMAGES_SUBDIR
    if not (imgs.is_dir() and any(imgs.iterdir())):
        print("[sft-prep] images missing — downloading ...")
        snapshot_download(HF_REPO, repo_type="dataset", local_dir=str(root),
                          allow_patterns=[f"{IMAGES_SUBDIR}/*"])

    # Official prompt = eval-time prompt. Upstream ships one evaluator per model family and
    # the prompts differ (see benchmarks/multihopspatial.py::FAMILY_MODULES); we SFT Qwen
    # bases, so this is the qwen one — the same module family_for() resolves at eval time.
    build_prompt = upstream(DEFAULT_FAMILY).build_prompt
    image_root = adapter.data_dir / IMAGES_SUBDIR         # adapter._abs()'s root, minus the private call
    train = json.load(open(tj, encoding="utf-8"))
    out = root / OUT_NAME
    n = skipped = missing_img = 0
    with open(out, "w", encoding="utf-8") as f:
        for row in train:
            img, ans, bbox, wh = (row.get(k) for k in ("image_path", "answer", "bbox", "image_resolution"))
            if not (img and ans and bbox and wh):
                skipped += 1
                continue
            try:
                xyxy = _norm_xyxy(bbox, wh)
            except Exception:
                skipped += 1
                continue
            image_abs = str((image_root / img).resolve())
            if not Path(image_abs).exists():
                missing_img += 1
                continue
            target = (f"Answer: {ans}\n"
                      f"Bounding Box: [{xyxy[0]}, {xyxy[1]}, {xyxy[2]}, {xyxy[3]}]")
            rec = {
                "messages": [                                  # upstream's prompt, verbatim
                    {"role": "user",
                     "content": "<image>" + build_prompt(row.get("question") or "")},
                    {"role": "assistant", "content": target},
                ],
                "images": [image_abs],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    print(f"[sft-prep] wrote {n} SFT records -> {out}  "
          f"(skipped {skipped} malformed, {missing_img} missing images)")
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
