#!/usr/bin/env python
"""Train/prepare_mhs_sft.py — build the MultihopSpatial SFT training set (ms-swift format).

Trains on the DISJOINT train split (multihop_train_6791.json), NOT the eval set
(multihop_test_4500.json) -> no leakage. Each record's assistant target is written in the
EXACT format the multihopspatial scorer parses:

    Answer: (b) green round container
    Bounding Box: {"bbox_2d": [x1, y1, x2, y2]}          # NORMALIZED xyxy in [0,1]

The train GT bbox is COCO-style [x, y, w, h] in PIXELS + an "image_resolution" (WxH), so we
convert it to normalized xyxy here (the same frame the scorer/prompt use). Output is one
ms-swift SFT record per line: {"messages": [system, user, assistant], "images": [abs_path]}.

Reuses benchmarks.multihopspatial (SYSTEM_PROMPT / data_dir / _abs) so the SFT format tracks
the eval harness. Run in the inference env (huggingface_hub); set BENCH_DATA_DIR to /data2.

Usage:
    BENCH_DATA_DIR=/data2/seungwon/post_crisp/data python Train/prepare_mhs_sft.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root -> import benchmarks/

from benchmarks.multihopspatial import (
    MultihopSpatialAdapter,
    SYSTEM_PROMPT,
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
            image_abs = adapter._abs(img)
            if not Path(image_abs).exists():
                missing_img += 1
                continue
            target = (f"Answer: {ans}\n"
                      f'Bounding Box: {{"bbox_2d": [{xyxy[0]}, {xyxy[1]}, {xyxy[2]}, {xyxy[3]}]}}')
            rec = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "<image>" + (row.get("question") or "")},
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
