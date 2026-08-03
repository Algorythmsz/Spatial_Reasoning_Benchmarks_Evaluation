#!/usr/bin/env python
"""train/prepare_mhs_sft.py — build the MultihopSpatial SFT training set (ms-swift format).

Trains on the DISJOINT train split (multihop_train_6791.json), NOT the eval set
(multihop_test_4500.json) -> no leakage. Each record's assistant target is written in the
EXACT format the official evaluator parses:

    Answer: (b) green round container
    Bounding Box: [876, 506, 940, 604]                   # xyxy on Qwen's 0-1000 scale

The train GT bbox is COCO-style [x, y, w, h] in PIXELS + an "image_resolution" (WxH), so we
convert it to 0-1000 xyxy here, matching the authors' own SFT conversion (their
train/src/dataset/data_utils.py). The evaluator's parser reads 0-1000 natively (any value > 1
is divided by 1000), so this is the space it scores in. Output is one
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
    HF_REPO,
    IMAGES_SUBDIR,
)

TRAIN_JSON = "data/multihop_train_6791.json"   # disjoint from the eval multihop_test_4500.json
OUT_NAME = "mhs_sft_train.jsonl"

# MSR_RESPONSE_FORMAT, verbatim from the authors' src/dataset/grpo_dataset.py. Same as the
# evaluator's build_prompt block plus the last line, which the eval prompt does not carry.
MSR_RESPONSE_FORMAT = """

Please respond in the following format:
Answer: (your choice, e.g., "(a) object name")
Bounding Box: [x1, y1, x2, y2]

where [x1, y1] is the top-left corner and [x2, y2] is the bottom-right corner.
The coordinates should be in the range [0, 1000]."""


def _xyxy_1000(bbox, wh_str: str) -> list[int]:
    """GT bbox [x, y, w, h] px + image_resolution 'WxH' -> [x1, y1, x2, y2] on Qwen's 0-1000 scale.

    Byte-for-byte the authors' SFT conversion (def bbox_xywh_pixel_to_xyxy_1000 in their
    train/src/dataset/data_utils.py): the image_resolution field rather than PIL, round()
    rather than int(), and NO clamping — their GRPO path clamps to [0,1000], the SFT path
    does not, so ~7% of boxes keep a coordinate slightly outside the range, as published.
    """
    if isinstance(bbox, str):
        bbox = json.loads(bbox)
    x, y, w, h = (float(v) for v in bbox)
    W, H = (float(v) for v in wh_str.lower().split("x"))
    return [round(x / W * 1000), round(y / H * 1000),
            round((x + w) / W * 1000), round((y + h) / H * 1000)]


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

    # The authors' TRAINING prompt, not the eval one. Two deliberate differences, both from
    # their train/ code and both flagged in their train/README.md as what reproduces the
    # released checkpoints:
    #   1. the tagged question field (def to_grpo_format uses record["question_tag"]), while
    #      evaluation strips those tags — an asymmetry they keep on purpose
    #   2. one extra line, "The coordinates should be in the range [0, 1000]."
    #      (MSR_RESPONSE_FORMAT in their src/dataset/grpo_dataset.py), absent from the eval
    #      prompt because the evaluator infers the scale instead
    # So this does NOT go through the vendored build_prompt: that one calls remove_tags().
    image_root = adapter.data_dir / IMAGES_SUBDIR         # adapter._abs()'s root, minus the private call
    train = json.load(open(tj, encoding="utf-8"))
    out = root / OUT_NAME
    n = skipped = missing_img = 0
    with open(out, "w", encoding="utf-8") as f:
        for row in train:
            img, ans, bbox, wh, q = (row.get(k) for k in
                                     ("image_path", "answer", "bbox", "image_resolution", "question_tag"))
            if not (img and ans and bbox and wh and q):
                skipped += 1
                continue
            try:
                xyxy = _xyxy_1000(bbox, wh)
            except Exception:
                skipped += 1
                continue
            image_abs = str((image_root / img).resolve())
            if not Path(image_abs).exists():
                missing_img += 1
                continue
            # same string def format_response_with_bbox builds in their data_utils.py
            target = (f"Answer: {ans}\n"
                      f"Bounding Box: [{xyxy[0]}, {xyxy[1]}, {xyxy[2]}, {xyxy[3]}]")
            rec = {
                "messages": [                                  # the authors' training prompt
                    {"role": "user", "content": "<image>\n" + q + MSR_RESPONSE_FORMAT},
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
