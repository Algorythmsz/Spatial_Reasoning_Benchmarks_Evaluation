"""benchmarks/multihopspatial.py — MultihopSpatial adapter.

HF: etri-vilab/MultihopSpatial (dataset)
    data/multihop_test_4500.json     4500 questions (test split)
    data/images/*.jpg                COCO-style images (~6.5k)

Questions already include <choice>(a)..(d)</choice> and the bbox request, so almost no
prompt shaping is needed.
Scoring: rule-based (choice correctness + bbox IoU); needs only numpy (see README).
         (reshape/score are implemented in the evaluate stage; here only prepare/preprocess.)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import BenchmarkAdapter, register, swift_record

HF_REPO = "etri-vilab/MultihopSpatial"
TEST_JSON = "data/multihop_test_4500.json"
IMAGES_SUBDIR = "data/images"


@register
class MultihopSpatialAdapter(BenchmarkAdapter):
    name = "multihopspatial"
    # scoring is rule-based (choice correctness + bbox IoU); needs only numpy (see README).

    # -- prepare: download from HF if missing (idempotent) --
    def ensure_data(self) -> None:
        from huggingface_hub import snapshot_download  # lazy

        root = self.data_dir
        root.mkdir(parents=True, exist_ok=True)
        images_dir = root / IMAGES_SUBDIR
        have_json = (root / TEST_JSON).exists()
        have_imgs = images_dir.is_dir() and any(images_dir.iterdir())
        if have_json and have_imgs:
            print(f"[multihopspatial] already present: {root / TEST_JSON}")
            return

        print("[multihopspatial] downloading test json + images ...")
        snapshot_download(
            HF_REPO,
            repo_type="dataset",
            local_dir=root,
            allow_patterns=[TEST_JSON, f"{IMAGES_SUBDIR}/*"],
        )
        print(f"[multihopspatial] ready: {root}")

    def load_raw(self) -> list[dict[str, Any]]:
        p = self.data_dir / TEST_JSON
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found — run `python data_preparation.py multihopspatial` first."
            )
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    def _abs(self, image_path: str) -> str:
        return str((self.data_dir / IMAGES_SUBDIR / image_path).resolve())

    def to_messages(self, row: dict[str, Any]) -> dict[str, Any]:
        text = row.get("question") or ""
        images = [self._abs(row["image_path"])] if row.get("image_path") else []
        meta = {
            "id": row.get("id"),
            "answer": row.get("answer"),
            "bbox": row.get("bbox"),
            "hop": row.get("hop"),
            "view": row.get("view"),
            "question_tag": row.get("question_tag"),
            "image_resolution": row.get("image_resolution"),
        }
        return swift_record(row.get("id"), text, images, meta=meta)

    def reshape(self, preds_path: Path, out_dir: Path) -> None:
        raise NotImplementedError("multihopspatial.reshape: to be implemented in the evaluate stage")

    def score(self, in_dir: Path) -> dict[str, Any]:
        raise NotImplementedError("multihopspatial.score: to be implemented in the evaluate stage")
