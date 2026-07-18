"""benchmarks/refspatial_expand.py — RefSpatial-Expand-Bench adapter.

HF: JingkunAn/RefSpatial-Expand-Bench (dataset)
    Location/question.json + Location/image/*.jpg + Location/mask/*.png
    Placement/question.json + Placement/image/*.jpg + Placement/mask/*.png
    (data/*.parquet are duplicates for HF datasets loading — here we use json+images only)

Each question: object/prompt/suffix + rgb_path/mask_path. The model outputs a list of
points, scored (geo) by whether they fall inside the ground-truth mask (needs PIL+numpy; see README).
         (reshape/score are implemented in the evaluate stage; here only prepare/preprocess.)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import BenchmarkAdapter, register, swift_record

HF_REPO = "JingkunAn/RefSpatial-Expand-Bench"
SUBSETS = ("Location", "Placement")  # each subset has question.json / image/ / mask/


@register
class RefSpatialExpandAdapter(BenchmarkAdapter):
    name = "refspatial_expand"
    # scoring is a points-in-mask geometry check; needs PIL + numpy (see README).

    # -- prepare: download from HF if missing (idempotent) --
    def ensure_data(self) -> None:
        from huggingface_hub import snapshot_download  # lazy

        root = self.data_dir
        root.mkdir(parents=True, exist_ok=True)
        if all((root / s / "question.json").exists() for s in SUBSETS):
            print(f"[refspatial_expand] already present: {root}")
            return

        print("[refspatial_expand] downloading Location/ + Placement/ (question.json, image/, mask/) ...")
        snapshot_download(
            HF_REPO,
            repo_type="dataset",
            local_dir=root,
            allow_patterns=[f"{s}/**" for s in SUBSETS],
        )
        print(f"[refspatial_expand] ready: {root}")

    def load_raw(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for subset in SUBSETS:
            qjson = self.data_dir / subset / "question.json"
            if not qjson.exists():
                raise FileNotFoundError(
                    f"{qjson} not found — run `python data_preparation.py refspatial_expand` first."
                )
            with open(qjson, encoding="utf-8") as f:
                for item in json.load(f):
                    item["_subset"] = subset  # carry which subset (used for path/scoring)
                    rows.append(item)
        return rows

    def _abs(self, subset: str, rel: str) -> str:
        return str((self.data_dir / subset / rel).resolve())

    def to_messages(self, row: dict[str, Any]) -> dict[str, Any]:
        subset = row["_subset"]
        prompt = row.get("prompt") or ""
        suffix = row.get("suffix") or ""
        text = f"{prompt} {suffix}".strip()
        images = [self._abs(subset, row["rgb_path"])] if row.get("rgb_path") else []
        meta = {
            "id": row.get("id"),
            "subset": subset,
            "object": row.get("object"),
            "category": row.get("category"),
            "step": row.get("step"),
            "scene": row.get("scene"),
            "mask_path": self._abs(subset, row["mask_path"]) if row.get("mask_path") else None,
        }
        uid = f"{subset}-{row.get('id')}"
        return swift_record(uid, text, images, meta=meta)

    def reshape(self, preds_path: Path, out_dir: Path) -> None:
        raise NotImplementedError("refspatial_expand.reshape: to be implemented in the evaluate stage")

    def score(self, in_dir: Path) -> dict[str, Any]:
        raise NotImplementedError("refspatial_expand.score: to be implemented in the evaluate stage")
