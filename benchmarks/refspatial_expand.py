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

    # Reshape ms-swift preds jsonl -> a flat all_results.json the scorer consumes.
    def reshape(self, preds_path: Path, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        entries: list[dict[str, Any]] = []
        with open(preds_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                p = json.loads(line)
                meta = p.get("meta") or {}

                pred = p.get("response")                       # swift stores the generation under "response"
                if pred is None:                               # fallback: last assistant turn
                    msgs = p.get("messages") or []
                    pred = next(
                        (m.get("content", "") for m in reversed(msgs) if m.get("role") == "assistant"),
                        "",
                    )

                entries.append({
                    "id":        p.get("id", meta.get("id")),  # our uid "<subset>-<id>"
                    "subset":    meta.get("subset"),           # Location | Placement (aggregation group)
                    "object":    meta.get("object"),           # target description (provenance)
                    "category":  meta.get("category"),         # aggregation group
                    "step":      meta.get("step"),             # reasoning steps (aggregation group)
                    "scene":     meta.get("scene"),            # indoor/outdoor (provenance)
                    "mask_path": meta.get("mask_path"),        # absolute GT mask path (set in to_messages)
                    "text":      pred,                         # model prediction (free text with points)
                })

        out = out_dir / "all_results.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        print(f"[refspatial_expand reshape] {len(entries)} rows -> {out}")

    # ── point parsing (ported from RoboRefer Evaluation/summarize_acc.py::text2pts) ──
    # The dataset suffix asks for normalized 0-1 tuples "[(x1, y1)]", so the default is
    # is_absolute=False: float coords are scaled by (width, height); integers are taken
    # as absolute pixels (RoboRefer's own heuristic). Set RS_ABSOLUTE=1 to treat every
    # coord as absolute pixels (mirrors RoboRefer's Qwen/RoboBrain branch) if a model
    # ignores the prompt and emits pixels.
    @staticmethod
    def _text2pts(text: str, width: int, height: int, is_absolute: bool):
        import re
        import numpy as np

        pattern = r"\(([-+]?\d+\.?\d*(?:,\s*[-+]?\d+\.?\d*)*?)\)"
        points: list[Any] = []
        for match in re.findall(pattern, text or ""):
            vector = [float(n) if "." in n else int(n) for n in match.split(",")]
            if len(vector) == 2:
                x, y = vector
                if not is_absolute and (isinstance(x, float) or isinstance(y, float)):
                    x, y = int(x * width), int(y * height)
                points.append((x, y))
            elif len(vector) == 4:                             # a box -> fill every pixel inside it
                x0, y0, x1, y1 = vector
                if not is_absolute:
                    x0, y0 = int(x0 * width), int(y0 * height)
                    x1, y1 = int(x1 * width), int(y1 * height)
                w, h = max(0, int(x1) - int(x0)), max(0, int(y1) - int(y0))
                if w and h:
                    ys, xs = np.where(np.ones((h, w)))
                    points.extend(list(np.stack([xs + int(x0), ys + int(y0)], axis=1)))
        return np.array(points) if points else np.empty((0, 2), dtype=int)

    # Score: fraction of predicted points that land inside the GT mask (RoboRefer metric).
    def score(self, in_dir: Path) -> dict[str, Any]:
        import os
        from collections import defaultdict

        import numpy as np
        from PIL import Image

        results_path = in_dir / "all_results.json"
        if not results_path.exists():
            raise FileNotFoundError(f"{results_path} not found — run reshape first.")
        with open(results_path, encoding="utf-8") as f:
            answers = json.load(f)

        is_absolute = os.environ.get("RS_ABSOLUTE", "0") == "1"

        per_subset: dict[str, list[float]] = defaultdict(list)
        per_step: dict[str, list[float]] = defaultdict(list)
        per_category: dict[str, list[float]] = defaultdict(list)
        all_acc: list[float] = []
        missing = 0

        for a in answers:
            mask_path = a.get("mask_path")
            if not mask_path or not os.path.exists(mask_path):
                missing += 1
                a["accuracy"] = None
                continue

            mask = np.array(Image.open(mask_path)) / 255.0
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            mask = (mask > 0).astype(np.uint8)

            pts = self._text2pts(a.get("text", ""), mask.shape[1], mask.shape[0], is_absolute)

            acc = 0.0
            if len(pts) > 0:
                pts = pts.astype(int)
                in_range = (
                    (pts[:, 0] >= 0) & (pts[:, 0] < mask.shape[1])
                    & (pts[:, 1] >= 0) & (pts[:, 1] < mask.shape[0])
                )
                hits = mask[pts[in_range, 1], pts[in_range, 0]]           # 1 if inside mask
                acc = float(np.concatenate([hits, np.zeros(len(pts) - int(in_range.sum()))]).mean())

            a["accuracy"] = acc
            all_acc.append(acc)
            per_subset[a.get("subset") or "?"].append(acc)
            per_step[str(a.get("step"))].append(acc)
            per_category[a.get("category") or "?"].append(acc)

        def _agg(d: dict[str, list[float]]) -> dict[str, dict[str, float]]:
            return {k: {"n": len(v), "acc": float(np.mean(v)) if v else 0.0} for k, v in sorted(d.items())}

        summary = {
            "overall": {"n": len(all_acc), "acc": float(np.mean(all_acc)) if all_acc else 0.0},
            "by_subset": _agg(per_subset),
            "by_step": _agg(per_step),
            "by_category": _agg(per_category),
            "is_absolute": is_absolute,
            "missing_mask": missing,
        }

        # persist per-question accuracy + the summary next to all_results.json
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(answers, f, ensure_ascii=False, indent=2)
        with open(in_dir / "summary_report.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[refspatial_expand score] overall acc={summary['overall']['acc']:.4f} "
              f"(n={summary['overall']['n']}, missing_mask={missing}, is_absolute={is_absolute})")
        return summary
