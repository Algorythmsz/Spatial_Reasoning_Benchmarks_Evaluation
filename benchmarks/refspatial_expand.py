"""benchmarks/refspatial_expand.py — RefSpatial-Expand-Bench adapter.

HF: JingkunAn/RefSpatial-Expand-Bench (dataset)
    Location/{question.json, image/*.jpg, mask/*.png}
    Placement/{question.json, image/*.jpg, mask/*.png}
    (data/*.parquet is the same content packed for HF `datasets`; we read the
     json + image/ + mask/ trees directly and ignore the parquet.)

Task: given an RGB image and a referring prompt (object + spatial instruction), the
model must POINT to the target location(s). The dataset suffix asks for normalized
0-1 tuples, e.g. "[(0.25, 0.40)]".

Scoring (RoboRefer point-in-mask metric; needs PIL + numpy):
  - Parse predicted points out of the free-text answer; non-point matches
    (e.g. boxes) are ignored, matching the official evaluator.
  - Per-sample accuracy = fraction of predicted points that land inside the GT mask
    (out-of-bounds points count as misses; no parseable points -> 0.0).
  - Aggregated overall and by subset (Location/Placement) / step / category.

──────────────────────────────────────────────────────────────────────────────
COORDINATE / PROMPT CONVENTIONS  (why there is more than one mode)
──────────────────────────────────────────────────────────────────────────────
Qwen3-VL / Qwen3.5 do NOT obey the dataset's "0-1 tuple" suffix: they emit points in a
0-1000 normalized space (ms-swift's qwen template logs `norm_bbox: norm1000`). Scoring
those numbers as-is under-reproduces badly (acc ~0.05-0.10), so we support env-selected
conventions:

  RS_PROMPT   (read in to_messages, BAKED into the fingerprinted jsonl):
    "official" (default) -> append the dataset `suffix` (asks for 0-1 tuples).
    "qwen_json"          -> cookbook prompt: '<expr>. Output the point coordinates in
                            JSON format.' -> model returns [{"point_2d": [x, y], ...}].
    Either prompt is scored correctly by RS_COORD=qwen1000, so re-inference is optional.

  RS_COORD    score() ALWAYS computes BOTH conventions below and persists both (per-
              question "accuracy_normalized"/"accuracy_qwen1000" in all_results.json, and
              summary_report.json's "by_mode"). RS_COORD only picks the PRIMARY — the value
              that feeds metrics.json / make_table / the terminal readout and the per-
              question "accuracy" field (default: "normalized").
    "normalized" (default) -> official _text2pts: float coords scale by (W, H);
                              integer coords are absolute ORIGINAL pixels.
                              RS_ABSOLUTE=1 skips the float scaling (models that emit
                              original pixels regardless of prompt).
    "qwen1000"   (Qwen3-VL / Qwen3.5, RECOMMENDED) -> Qwen's norm1000 convention:
                              coords are normalized to 0-1000, NOT 0-1 and NOT pixels.
                              Confirmed by ms-swift itself (`norm_bbox: norm1000`), by the
                              emitted ranges (y hits ~1000 while the image is only ~450
                              tall), and empirically (re-scoring lifts acc ~5-7x). Outputs
                              MIX conventions in practice, so a per-value magnitude rule
                              maps each coord: v in [0,1] -> v*dim (the model obeying the
                              0-1 suffix); |v|>1 -> v/1000*dim (norm1000). Parses BOTH the
                              (x, y) tuple format (default prompt) and JSON point_2d
                              (RS_PROMPT=qwen_json) — so existing tuple preds can be
                              re-scored with NO re-inference.

To reproduce Qwen3-VL / Qwen3.5 RefSpatial numbers:
    Just re-score existing preds with  RS_COORD=qwen1000  (no re-inference needed —
    the norm1000 magnitude rule handles the tuple-format output the default prompt
    already produced). RS_PROMPT=qwen_json (cookbook JSON output) is optional and also
    scored correctly by qwen1000.

This adapter implements the whole pipeline: ensure_data (HF download), load_raw,
to_messages (preprocess), reshape (preds jsonl -> scorer schema), and score.
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

    # Cookbook grounding prompt (Qwen3-VL spatial_understanding.ipynb, Part 2):
    # a referring expression followed by an explicit request for JSON point output.
    # The model then returns [{"point_2d": [x, y], "label": ...}] in the format it was
    # trained on (see RS_PROMPT / RS_COORD notes in the module docstring).
    QWEN_JSON_SUFFIX = "Output the point coordinates in JSON format."

    def to_messages(self, row: dict[str, Any]) -> dict[str, Any]:
        import os

        subset = row["_subset"]
        prompt = row.get("prompt") or ""
        # RS_PROMPT selects the instruction, and — because preprocess() fingerprints
        # the built jsonl — the choice is BAKED into the cached file: flipping the env
        # changes the fingerprint and auto-regenerates (base.py::preprocess). Keep it
        # consistent across preprocess+infer for a given run.
        #   official  -> dataset suffix (0-1 normalized tuple), model-agnostic default.
        #   qwen_json -> cookbook JSON grounding prompt for Qwen-VL native output.
        if os.environ.get("RS_PROMPT", "official") == "qwen_json":
            suffix = self.QWEN_JSON_SUFFIX
        else:
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

    # ── point parsing (ported verbatim from the official RefSpatial-Expand-Bench eval) ──
    # The authoritative evaluator (dataset card) parses ONLY (x, y) points:
    #   - a coord is scaled by (W, H) when it is a float; integer coords are absolute pixels.
    #   - matches that are not 2-tuples (e.g. 4-number boxes) are IGNORED — they contribute
    #     no points, so a box-only or unparseable prediction yields [] and scores 0.0.
    # There is no box branch and no is_absolute knob in the official code; we keep an optional
    # RS_ABSOLUTE=1 (is_absolute) toggle that additionally skips float scaling for models that
    # emit absolute pixels, but the default (is_absolute=False) is bit-identical to the official.
    @staticmethod
    def _text2pts(text: str, width: int, height: int, is_absolute: bool) -> list[tuple[int, int]]:
        import re

        pattern = r"\(([-+]?\d+\.?\d*(?:,\s*[-+]?\d+\.?\d*)*?)\)"
        points: list[tuple[int, int]] = []
        for match in re.findall(pattern, text or ""):
            vector = [float(n) if "." in n else int(n) for n in match.split(",")]
            if len(vector) == 2:                               # official: points only
                x, y = vector
                if not is_absolute and (isinstance(x, float) or isinstance(y, float)):
                    x, y = int(x * width), int(y * height)
                points.append((int(x), int(y)))
        return points

    # ── Qwen cookbook JSON parsing (RS_COORD=qwen) ──
    # Mirrors decode_json_points() from spatial_understanding.ipynb: strip a ```json
    # fence, json.loads, read each item's "point_2d": [x, y]. Falls back to a regex
    # over bare "point_2d" pairs when the response has prose around the JSON or the
    # JSON is malformed. Coords are returned RAW (resized-frame pixels); the caller
    # applies the smart_resize inverse-transform. No normalized/pixel guessing here.
    @staticmethod
    def _json2pts(text: str) -> list[tuple[float, float]]:
        import re

        raw = text or ""
        if "```json" in raw:                                   # ```json ... ``` fence -> inner block
            raw = raw.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in raw:                                     # generic fence
            raw = raw.split("```", 1)[1].split("```", 1)[0]

        pts: list[tuple[float, float]] = []
        try:
            data = json.loads(raw)
            if isinstance(data, dict):                         # tolerate a single object
                data = [data]
            for item in data:
                if isinstance(item, dict) and "point_2d" in item:
                    xy = item["point_2d"]
                    if isinstance(xy, (list, tuple)) and len(xy) == 2:
                        pts.append((float(xy[0]), float(xy[1])))
        except Exception:
            pass
        if pts:
            return pts
        # Fallback: pull "point_2d": [x, y] pairs out of surrounding text.
        for mx, my in re.findall(
            r'"point_2d"\s*:\s*\[\s*([-+]?\d+\.?\d*)\s*,\s*([-+]?\d+\.?\d*)\s*\]', text or ""
        ):
            pts.append((float(mx), float(my)))
        return pts

    # ── Qwen norm1000 coords (RS_COORD=qwen1000) ──
    # ms-swift's qwen template uses norm_bbox=norm1000: Qwen3-VL/3.5 emit point coords in
    # a 0-1000 normalized space (NOT the 0-1 the dataset suffix asks for, and NOT resized-
    # frame pixels). In practice outputs MIX conventions within a single run, so we decide
    # per value by magnitude:
    #   |v| <= 1  -> v * dim          (the model did obey the 0-1 suffix)
    #   |v|  > 1  -> v / 1000 * dim   (norm1000: integers like 745, or floats like 580.0)
    # This magnitude rule matched or beat the int-vs-float-type rule on every model tested
    # (and both beat a blanket /1000, which mangles the genuine 0-1 outputs). We accept the
    # (x, y) tuple format (default prompt) OR JSON point_2d (RS_PROMPT=qwen_json): try JSON
    # first, fall back to tuples — so tuple-format preds re-score with no re-inference.
    @staticmethod
    def _norm1000_pts(text: str, width: int, height: int) -> list[tuple[int, int]]:
        import re

        raw = RefSpatialExpandAdapter._json2pts(text)      # JSON point_2d (qwen_json prompt)
        if not raw:                                        # else (x, y) tuples (default prompt)
            for match in re.findall(r"\(([-+]?\d+\.?\d*(?:,\s*[-+]?\d+\.?\d*)*?)\)", text or ""):
                nums = [float(n) for n in match.split(",")]
                if len(nums) == 2:
                    raw.append((nums[0], nums[1]))
        points: list[tuple[int, int]] = []
        for x, y in raw:
            x = x * width if abs(x) <= 1 else x / 1000 * width
            y = y * height if abs(y) <= 1 else y / 1000 * height
            points.append((int(x), int(y)))
        return points

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

        # ── coord interpretation (see module docstring) ──
        # We ALWAYS score under BOTH conventions and persist both, so normalized and
        # qwen1000 are directly comparable with no re-run. RS_COORD only picks the PRIMARY
        # one — the value that feeds metrics.json / make_table / the terminal summary and
        # the per-question "accuracy" field (default: normalized). The other is still saved
        # (per-question "accuracy_<mode>", and summary_report's "by_mode").
        #   normalized -> official _text2pts (float*WH / int=pixel); RS_ABSOLUTE=1 skips the
        #                 float scaling (models emitting original pixels regardless of prompt).
        #   qwen1000   -> Qwen norm1000 magnitude rule (_norm1000_pts).
        MODES = ("normalized", "qwen1000")
        primary = os.environ.get("RS_COORD", "normalized").lower()
        if primary not in MODES:
            primary = "normalized"
        is_absolute = os.environ.get("RS_ABSOLUTE", "0") == "1"

        def _parse(text: str, w: int, h: int, mode: str) -> list[tuple[int, int]]:
            if mode == "qwen1000":
                return self._norm1000_pts(text, w, h)
            return self._text2pts(text, w, h, is_absolute)

        # per-mode accumulators: overall list + the three breakdown groups
        acc_all: dict[str, list[float]] = {m: [] for m in MODES}
        by_subset = {m: defaultdict(list) for m in MODES}
        by_step = {m: defaultdict(list) for m in MODES}
        by_category = {m: defaultdict(list) for m in MODES}
        missing = 0

        for a in answers:
            mask_path = a.get("mask_path")
            if not mask_path or not os.path.exists(mask_path):
                missing += 1
                for m in MODES:
                    a[f"accuracy_{m}"] = None
                a["accuracy"] = None
                continue

            # GT mask -> binary {0,1}: drop the alpha/extra channels (use channel 0)
            # and treat any non-zero pixel as "inside the target region".
            mask = np.array(Image.open(mask_path)) / 255.0
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            mask = (mask > 0).astype(np.uint8)

            # acc = (# predicted points inside the mask) / (# predicted points).
            # Out-of-image points count as misses; no parseable points -> acc = 0.0
            # (matches the official evaluator: a failed/box-only prediction scores 0).
            # NOTE: (h, w) is the ORIGINAL-image frame (mask is stored at full image
            # resolution), which is also the reference frame for the mask lookup below.
            h, w = mask.shape
            text = a.get("text", "") or ""
            subset, step, category = a.get("subset") or "?", str(a.get("step")), a.get("category") or "?"
            for m in MODES:
                pts = _parse(text, w, h, m)
                if pts:
                    hits = sum(1 for x, y in pts if 0 <= x < w and 0 <= y < h and mask[y, x] > 0)
                    acc = hits / len(pts)
                else:
                    acc = 0.0
                a[f"accuracy_{m}"] = acc
                acc_all[m].append(acc)
                by_subset[m][subset].append(acc)
                by_step[m][step].append(acc)
                by_category[m][category].append(acc)
            a["accuracy"] = a[f"accuracy_{primary}"]           # primary (backward-compat field)

        def _agg(d: dict[str, list[float]]) -> dict[str, dict[str, float]]:
            return {k: {"n": len(v), "acc": float(np.mean(v)) if v else 0.0} for k, v in sorted(d.items())}

        def _mode_report(m: str) -> dict[str, Any]:
            return {
                "overall": {"n": len(acc_all[m]), "acc": float(np.mean(acc_all[m])) if acc_all[m] else 0.0},
                "by_subset": _agg(by_subset[m]),
                "by_step": _agg(by_step[m]),
                "by_category": _agg(by_category[m]),
            }

        reports = {m: _mode_report(m) for m in MODES}

        # summary_report.json: BOTH modes under "by_mode"; top-level mirrors the primary.
        summary = {
            **reports[primary],                                # overall/by_subset/by_step/by_category (primary)
            "primary_coord_mode": primary,
            "is_absolute": is_absolute,                        # only meaningful for the normalized mode
            "missing_mask": missing,
            "by_mode": reports,                                # normalized + qwen1000 (full breakdowns)
        }
        with open(results_path, "w", encoding="utf-8") as f:   # per-question accuracy_* back into all_results.json
            json.dump(answers, f, ensure_ascii=False, indent=2)
        with open(in_dir / "summary_report.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("[refspatial_expand score] "
              + "  ".join(f"{m}={reports[m]['overall']['acc']:.4f}" for m in MODES)
              + f"  (primary={primary}, n={reports[primary]['overall']['n']}, missing_mask={missing})")

        # metrics.json -> make_table / evaluate.py print_summary shape ({accuracy, count}),
        # from the PRIMARY mode. subset (Location/Placement) -> sub_task, step -> task.
        # "by_coord_mode" carries BOTH overalls for comparison (extra key; print_summary /
        # make_table read only the known keys above, so it is ignored there).
        def _cells(d: dict[str, list[float]]) -> dict[str, dict[str, float]]:
            return {k: {"accuracy": float(np.mean(v)) if v else 0.0, "count": len(v)}
                    for k, v in sorted(d.items())}
        return {
            "overall": {"accuracy": reports[primary]["overall"]["acc"], "count": reports[primary]["overall"]["n"]},
            "category": _cells(by_category[primary]),
            "sub_task": _cells(by_subset[primary]),
            "task": _cells(by_step[primary]),
            "coord_mode": primary,
            "by_coord_mode": {m: {"accuracy": reports[m]["overall"]["acc"], "count": reports[m]["overall"]["n"]}
                              for m in MODES},
        }
