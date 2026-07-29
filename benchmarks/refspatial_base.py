from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
import re
import os
from PIL import Image
from tqdm import tqdm

from .base import BenchmarkAdapter, swift_record


class RefSpatialBase(BenchmarkAdapter):

    name: str = ""                                     # subclass: registered benchmark name
    HF_REPO: str = ""                                  # subclass: HF dataset repo id
    SUBSETS: tuple[str, ...] = ()                      # subclass: each has question.json / image/ / mask/
    # Subsets scored and reported per-subset, but kept OUT of `overall` and out of the
    # step/category/scene breakdowns, so those all share one denominator. Empty by default.
    OVERALL_EXCLUDE_SUBSETS: tuple[str, ...] = ()

    MODEL_SPECIFIC_PROMPT = True
    INFER_DEFAULTS = {"min_pixels": 1024 * 32 * 32}

    # -- prepare: download from HF if missing --
    def ensure_data(self) -> None:
        from huggingface_hub import snapshot_download

        root = self.data_dir
        root.mkdir(parents=True, exist_ok=True)
        if all((root / s / "question.json").exists() for s in self.SUBSETS):
            print(f"[{self.name}] already present: {root}")
            return

        subsets = " + ".join(f"{s}/" for s in self.SUBSETS)
        print(f"[{self.name}] downloading {subsets} (question.json, image/, mask/) ...")
        snapshot_download(
            self.HF_REPO,
            repo_type="dataset",
            local_dir=root,
            allow_patterns=[f"{s}/**" for s in self.SUBSETS],
        )
        print(f"[{self.name}] ready: {root}")

    def load_raw(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for subset in self.SUBSETS:
            qjson = self.data_dir / subset / "question.json"
            if not qjson.exists():
                raise FileNotFoundError(
                    f"{qjson} not found — run `python data_preparation.py {self.name}` first."
                )
            with open(qjson, encoding="utf-8") as f:
                for item in json.load(f):
                    item["_subset"] = subset  # carry which subset (used for path/scoring)
                    rows.append(item)
        return rows

    def _abs(self, subset: str, rel: str) -> str:
        return str((self.data_dir / subset / rel).resolve())

    def _prompt_for(self, row: dict[str, Any], model) -> str:
        # per-model grounding prompt (def get_prompt from RoboRefer Evaluation/test_benchmark.py)
        # Port of the official get_prompt(), branch-for-branch.
        # The official keys on the model NAME; our Model exposes tag + path, so we match
        # the same family substrings against "<tag> <path>" (e.g. "qwen3vl-32b
        # Qwen/Qwen3-VL-32B-Instruct"). Order mirrors the official if/elif chain.
        obj = row.get("object") or ""
        prompt, suffix = row.get("prompt") or "", row.get("suffix") or ""
        name = f"{model.tag} {model.path}".lower() if model is not None else ""
        if "molmo" in name:
            return f"Locate several points of {obj}."
        if "robobrain" in name:
            return f"{prompt} Please provide its 2D coordinates."
        if "gemini" in name:
            return f"Locate one point of {obj}."
        if "qwen" in name:
            return f"Locate {obj} in this image and output the point coordinates in JSON format."
        else:
            return f"{prompt} {suffix}".strip()

    def to_messages(self, row: dict[str, Any], model=None) -> dict[str, Any]:
        subset = row["_subset"]
        text = self._prompt_for(row, model)
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

    # Reshape ms-swift preds jsonl -> a flat all_results.json the scorer consumes
    # (the answers jsonl def eval_task writes in RoboRefer Evaluation/test_benchmark.py:
    #  question_id / text / mask_path)
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
                    "subset":    meta.get("subset"),           # aggregation group (see SUBSETS)
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
        print(f"[{self.name} reshape] {len(entries)} rows -> {out}")


    # (def text2pts from RoboRefer Evaluation/summarize_acc.py)
    @staticmethod
    def _text2pts(text: str, width: int = 640, height: int = 480, is_absolute: bool = False) -> list[tuple[int, int]]:

        pattern = r"\(([-+]?\d+\.?\d*(?:,\s*[-+]?\d+\.?\d*)*?)\)"
        matches = re.findall(pattern, text)
        points = []
        
        for match in matches:
            vector = [float(num) if "." in num else int(num) for num in match.split(",")]
            if len(vector) == 2:                               # single point
                x, y = vector
                if not is_absolute and (isinstance(x, float) or isinstance(y, float)):
                    x = int(x * width)
                    y = int(y * height)
                points.append((x, y))
            elif len(vector) == 4:                             # box -> fill every pixel inside
                x0, y0, x1, y1 = vector
                if not is_absolute:
                    x0 = int(x0 * width)
                    y0 = int(y0 * height)
                    x1 = int(x1 * width)
                    y1 = int(y1 * height)
                y, x = np.where(np.ones((y1 - y0, x1 - x0)))
                points.extend(list(np.stack([x + x0, y + y0], axis=1)))
        return np.array(points)

    # (def xml2pts from RoboRefer Evaluation/summarize_acc.py)
    @staticmethod
    def _xml2pts(text: str, width: int, height: int) -> np.ndarray:
        pattern = re.compile(r'(x\d+)="(-?\d+\.?\d*)"\s+(y\d+)="(-?\d+\.?\d*)"')
        matches = pattern.findall(text)
        return np.array([
            (int(float(x) / 100 * width), int(float(y) / 100 * height))
            for _, x, _, y in matches
        ])

    # (def json2pts from RoboRefer Evaluation/summarize_acc.py)
    @staticmethod
    def _json2pts(text: str, width=640, height=480) -> np.ndarray:
        match = re.search(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
        if not match:
            return np.empty((0, 2), dtype=int)
        try:
            data = json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            return np.empty((0, 2), dtype=int)

        points = []
        for item in data:
            if "point" in item and isinstance(item["point"], list) and len(item["point"]) == 2:
                y_norm, x_norm = item["point"]
                x = int(x_norm / 1000 * width)
                y = int(y_norm / 1000 * height)
                points.append((x, y))
        return np.array(points)


    # (ours — json2pts adapted to Qwen's key/axis order; see the docstring)
    @staticmethod
    def _qwen2pts(text: str, width: int, height: int) -> np.ndarray:
        text = text or ""
        raw: list[tuple[float, float]] = []

        # 1) JSON points — `point_2d` (Qwen, [x, y]) or `point` (Gemini, [y, x]). Extract the
        #    code fence if present, else try the whole string.
        m = re.search(r"```(?:\w+)?\s*(.*?)```", text, re.DOTALL)
        try:
            data = json.loads((m.group(1) if m else text).strip())
        except (json.JSONDecodeError, ValueError):
            data = None
        for item in ([data] if isinstance(data, dict) else data or []):
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("point_2d"), list) and len(item["point_2d"]) == 2:
                x, y = item["point_2d"]                        # Qwen order: [x, y]
                raw.append((float(x), float(y)))
            elif isinstance(item.get("point"), list) and len(item["point"]) == 2:
                y, x = item["point"]                           # Gemini order: [y, x]
                raw.append((float(x), float(y)))

        # 2) fallback: (x, y) tuples in free text (older Qwen tuple format).
        if not raw:
            for match in re.findall(r"\(([-+]?\d+\.?\d*(?:,\s*[-+]?\d+\.?\d*)*?)\)", text):
                nums = [float(n) for n in match.split(",")]
                if len(nums) == 2:
                    raw.append((nums[0], nums[1]))

        points = [(int(x / 1000 * width), int(y / 1000 * height)) for x, y in raw]
        return np.array(points) if points else np.empty((0, 2), dtype=int)

    # per-model parser dispatch (def main from RoboRefer Evaluation/summarize_acc.py),
    # except Qwen -> qwen instead of absolute
    @staticmethod
    def _parser_for_model(model) -> str:
        name = f"{getattr(model, 'tag', '')} {getattr(model, 'path', '')}".lower() if model is not None else ""
        if "molmo" in name:
            return "xml"
        if "gemini" in name:
            return "json"
        if "robobrain" in name:
            return "absolute"
        if "qwen" in name:
            return "qwen"
        return "normalized"                                    # RoboPoint/Claude/GPT4O/RoboRefer + default

    def score(self, in_dir: Path, parse_function: str | None = None, model=None, **opts: Any) -> dict[str, Any]:
        """Point-in-mask accuracy (def compute_accuracy from RoboRefer Evaluation/summarize_acc.py),
        scored with exactly ONE parser.

        parse_function: OMIT -> auto-picked from the model (_parser_for_model); PASS -> forced.
        Either way it is a single parser, fixed before any scoring happens, so the reported
        number can never be the best of several tried. **opts: ignored.
        """

        results_path = in_dir / "all_results.json"
        if not results_path.exists():
            raise FileNotFoundError(f"{results_path} not found — run reshape first.")
        with open(results_path, encoding="utf-8") as f:
            answers = json.load(f)

        #   normalized -> _text2pts (float*WH, int coords = absolute pixels; RoboPoint/Claude/
        #                 GPT4O/RoboRefer in main())
        #   absolute   -> _text2pts with is_absolute=True (no scaling; RoboBrain/Qwen in main())
        #   xml        -> _xml2pts  (Molmo x_i="..." y_i="..." / 100)
        #   json       -> _json2pts (```json [{"point":[y,x]}] /1000)
        #   qwen       -> _qwen2pts  (Qwen point_2d/[x,y]; _json2pts' /1000 conversion reused)
        PARSERS = {
            "normalized": lambda text, w, h: self._text2pts(text, w, h, False),
            "absolute":   lambda text, w, h: self._text2pts(text, w, h, True),
            "xml":        self._xml2pts,
            "json":       self._json2pts,
            "qwen":       self._qwen2pts,
        }
        if parse_function is None:
            name = self._parser_for_model(model)
            print(f"[{self.name} score] auto: parse_function={name!r} for model="
                  f"{getattr(model, 'tag', None)!r}; --parse-function overrides")
        else:
            name = parse_function.strip().lower()
        if name not in PARSERS:
            raise SystemExit(f"parse_function={name!r} unknown; choose one of {sorted(PARSERS)}")
        parse = PARSERS[name]

        acc_all: list[float] = []
        by_subset: dict[str, list[float]] = defaultdict(list)
        by_step: dict[str, list[float]] = defaultdict(list)
        by_category: dict[str, list[float]] = defaultdict(list)
        # scene (indoor/outdoor) is only labelled in some releases — RefSpatial-Bench's
        # question.json has no such field at all, so rows without it are left out of the
        # breakdown rather than bucketed under a guessed label.
        by_scene: dict[str, list[float]] = defaultdict(list)
        excluded: dict[str, int] = defaultdict(int)             # subset -> rows kept out of overall
        missing = no_scene = 0

        for answer in tqdm(answers):
            mask_path = answer.get("mask_path")
            if not mask_path or not os.path.exists(mask_path):
                missing += 1
                answer["accuracy"] = None
                continue

            mask = np.array(Image.open(mask_path)) / 255.0
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            mask = (mask > 0).astype(np.uint8)

            try:                                               # official compute_accuracy: print + skip
                points = parse(answer["text"], mask.shape[1], mask.shape[0])
            except Exception as e:
                print(f"Failed to parse question {answer.get('id')} ({name}): {e}")
                answer["accuracy"] = None
                continue

            acc = 0.0
            if len(points) > 0:
                in_range = (points[:, 0] >= 0) & (points[:, 0] < mask.shape[1]) & \
                           (points[:, 1] >= 0) & (points[:, 1] < mask.shape[0])
                acc = float(np.concatenate([
                    mask[points[in_range, 1], points[in_range, 0]],
                    np.zeros(points.shape[0] - in_range.sum())
                ]).mean())

            answer["accuracy"] = acc
            subset = answer.get("subset") or "?"
            by_subset[subset].append(acc)                      # every subset stays visible here
            if subset in self.OVERALL_EXCLUDE_SUBSETS:
                excluded[subset] += 1
                continue

            acc_all.append(acc)
            by_step[str(answer.get("step"))].append(acc)
            by_category[answer.get("category") or "?"].append(acc)
            scene = answer.get("scene") or None
            if scene is None:
                no_scene += 1
            else:
                by_scene[scene].append(acc)

        overall = float(np.mean(acc_all)) if acc_all else 0.0
        exc = ", ".join(f"{k} {v}" for k, v in sorted(excluded.items()))
        print(f"[{self.name} score] {name}={overall:.4f}  (n={len(acc_all)}, missing_mask={missing}"
              + (f", excluded_from_overall: {exc}" if exc else "")
              + (f", unlabelled_scene={no_scene}" if no_scene else "") + ")")

        # _agg -> summary_report shape; _cells -> make_table / print_summary shape
        # ({accuracy, count}; subset -> sub_task, step -> task).
        def _agg(d: dict[str, list[float]]) -> dict[str, dict[str, float]]:
            return {k: {"n": len(v), "acc": float(np.mean(v)) if v else 0.0} for k, v in sorted(d.items())}
        def _cells(d: dict[str, list[float]]) -> dict[str, dict[str, float]]:
            return {k: {"accuracy": float(np.mean(v)) if v else 0.0, "count": len(v)} for k, v in sorted(d.items())}

        with open(results_path, "w", encoding="utf-8") as f:   # per-question accuracy back in
            json.dump(answers, f, ensure_ascii=False, indent=2)

        with open(in_dir / "summary_report.json", "w", encoding="utf-8") as f:
            json.dump({
                "overall": {"n": len(acc_all), "acc": overall},
                "by_subset": _agg(by_subset),
                "by_step": _agg(by_step),
                "by_category": _agg(by_category),
                "by_scene": _agg(by_scene),                     # empty when the release has no scene labels
                "parse_function": name,
                "missing_mask": missing,
                "unlabelled_scene": no_scene,
                "excluded_from_overall": dict(sorted(excluded.items())),
            }, f, ensure_ascii=False, indent=2)

        # metrics.json (written by evaluate.py from this return) -> feeds make_table.
        return {
            "overall": {"accuracy": overall, "count": len(acc_all)},
            "category": _cells(by_category),
            "sub_task": _cells(by_subset),
            "task": _cells(by_step),
            "scene": _cells(by_scene),                          # empty when the release has no scene labels
            "parse_function": name,
            "excluded_from_overall": dict(sorted(excluded.items())),
        }
