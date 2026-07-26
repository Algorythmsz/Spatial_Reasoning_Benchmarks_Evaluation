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

from .base import BenchmarkAdapter, register, swift_record

HF_REPO = "BAAI/RefSpatial-Bench"
SUBSETS = ("Location", "Placement", "Unseen")  # each subset has question.json / image/ / mask/


@register
class RefSpatialBenchAdapter(BenchmarkAdapter):
    name = "refspatial_bench"

    MODEL_SPECIFIC_PROMPT = True

    # -- prepare: download from HF if missing --
    def ensure_data(self) -> None:
        from huggingface_hub import snapshot_download

        root = self.data_dir
        root.mkdir(parents=True, exist_ok=True)
        if all((root / s / "question.json").exists() for s in SUBSETS):
            print(f"[refspatial_bench] already present: {root}")
            return

        print("[refspatial_bench] downloading Location/ + Placement/ + Unseen/ (question.json, image/, mask/) ...")
        snapshot_download(
            HF_REPO,
            repo_type="dataset",
            local_dir=root,
            allow_patterns=[f"{s}/**" for s in SUBSETS],
        )
        print(f"[refspatial_bench] ready: {root}")

    def load_raw(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for subset in SUBSETS:
            qjson = self.data_dir / subset / "question.json"
            if not qjson.exists():
                raise FileNotFoundError(
                    f"{qjson} not found — run `python data_preparation.py refspatial_bench` first."
                )
            with open(qjson, encoding="utf-8") as f:
                for item in json.load(f):
                    item["_subset"] = subset  # carry which subset (used for path/scoring)
                    rows.append(item)
        return rows

    def _abs(self, subset: str, rel: str) -> str:
        return str((self.data_dir / subset / rel).resolve())

    def _prompt_for(self, row: dict[str, Any], model) -> str:
        # Port of the official RefSpatial-Bench get_prompt(), branch-for-branch.
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
        return f"{prompt} {suffix}".strip()                    # dataset default (official `else`)

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
                    "subset":    meta.get("subset"),           # Location | Placement | Unseen (aggregation group)
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
        print(f"[refspatial_bench reshape] {len(entries)} rows -> {out}")


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

    @staticmethod
    def _xml2pts(text: str, width: int, height: int) -> np.ndarray:
        pattern = re.compile(r'(x\d+)="(-?\d+\.?\d*)"\s+(y\d+)="(-?\d+\.?\d*)"')
        matches = pattern.findall(text)
        return np.array([
            (int(float(x) / 100 * width), int(float(y) / 100 * height))
            for _, x, _, y in matches
        ])

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


    @staticmethod
    def _norm1000_pts(text: str, width: int, height: int) -> np.ndarray:
        text = text or ""
        raw: list[tuple[float, float]] = []

        # 1) JSON points. Qwen3-VL grounding emits ```json [{"point_2d": [x, y], "label": …}]```
        #    (point_2d is [x, y]); also accept a Gemini-style "point": [y, x]. Extract the code
        #    fence if present, else try the whole string.
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

        # magnitude rule: v in [0,1] -> v*dim ; |v|>1 -> v/1000*dim (norm1000).
        points: list[tuple[int, int]] = []
        for x, y in raw:
            x = x * width if abs(x) <= 1 else x / 1000 * width
            y = y * height if abs(y) <= 1 else y / 1000 * height
            points.append((int(x), int(y)))
        return np.array(points) if points else np.empty((0, 2), dtype=int)

    @staticmethod
    def _parser_for_model(model) -> str:
        # Auto parser when --parse-function is OMITTED: mirrors the official main()'s per-model
        # dispatch (keyed on tag+path substrings), but Qwen -> qwen1000 because our JSON prompt
        # makes Qwen emit point_2d, which the original's `absolute`/text2pts cannot read. Order
        # mirrors the official if/elif chain. The user can still force any parser via
        # --parse-function (that overrides this).
        name = f"{getattr(model, 'tag', '')} {getattr(model, 'path', '')}".lower() if model is not None else ""
        if "molmo" in name:
            return "xml"
        if "gemini" in name:
            return "json"
        if "robobrain" in name:
            return "absolute"
        if "qwen" in name:
            return "qwen1000"
        return "normalized"                                    # RoboPoint/Claude/GPT4O/RoboRefer + default

    # Score: fraction of predicted points that land inside the GT mask (RoboRefer metric).
    # parse_function: which point parser(s) to score with. OMIT -> AUTO best-of: score the
    #   model's parser (_parser_for_model, evaluate.py passes `model`) AND text2pts/normalized,
    #   report whichever scores higher overall. PASS -> force it for every model (comma-separate
    #   for several; first = primary, no best-of). See PARSERS below. **opts: ignored.
    def score(self, in_dir: Path, parse_function: str | None = None, model=None, **opts: Any) -> dict[str, Any]:

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
        #   qwen1000   -> _norm1000_pts (Qwen norm1000 magnitude rule; (x,y) tuples + point_2d)
        PARSERS = {
            "normalized": lambda text, w, h: self._text2pts(text, w, h, False),
            "absolute":   lambda text, w, h: self._text2pts(text, w, h, True),
            "xml":        self._xml2pts,
            "json":       self._json2pts,
            "qwen1000":   self._norm1000_pts,
        }
        if parse_function is None:
            # AUTO: score with the model's parser AND text2pts (normalized), then report
            # whichever scores HIGHER overall (pick_best). Guards against a wrong auto-pick or
            # a model that emits an unexpected format. (If the model's parser already IS
            # normalized, there's nothing to compare -> single parser.)
            auto = self._parser_for_model(model)
            names = [auto] if auto == "normalized" else [auto, "normalized"]
            pick_best = True
            print(f"[refspatial_bench score] auto: scoring {names} for model="
                  f"{getattr(model, 'tag', None)!r}, keeping the higher overall; --parse-function overrides")
        else:
            # EXPLICIT: user forces the parser(s); first one is primary (no best-of).
            names = [p.strip().lower() for p in parse_function.split(",") if p.strip()]
            pick_best = False
            if not names:
                raise SystemExit(f"refspatial_bench --parse-function is empty; choose one of {sorted(PARSERS)}")
        for n in names:
            if n not in PARSERS:
                raise SystemExit(f"parse_function={n!r} unknown; choose one of {sorted(PARSERS)}")

        acc_all: dict[str, list[float]] = {n: [] for n in names}
        by_subset = {n: defaultdict(list) for n in names}
        by_step = {n: defaultdict(list) for n in names}
        by_category = {n: defaultdict(list) for n in names}
        missing = 0

        for answer in tqdm(answers):
            mask_path = answer.get("mask_path")
            if not mask_path or not os.path.exists(mask_path):
                missing += 1
                for n in names:
                    answer[f"accuracy_{n}"] = None
                answer["accuracy"] = None
                continue

            mask = np.array(Image.open(mask_path)) / 255.0

            if mask.ndim == 3:
                mask = mask[:, :, 0]
            mask = (mask > 0).astype(np.uint8)

            subset, step, category = answer.get("subset") or "?", str(answer.get("step")), answer.get("category") or "?"
            for n in names:
                try:                                           # official compute_accuracy: print + skip
                    points = PARSERS[n](answer["text"], mask.shape[1], mask.shape[0])
                except Exception as e:
                    print(f"Failed to parse question {answer.get('id')} ({n}): {e}")
                    answer[f"accuracy_{n}"] = None
                    continue

                acc = 0.0
                if len(points) > 0:
                    in_range = (points[:, 0] >= 0) & (points[:, 0] < mask.shape[1]) & \
                               (points[:, 1] >= 0) & (points[:, 1] < mask.shape[0])
                    acc = float(np.concatenate([
                        mask[points[in_range, 1], points[in_range, 0]],
                        np.zeros(points.shape[0] - in_range.sum())
                    ]).mean())

                answer[f"accuracy_{n}"] = acc
                acc_all[n].append(acc)
                by_subset[n][subset].append(acc)
                by_step[n][step].append(acc)
                by_category[n][category].append(acc)

        # primary parser: best-of overall in AUTO mode; else the first the user listed.
        def _overall(n: str) -> float:
            return float(np.mean(acc_all[n])) if acc_all[n] else 0.0
        primary = max(names, key=_overall) if pick_best else names[0]
        for answer in answers:                                 # backward-compat per-question field
            answer["accuracy"] = answer.get(f"accuracy_{primary}")

        print("[refspatial_bench score] "
              + "  ".join(f"{n}={(float(np.mean(acc_all[n])) if acc_all[n] else 0.0):.4f}" for n in names)
              + f"  (primary={primary}, n={len(acc_all[primary])}, missing_mask={missing})")

        # Per-parser artifacts. _summary -> summary_report shape; _metrics -> make_table /
        # print_summary shape ({accuracy, count}; subset -> sub_task, step -> task).
        def _agg(d: dict[str, list[float]]) -> dict[str, dict[str, float]]:
            return {k: {"n": len(v), "acc": float(np.mean(v)) if v else 0.0} for k, v in sorted(d.items())}
        def _cells(d: dict[str, list[float]]) -> dict[str, dict[str, float]]:
            return {k: {"accuracy": float(np.mean(v)) if v else 0.0, "count": len(v)} for k, v in sorted(d.items())}
        def _summary(n: str) -> dict[str, Any]:
            return {
                "overall": {"n": len(acc_all[n]), "acc": float(np.mean(acc_all[n])) if acc_all[n] else 0.0},
                "by_subset": _agg(by_subset[n]),
                "by_step": _agg(by_step[n]),
                "by_category": _agg(by_category[n]),
                "parse_function": n,
                "missing_mask": missing,
            }
        def _metrics(n: str) -> dict[str, Any]:
            return {
                "overall": {"accuracy": float(np.mean(acc_all[n])) if acc_all[n] else 0.0, "count": len(acc_all[n])},
                "category": _cells(by_category[n]),
                "sub_task": _cells(by_subset[n]),
                "task": _cells(by_step[n]),
                "parse_function": n,
            }

        with open(results_path, "w", encoding="utf-8") as f:   # per-question accuracy_* back into all_results.json
            json.dump(answers, f, ensure_ascii=False, indent=2)

        # summary_report.json is always the PRIMARY (so single-parser output is unchanged).
        with open(in_dir / "summary_report.json", "w", encoding="utf-8") as f:
            json.dump(_summary(primary), f, ensure_ascii=False, indent=2)

        # Several parsers -> also drop a standalone file pair PER parser, name-suffixed
        # (metrics_<name>.json / summary_report_<name>.json). Each is a full, independent
        # result for that parser — no embedded by_parse_function.
        if len(names) > 1:
            for n in names:
                with open(in_dir / f"summary_report_{n}.json", "w", encoding="utf-8") as f:
                    json.dump(_summary(n), f, ensure_ascii=False, indent=2)
                with open(in_dir / f"metrics_{n}.json", "w", encoding="utf-8") as f:
                    json.dump(_metrics(n), f, ensure_ascii=False, indent=2)

        # metrics.json (written by evaluate.py from this return) = the PRIMARY parser -> feeds
        # make_table. parse_functions lists the others so a reader knows the _<name> files exist.
        return {**_metrics(primary), "parse_functions": names}
