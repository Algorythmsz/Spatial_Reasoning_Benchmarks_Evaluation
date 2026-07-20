"""benchmarks/multihopspatial.py — MultihopSpatial adapter.

HF: etri-vilab/MultihopSpatial (dataset)
    data/multihop_test_4500.json     4500 questions (test split)
    data/images/*.jpg                COCO-style images (~6.5k)

Questions already include <choice>(a)..(d)</choice> and the bbox request; a system prompt
(SYSTEM_PROMPT) pins the answer/bbox output format the scorer parses.

Scoring (rule-based, pure stdlib — no numpy/torch):
  - MCQ accuracy  : parsed answer letter vs gt (parse failure counts as wrong)
  - Acc@50IoU     : MCQ-correct AND predicted bbox IoU >= 0.5
  - Avg IoU       : conditional mean IoU over MCQ-correct samples only (paper Sec 5.2)
Parsing IS part of the metric (a parse failure is wrong / IoU=0); parse-failure and
coordinate-heuristic rates are logged per sample so "format failure" vs "reasoning
failure" stays distinguishable. Env knobs: MHS_IOU_THR (default 0.5),
MHS_STRICT=1 (reject coord scale/xywh rescue heuristics).
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import BenchmarkAdapter, register, swift_record

HF_REPO = "etri-vilab/MultihopSpatial"
TEST_JSON = "data/multihop_test_4500.json"
IMAGES_SUBDIR = "data/images"

# System prompt pins the output contract the scorer parses: a letter answer + a bbox_2d
# in NORMALIZED [0,1] xyxy (so predictions are directly comparable to the normalized GT).
SYSTEM_PROMPT = (
    "Please respond in the following format:\n"
    'Answer: (your choice, e.g., "(a) object name")\n'
    'Bounding Box: {"bbox_2d": [x1, y1, x2, y2]}\n'
    "Important: Use NORMALIZED coordinates (0.0 to 1.0).\n"
    'Example: {"bbox_2d": [0.25, 0.1, 0.75, 0.8]}'
)


# ─────────────────────────────────────────────────────────────────────────────
# Prediction parsing  (parsing failures are part of the metric — see module docstring)
# ─────────────────────────────────────────────────────────────────────────────
LETTER_SET = "abcd"

# "Answer: (a) ..." anchored pattern — highest priority.
ANSWER_LINE_RE = re.compile(r"answer\s*[:\-]?\s*\(?([abcd])\)?", re.IGNORECASE)
# Any "(a)"-style occurrence — fallback; take the LAST since rationales often
# enumerate other options before committing.
ANY_LETTER_RE = re.compile(r"\(([abcd])\)", re.IGNORECASE)
# bbox: {"bbox_2d": [..]}, "Bounding Box: [..]", or a bare 4-number list.
BBOX_JSON_RE = re.compile(r'bbox_2d"?\s*[:=]\s*\[([^\]]+)\]', re.IGNORECASE)
BBOX_LINE_RE = re.compile(r"bounding\s*box\s*[:\-]?\s*\[([^\]]+)\]", re.IGNORECASE)
BARE_LIST_RE = re.compile(r"\[\s*([\d.,\s\-eE]+?)\s*\]")


@dataclass
class ParsedPrediction:
    letter: str | None = None
    bbox: list[float] | None = None          # canonical: normalized xyxy in [0,1]
    letter_source: str = "none"              # answer_line | last_paren | text_match | none
    bbox_source: str = "none"                # json | line | bare | none
    scale_heuristic: str = "none"            # none | div1000 | pixel_guess
    xywh_heuristic: bool = False
    notes: list[str] = field(default_factory=list)


def parse_answer_letter(text: str, options: list[str] | None) -> tuple[str | None, str]:
    """Anchored 'Answer:' line, then last (x), then option-text substring match."""
    m = ANSWER_LINE_RE.search(text)
    if m:
        return m.group(1).lower(), "answer_line"
    all_letters = ANY_LETTER_RE.findall(text)
    if all_letters:
        return all_letters[-1].lower(), "last_paren"
    if options:
        lowered = text.lower()                                 # longest-option-first so "black cellphone" wins over "cellphone"
        for idx, opt in sorted(enumerate(options), key=lambda t: -len(t[1])):
            if opt.lower().strip() in lowered:
                return LETTER_SET[idx], "text_match"
    return None, "none"


def _extract_floats(s: str) -> list[float] | None:
    try:
        vals = [float(x) for x in re.split(r"[,\s]+", s.strip()) if x]
    except ValueError:
        return None
    return vals if len(vals) == 4 else None


def parse_bbox(text: str, parsed: ParsedPrediction,
               image_wh: tuple[int, int] | None = None, strict: bool = False) -> list[float] | None:
    """Extract a bbox and canonicalize to normalized xyxy in [0,1].

    Heuristics (each logged; disabled under strict):
      - values > 1.5       -> assume 0-1000 scale (or pixel if image size known)
      - x2 <= x1 / y2 <= y1 -> try xywh -> xyxy reinterpretation
    """
    raw = None
    for pattern, src in ((BBOX_JSON_RE, "json"), (BBOX_LINE_RE, "line")):
        m = pattern.search(text)
        if m and (raw := _extract_floats(m.group(1))):
            parsed.bbox_source = src
            break
    if raw is None:                                            # last bare 4-number list (bbox usually comes last)
        candidates = [c for c in BARE_LIST_RE.findall(text) if _extract_floats(c)]
        if candidates:
            raw = _extract_floats(candidates[-1])
            parsed.bbox_source = "bare"
    if raw is None:
        return None

    x1, y1, x2, y2 = raw
    if max(raw) > 1.5:                                         # scale heuristic
        if image_wh and max(raw) > 1000:
            w, h = image_wh
            x1, x2, y1, y2 = x1 / w, x2 / w, y1 / h, y2 / h
            parsed.scale_heuristic = "pixel_guess"
        else:
            x1, y1, x2, y2 = (v / 1000.0 for v in raw)
            parsed.scale_heuristic = "div1000"

    if x2 <= x1 or y2 <= y1:                                   # xywh heuristic
        nx2, ny2 = x1 + x2, y1 + y2
        if x1 < nx2 <= 1.5 and y1 < ny2 <= 1.5:
            x2, y2 = nx2, ny2
            parsed.xywh_heuristic = True
        else:
            parsed.notes.append("degenerate_bbox")
            return None

    if strict and (parsed.scale_heuristic != "none" or parsed.xywh_heuristic):
        parsed.notes.append("rejected_by_strict")
        return None

    box = [min(max(v, 0.0), 1.0) for v in (x1, y1, x2, y2)]    # clamp to [0,1]
    if box[2] <= box[0] or box[3] <= box[1]:
        parsed.notes.append("degenerate_after_clamp")
        return None
    return box


def parse_prediction(text: str, options: list[str] | None = None,
                     image_wh: tuple[int, int] | None = None, strict: bool = False) -> ParsedPrediction:
    p = ParsedPrediction()
    if not isinstance(text, str) or not text.strip():
        p.notes.append("empty_prediction")
        return p
    p.letter, p.letter_source = parse_answer_letter(text, options)
    p.bbox = parse_bbox(text, p, image_wh, strict)
    return p


def iou_xyxy(a: list[float], b: list[float]) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def parse_wh(wh: Any) -> tuple[int, int] | None:
    """image_resolution is a 'WxH' string (e.g. '640x480', width first) or a [W, H] list."""
    if isinstance(wh, str):
        m = re.fullmatch(r"\s*(\d+)\s*[xX*]\s*(\d+)\s*", wh)
        return (int(m.group(1)), int(m.group(2))) if m else None
    if isinstance(wh, (list, tuple)) and len(wh) == 2:
        try:
            return int(wh[0]), int(wh[1])
        except (TypeError, ValueError):
            return None
    return None


def gt_to_norm_xyxy(bbox: Any, image_wh: tuple[int, int] | None) -> list[float] | None:
    """MultihopSpatial GT is COCO-style [x, y, w, h] in absolute pixels. Convert to
    normalized xyxy [0,1] so it's comparable to the parsed (normalized) prediction.
    Needs image_wh (W, H); returns None if it's missing or the box is degenerate."""
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4 and image_wh):
        return None
    w_img, h_img = image_wh
    if not (w_img and h_img):
        return None
    x, y, bw, bh = bbox
    box = [x / w_img, y / h_img, (x + bw) / w_img, (y + bh) / h_img]
    box = [min(max(v, 0.0), 1.0) for v in box]
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


@dataclass
class _Bucket:
    """Accumulator for one group (overall / per-hop / per-view)."""
    n: int = 0
    mcq_correct: int = 0
    grounded_correct: int = 0            # mcq correct AND IoU >= thr
    iou_sum_on_correct: float = 0.0      # numerator of the conditional Avg IoU
    letter_parse_fail: int = 0
    bbox_parse_fail: int = 0
    heuristic_fired: int = 0

    def add(self, mcq_ok: bool, iou: float, thr: float, p: ParsedPrediction) -> None:
        self.n += 1
        self.letter_parse_fail += p.letter is None
        self.bbox_parse_fail += p.bbox is None
        self.heuristic_fired += (p.scale_heuristic != "none") or p.xywh_heuristic
        if mcq_ok:
            self.mcq_correct += 1
            self.iou_sum_on_correct += iou
            if iou >= thr:
                self.grounded_correct += 1

    def cell(self) -> dict[str, Any]:
        """make_table/print_summary shape: headline accuracy (MCQ) + count."""
        return {"accuracy": self.mcq_correct / self.n if self.n else 0.0, "count": self.n}

    def report(self) -> dict[str, Any]:
        """Rich per-group readout persisted to summary_report.json (percentages)."""
        if self.n == 0:
            return {}
        pc = lambda x: round(100 * x, 1)
        return {
            "n": self.n,
            "mcq_acc": pc(self.mcq_correct / self.n),
            "acc@50iou": pc(self.grounded_correct / self.n),
            "avg_iou": pc(self.iou_sum_on_correct / self.mcq_correct) if self.mcq_correct else None,
            "ungrounded_ratio": pc(1 - self.grounded_correct / self.mcq_correct) if self.mcq_correct else None,
            "letter_parse_fail_rate": pc(self.letter_parse_fail / self.n),
            "bbox_parse_fail_rate": pc(self.bbox_parse_fail / self.n),
            "coord_heuristic_rate": pc(self.heuristic_fired / self.n),
        }


@register
class MultihopSpatialAdapter(BenchmarkAdapter):
    name = "multihopspatial"
    # scoring is rule-based (MCQ accuracy + bbox IoU) and pure-stdlib — runs in the
    # inference env (no extra deps). See module docstring for the metric definitions.

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
        question = row.get("question") or ""
        images = [self._abs(row["image_path"])] if row.get("image_path") else []
        messages = [                                           # system pins the answer/bbox output contract
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "<image>" * len(images) + question},
        ]
        meta = {
            "id": row.get("id"),
            "answer": row.get("answer"),
            "bbox": row.get("bbox"),                           # GT bbox: COCO-style [x, y, w, h] in pixels
            "options": row.get("options") or row.get("choices"),  # enables the text-match letter fallback
            "hop": row.get("hop"),
            "view": row.get("view"),
            "question_tag": row.get("question_tag"),
            "image_resolution": row.get("image_resolution"),   # "WxH" string (e.g. "640x480") -> normalize GT bbox
        }
        return swift_record(row.get("id"), images=images, meta=meta, messages=messages)

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
                    "id":        p.get("id", meta.get("id")),
                    "hop":       meta.get("hop"),
                    "view":      meta.get("view"),
                    "gt_answer": meta.get("answer"),
                    "gt_bbox":   meta.get("bbox"),
                    "options":   meta.get("options"),
                    "image_wh":  meta.get("image_resolution"),
                    "prediction": pred,
                })

        out = out_dir / "all_results.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        print(f"[multihopspatial reshape] {len(entries)} rows -> {out}")

    # Score: MCQ accuracy + bbox-IoU grounding, aggregated overall / per-hop / per-view.
    def score(self, in_dir: Path) -> dict[str, Any]:
        results_path = in_dir / "all_results.json"
        if not results_path.exists():
            raise FileNotFoundError(f"{results_path} not found — run reshape first.")
        with open(results_path, encoding="utf-8") as f:
            records = json.load(f)

        iou_thr = float(os.environ.get("MHS_IOU_THR", "0.5"))
        strict = os.environ.get("MHS_STRICT", "0") == "1"

        overall = _Bucket()
        by_hop: dict[str, _Bucket] = defaultdict(_Bucket)
        by_view: dict[str, _Bucket] = defaultdict(_Bucket)
        gt_unavailable = 0                                     # GT bbox missing / no image_wh -> IoU forced to 0

        for r in records:
            image_wh = parse_wh(r.get("image_wh"))            # "640x480" (W first) or [W, H] -> (W, H)
            p = parse_prediction(r.get("prediction", ""), r.get("options"), image_wh, strict)

            # gt_answer is the full choice string, e.g. "(c) frame of the reed picture" —
            # pull just the "(x)" letter to compare against the parsed prediction letter.
            # (A plain .strip("()") would leave "c) frame ..." and never match -> 0% MCQ.)
            gt_raw = str(r.get("gt_answer", "")).strip().lower()
            gm = ANY_LETTER_RE.search(gt_raw)
            gt_letter = gm.group(1) if gm else gt_raw.strip("() ")[:1]
            mcq_ok = p.letter is not None and p.letter == gt_letter
            gt_box = gt_to_norm_xyxy(r.get("gt_bbox"), image_wh)  # COCO pixel xywh -> normalized xyxy
            if gt_box is None:
                gt_unavailable += 1
            iou = iou_xyxy(p.bbox, gt_box) if (p.bbox and gt_box) else 0.0

            overall.add(mcq_ok, iou, iou_thr, p)
            by_hop[f"{r.get('hop', '?')}hop"].add(mcq_ok, iou, iou_thr, p)
            by_view[str(r.get("view", "?"))].add(mcq_ok, iou, iou_thr, p)

            r["mcq_correct"] = mcq_ok                          # annotate per-sample for the persisted all_results.json
            r["iou"] = round(iou, 4)
            r["grounded"] = mcq_ok and iou >= iou_thr
            r["pred_letter"] = p.letter
            r["letter_source"] = p.letter_source
            r["bbox_source"] = p.bbox_source
            r["scale_heuristic"] = p.scale_heuristic
            r["xywh_heuristic"] = p.xywh_heuristic
            r["parse_notes"] = p.notes

        # rich report (percentages) for summary_report.json
        report = {
            "overall": overall.report(),
            "by_hop": {k: b.report() for k, b in sorted(by_hop.items())},
            "by_view": {k: b.report() for k, b in sorted(by_view.items())},
            "iou_thr": iou_thr,
            "strict": strict,
            "gt_bbox_unavailable": gt_unavailable,             # samples where IoU couldn't be computed
        }
        with open(results_path, "w", encoding="utf-8") as f:   # persist per-sample annotations back
            json.dump(records, f, ensure_ascii=False, indent=2)
        with open(in_dir / "summary_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        o = overall.report()
        print(f"[multihopspatial score] mcq={o.get('mcq_acc')}%  acc@50iou={o.get('acc@50iou')}%  "
              f"avg_iou={o.get('avg_iou')}  (n={overall.n}, iou_thr={iou_thr}, strict={strict}, "
              f"gt_bbox_unavailable={gt_unavailable})")

        # make_table / evaluate.py print_summary shape: overall + category(hop) + sub_task(view)
        return {
            "overall": {**overall.cell(),                      # headline accuracy = MCQ; extras kept for reference
                        "acc@50iou": overall.grounded_correct / overall.n if overall.n else 0.0,
                        "avg_iou": overall.iou_sum_on_correct / overall.mcq_correct if overall.mcq_correct else 0.0},
            "category": {k: b.cell() for k, b in sorted(by_hop.items())},
            "sub_task": {k: b.cell() for k, b in sorted(by_view.items())},
        }
