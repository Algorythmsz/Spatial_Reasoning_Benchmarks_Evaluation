"""MultihopSpatial — the authors' evaluator, minus its retry loop.

`benchmarks/vendor/multihopspatial/benchmark_qwen_vllm.py` is a byte-identical copy of the
official harness. Everything that decides what a score MEANS is imported from it and used
unchanged:

    build_prompt            the exact prompt text and layout
    parse_response          answer letter + bbox, including the /1000 scale rule
    calculate_iou           COCO xywh GT -> pixel-space IoU against the predicted box
    compute_score           MCQ correctness
    calculate_full_metrics  Acc / Acc@50IoU / avg IoU, and their denominators

The ONE thing not reproduced is upstream's retry loop, which re-generates any response
whose answer or bbox fails to parse (up to 3 rounds). Reproducing it meant handing the
generation loop to upstream's code, which in turn needed a custom inference path, an engine
shim, and sampled (non-deterministic) decoding — a lot of machinery for a bound measured at
1.8% of responses (80/4500 on qwen3vl-4b). Dropped deliberately: this bench now runs the
same single-pass, greedy, deterministic path as every other one.

    Consequence: an unparseable response is scored as-is (wrong answer / no box) rather than
    retried, so these numbers are a slight UNDER-estimate versus the paper's protocol.

Generation settings still follow upstream, via INFER_DEFAULTS: image resolution unpinned
(upstream never pins it) and max_new_tokens 8192 (not this repo's usual 512).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import BenchmarkAdapter, register, swift_record

HF_REPO = "etri-vilab/MultihopSpatial"
TEST_JSON = "data/multihop_test_4500.json"
IMAGES_SUBDIR = "data/images"


def upstream():
    """The vendored official evaluator. Imported lazily — the module pulls in vllm."""
    from .vendor.multihopspatial import benchmark_qwen_vllm

    return benchmark_qwen_vllm


@register
class MultihopSpatialAdapter(BenchmarkAdapter):
    name = "multihopspatial"
    # Upstream leaves image resolution at the model default and generates up to 8192 tokens;
    # this repo's defaults (SpatialScore pixel bounds, 512 tokens) would change image-token
    # counts and truncation behaviour, and the numbers would stop being comparable.
    INFER_DEFAULTS = {"max_new_tokens": 8192, "pin_pixels": False}

    # ── data ────────────────────────────────────────────────────────────────
    def ensure_data(self) -> None:
        from huggingface_hub import snapshot_download  # lazy

        root = self.data_dir
        root.mkdir(parents=True, exist_ok=True)
        images_dir = root / IMAGES_SUBDIR
        if (root / TEST_JSON).exists() and images_dir.is_dir() and any(images_dir.iterdir()):
            print(f"[multihopspatial] already present: {root / TEST_JSON}")
            return

        print("[multihopspatial] downloading test json + images ...")
        snapshot_download(HF_REPO, repo_type="dataset", local_dir=root,
                          allow_patterns=[TEST_JSON, f"{IMAGES_SUBDIR}/*"])
        print(f"[multihopspatial] ready: {root}")

    def load_raw(self) -> list[dict[str, Any]]:
        p = self.data_dir / TEST_JSON
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found — run `python data_preparation.py multihopspatial` first.")
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    def _abs(self, image_path: str) -> str:
        return str((self.data_dir / IMAGES_SUBDIR / image_path).resolve())

    def to_messages(self, row: dict[str, Any], model=None) -> dict[str, Any]:
        """Upstream's build_prompt verbatim: one user turn, question then format block."""
        images = [self._abs(row["image_path"])] if row.get("image_path") else []
        content = "<image>" * len(images) + upstream().build_prompt(row.get("question", ""))
        meta = {
            "id": row.get("id"),
            "answer": row.get("answer"),                       # "(c) frame of the reed picture"
            "bbox": row.get("bbox"),                           # GT: COCO [x, y, w, h] in pixels
            "hop": row.get("hop"),                             # 1hop | 2hop | 3hop
            "view": row.get("view"),                           # ego | exo
            "image_path": row.get("image_path"),               # to read the image size at scoring
        }
        return swift_record(row.get("id"), images=images, meta=meta,
                            messages=[{"role": "user", "content": content}])

    # ── scoring: every decision below is upstream's ─────────────────────────
    def reshape(self, preds_path: Path, out_dir: Path) -> None:
        """swift preds -> upstream's per-sample result shape.

        Runs upstream's own parse_response / calculate_iou / compute_score, so parsing and
        IoU are exactly the official ones. Image size comes from PIL, as upstream does.
        """
        from PIL import Image

        up = upstream()
        out_dir.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, Any]] = []
        with open(preds_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                p = json.loads(line)
                meta = p.get("meta") or {}
                raw = p.get("response")
                if raw is None:                                # fallback: last assistant turn
                    msgs = p.get("messages") or []
                    raw = next((m.get("content", "") for m in reversed(msgs)
                                if m.get("role") == "assistant"), "")

                prediction, pred_bbox = up.parse_response(raw or "")
                with Image.open(self._abs(meta["image_path"])) as img:
                    w, h = img.size
                records.append({
                    "id": p.get("id", meta.get("id")),
                    "hop": meta.get("hop"),
                    "view": meta.get("view"),
                    "answer": meta.get("answer"),
                    "bbox": meta.get("bbox"),
                    "prediction": prediction,
                    "pred_bbox": pred_bbox,
                    "iou": up.calculate_iou(meta.get("bbox"), pred_bbox, w, h),
                    "score": up.compute_score(prediction, meta.get("answer")),
                    "raw_response": raw,
                })

        out = out_dir / "all_results.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"[multihopspatial reshape] {len(records)} rows -> {out}")

    def score(self, in_dir: Path, **opts: Any) -> dict[str, Any]:
        """Metrics from upstream's calculate_full_metrics: overall, per hop, per view, and
        per hop x view cell (the paper's published layout)."""
        up = upstream()
        results_path = in_dir / "all_results.json"
        if not results_path.exists():
            raise FileNotFoundError(f"{results_path} not found — run reshape first.")
        records = json.loads(results_path.read_text(encoding="utf-8"))

        def cell(rows: list[dict]) -> dict[str, Any]:
            m = up.calculate_full_metrics(rows)
            return {"accuracy": m["accuracy"] / 100.0,         # repo shape: fractions + count
                    "count": m["total_evaluated"],
                    "acc@50iou": m["acc_at_iou50"] / 100.0,
                    "avg_iou": m["avg_iou"]}

        hops = sorted({r.get("hop") for r in records if r.get("hop")}, reverse=True)  # 3hop..1hop
        views = sorted({r.get("view") for r in records if r.get("view")})
        overall = cell(records)
        by_hop = {h: cell([r for r in records if r.get("hop") == h]) for h in hops}
        by_view = {v: cell([r for r in records if r.get("view") == v]) for v in views}
        by_cell = {f"{h.replace('hop', 'Hop')}-{v.capitalize()}":
                   cell([r for r in records if r.get("hop") == h and r.get("view") == v])
                   for h in hops for v in views}

        report = {
            "paper_table": _paper_table(overall, by_cell),
            "overall": overall, "by_hop": by_hop, "by_view": by_view, "by_hop_view": by_cell,
            "scorer": "official (vendored benchmark_qwen_vllm.py), single-pass — no retry rounds",
        }
        with open(in_dir / "summary_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        _print_table(overall, by_cell)
        return {"overall": overall, "category": by_hop, "sub_task": by_view, "task": by_cell}


def _paper_table(overall: dict, by_cell: dict) -> dict[str, Any]:
    """Flat, pasteable `"3Hop-Ego Acc@50": 9.9` pairs in the published column order."""
    pc = lambda x: round(100 * x, 1)
    table = {"Overall Acc.": pc(overall["accuracy"]),
             "Overall Acc@50": pc(overall["acc@50iou"]),
             "Overall avg. IoU": pc(overall["avg_iou"])}
    for name, c in by_cell.items():
        table[f"{name} Acc."] = pc(c["accuracy"])
        table[f"{name} Acc@50"] = pc(c["acc@50iou"])
    return table


def _print_table(overall: dict, by_cell: dict) -> None:
    rows = [("Overall", overall)] + list(by_cell.items())
    w = max(len(n) for n, _ in rows)
    print(f"\n{'group'.ljust(w)}  {'Acc.':>7}  {'Acc@50':>7}  {'avg. IoU':>8}  {'n':>5}")
    print("-" * (w + 34))
    for name, c in rows:
        print(f"{name.ljust(w)}  {100 * c['accuracy']:7.1f}  {100 * c['acc@50iou']:7.1f}  "
              f"{100 * c['avg_iou']:8.1f}  {c['count']:5d}")
