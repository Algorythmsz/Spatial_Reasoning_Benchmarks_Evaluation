"""MultihopSpatial — driven by the OFFICIAL evaluator, on ms-swift's inference engine.

The authors published their harness (github.com/youngwanLEE/multihopspatial, eval/), so
none of the protocol is ours any more. `benchmarks/vendor/multihopspatial/` holds a
byte-identical copy of `eval/benchmark_qwen_vllm.py`; this adapter only:

  1. points it at the dataset we already downloaded,
  2. swaps its vLLM calls for ms-swift ones (swift_backend.py rebinds two names),
  3. reshapes its output into the repo's preds/results layout.

Prompt, retry rounds, answer/bbox parsing, coordinate scaling, IoU and the hop x view
metric table all execute upstream's code. Nothing here reimplements them — earlier
versions of this file did, and every one of those guesses turned out to differ from the
real protocol in some way.

Protocol notes worth knowing when comparing to our other benchmarks:
  - Image resolution is NOT pinned. Upstream leaves min/max_pixels at the model default;
    our other benches pin them to the SpatialScore protocol. Different by design.
  - Decoding is NOT greedy by default: upstream reads the checkpoint's
    generation_config.json (Qwen3-VL-4B: t=0.7 / top_p=0.8 / top_k=20). Retries only mean
    something with sampling, so scores vary run to run. `--greedy` restores determinism
    and, with it, disables the retry rounds.
  - max_new_tokens is 8192 upstream (not our usual 512).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import BenchmarkAdapter, register

HF_REPO = "etri-vilab/MultihopSpatial"
TEST_JSON = "data/multihop_test_4500.json"
IMAGES_SUBDIR = "data/images"

# Upstream run_benchmark() defaults (eval/benchmark_qwen_vllm.py argparse). Repeated here
# so a change on either side is visible as a diff rather than silently inherited.
OFFICIAL_DEFAULTS = {
    "max_model_len": 32768,
    "max_new_tokens": 8192,
    "gpu_memory_utilization": 0.9,
    "max_retries": 3,
    "batch_size": 100,
}


def load_upstream():
    """Import the vendored official evaluator with ms-swift wired in as its backend."""
    from .vendor.multihopspatial import swift_backend

    return swift_backend.install()


def _local_model_dir(model_path: str) -> str:
    """Resolve to a real directory on disk.

    Upstream's build_sampling_params only reads generation_config.json when
    `os.path.isdir(model_path)` — hand it an HF repo id and it silently falls back to
    greedy decoding, which in turn makes the retry rounds a no-op (a retry at temperature 0
    reproduces the same string). A quiet protocol downgrade, so resolve it here.
    """
    if Path(model_path).is_dir():
        return model_path
    from huggingface_hub import snapshot_download  # lazy

    local = snapshot_download(model_path)
    print(f"[multihopspatial] resolved {model_path} -> {local}")
    return local


@register
class MultihopSpatialAdapter(BenchmarkAdapter):
    name = "multihopspatial"
    # The vendored official evaluator owns the generation loop: it inspects each response
    # and RE-GENERATES the invalid ones (up to 3 rounds), which the args-driven
    # infer_main(InferArguments(...)) path cannot express. So infer.py hands the whole
    # inference step to run_inference() below. See infer.py's UPSTREAM_OWNS_LOOP branch.
    UPSTREAM_OWNS_LOOP = True

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
        snapshot_download(
            HF_REPO,
            repo_type="dataset",
            local_dir=root,
            allow_patterns=[TEST_JSON, f"{IMAGES_SUBDIR}/*"],
        )
        print(f"[multihopspatial] ready: {root}")

    def load_raw(self) -> list[dict[str, Any]]:
        with open(self.test_json, encoding="utf-8") as f:
            return json.load(f)

    @property
    def test_json(self) -> Path:
        p = self.data_dir / TEST_JSON
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found — run `python data_preparation.py multihopspatial` first.")
        return p

    @property
    def image_root(self) -> Path:
        return self.data_dir / IMAGES_SUBDIR

    def preprocess(self, model=None) -> Path:
        """No ms-swift jsonl is built: upstream reads the raw test JSON and renders its own
        prompt. Overriding the base implementation keeps a stale, unused prompt file from
        being written (and from looking authoritative)."""
        return self.test_json

    def to_messages(self, row, model=None):                    # unused; upstream owns the prompt
        raise NotImplementedError(
            "multihopspatial prompts come from the vendored official evaluator "
            "(benchmarks/vendor/multihopspatial/benchmark_qwen_vllm.py::build_prompt).")

    # ── inference (called by infer.py because UPSTREAM_OWNS_LOOP) ───────────
    def run_inference(self, model, model_path: str, max_new_tokens: int | None = None,
                      *, greedy: bool = False, temperature: float | None = None,
                      max_retries: int | None = None, seed: int | None = None,
                      test_samples: int | None = None, **_: Any) -> int:
        """Run upstream's run_benchmark() and write the repo's preds files."""
        from .vendor.multihopspatial import swift_backend

        model_path = _local_model_dir(model_path)
        upstream = load_upstream()
        swift_backend.configure(model_type=model.model_type,
                                enable_thinking=model.enable_thinking)

        preds = self.preds_path(model)
        preds.parent.mkdir(parents=True, exist_ok=True)
        official_out = preds.with_suffix(".official.json")     # upstream owns this file (also its --resume state)

        tp = model.vllm_tensor_parallel_size
        if tp is None:                                         # else split over every visible GPU
            import os

            tp = len([d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]) or 1

        upstream.run_benchmark(
            json_path=str(self.test_json),
            image_root=str(self.image_root),
            output_path=str(official_out),
            model_path=model_path,
            max_model_len=OFFICIAL_DEFAULTS["max_model_len"],
            gpu_memory_utilization=OFFICIAL_DEFAULTS["gpu_memory_utilization"],
            tensor_parallel_size=tp,
            max_new_tokens=max_new_tokens or OFFICIAL_DEFAULTS["max_new_tokens"],
            greedy=greedy,
            temperature_override=temperature,
            seed=seed,
            test_samples=test_samples,
            max_retries=OFFICIAL_DEFAULTS["max_retries"] if max_retries is None else max_retries,
            batch_size=OFFICIAL_DEFAULTS["batch_size"],
        )

        records = json.loads(official_out.read_text(encoding="utf-8"))
        records = [r for r in records if r is not None]
        with open(preds, "w", encoding="utf-8") as f:          # jsonl so base.is_complete can count rows
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[multihopspatial] {len(records)} results -> {preds}")
        return len(records)

    # ── scoring ─────────────────────────────────────────────────────────────
    def reshape(self, preds_path: Path, out_dir: Path) -> None:
        """Upstream already emits per-sample results (prediction / pred_bbox / iou / score),
        so this is a straight carry-over into the repo's results layout."""
        out_dir.mkdir(parents=True, exist_ok=True)
        records = [json.loads(ln) for ln in
                   preds_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        out = out_dir / "all_results.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"[multihopspatial reshape] {len(records)} rows -> {out}")

    def score(self, in_dir: Path, **opts: Any) -> dict[str, Any]:
        """Metrics come from upstream's calculate_full_metrics — overall, per hop, and per
        hop x view cell (the paper's published layout)."""
        upstream = load_upstream()

        results_path = in_dir / "all_results.json"
        if not results_path.exists():
            raise FileNotFoundError(f"{results_path} not found — run reshape first.")
        records = json.loads(results_path.read_text(encoding="utf-8"))

        def cell(rows: list[dict]) -> dict[str, Any]:
            m = upstream.calculate_full_metrics(rows)
            return {                                           # repo shape: fractions + count
                "accuracy": m["accuracy"] / 100.0,
                "count": m["total_evaluated"],
                "acc@50iou": m["acc_at_iou50"] / 100.0,
                "avg_iou": m["avg_iou"],
            }

        hops = sorted({r.get("hop") for r in records if r.get("hop")}, reverse=True)   # 3hop..1hop
        views = sorted({r.get("view") for r in records if r.get("view")})
        by_hop = {h: cell([r for r in records if r.get("hop") == h]) for h in hops}
        by_view = {v: cell([r for r in records if r.get("view") == v]) for v in views}
        by_cell = {f"{h.replace('hop', 'Hop')}-{v.capitalize()}":
                   cell([r for r in records if r.get("hop") == h and r.get("view") == v])
                   for h in hops for v in views}

        overall = cell(records)
        report = {
            "paper_table": _paper_table(overall, by_cell),
            "overall": overall,
            "by_hop": by_hop,
            "by_view": by_view,
            "by_hop_view": by_cell,
            "scorer": "official (vendored benchmark_qwen_vllm.py::calculate_full_metrics)",
        }
        with open(in_dir / "summary_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        _print_table(overall, by_cell)
        return {                                               # make_table / print_summary shape
            "overall": overall,
            "category": by_hop,
            "sub_task": by_view,
            "task": by_cell,
        }


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
