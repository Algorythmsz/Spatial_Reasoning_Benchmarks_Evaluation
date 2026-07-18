"""
benchmarks/base.py
────────────────────────────────────────────────────────────────────────
Shared foundation for POST_CRISP benchmark adapters.

★ This file is also imported in env2 (the scoring env), so only stdlib is imported
  at the top. Heavy deps (datasets / torch / vllm, ...) are lazy-imported inside
  each adapter's methods.

Contains:
  - Model             : one entry from models.yaml (minimal info for inference)
  - register / get_adapter / list_adapters / resolve : name -> adapter instance
  - path/completion contract : cache / preds / results / done.flag / is_complete
  - BenchmarkAdapter  : the 4 methods each bench implements + shared preprocess
                        (fingerprint-based auto-invalidation)
                        The input jsonl is model-agnostic. Per-model settings live
                        in test.py + models.yaml + swift flags.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── Root paths (all gitignored; absolute paths injected only via runtime env) ──
ROOT        = Path(os.environ.get("POST_CRISP_ROOT", ".")).resolve()
CACHE_DIR   = Path(os.environ.get("CACHE_DIR",   ROOT / "cache"))
PREDS_DIR   = Path(os.environ.get("PREDS_DIR",   ROOT / "preds"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))

# Raw/preprocessed data lives under benchmarks/data/<name>/ (anchored to the real
# folder, independent of cwd).
# NOTE: kept under data/ to avoid a name clash with benchmarks/spatialscore.py (module).
BENCH_DIR   = Path(__file__).resolve().parent
DATA_DIR    = Path(os.environ.get("BENCH_DATA_DIR", BENCH_DIR / "data"))


# ── Build one ms-swift jsonl row ─────────────────────────────────────────────
def swift_record(
    uid: Any,
    text: str | None = None,
    images: list[str] | None = None,
    videos: list[str] | None = None,
    meta: dict[str, Any] | None = None,
    *,
    messages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Build one row that ms-swift `swift infer --val_dataset` consumes.
      - Default (text given): a single user turn with one <image>/<video> placeholder
        per image/video prepended before the content.
      - Pass `messages` for full control (multi-turn / interleaved <image> tokens /
        a leading system|assistant instruction). The caller owns tag placement then;
        the <image>/<video> tokens are matched to images/videos in order by ms-swift
        (see StdTemplateInputs.remove_messages_media). `text` is ignored if given.
      - Keep images/videos always as lists (possibly empty) so the schema stays
        consistent within a bench.
      - id / meta are extra keys for scoring (preserved at inference with
        --remove_unused_columns false). meta's key set must be consistent within a
        bench, otherwise datasets schema inference breaks.
    """
    images = list(images or [])
    videos = list(videos or [])
    if messages is None:
        tags = "<image>" * len(images) + "<video>" * len(videos)
        messages = [{"role": "user", "content": tags + (text or "")}]
    return {
        "messages": messages,
        "images": images,
        "videos": videos,
        "id": uid,
        "meta": meta or {},
    }


# ── Model registry entry ─────────────────────────────────────────────────────
@dataclass
class Model:
    tag: str                                  # unique alias (used in paths/result folders), e.g. "qwen3.5-27b"
    path: str                                 # HF repo id or local ckpt path (absolute, runtime)
    subfolder: str | None = None              # subfolder within `path` (HF repo with multiple ckpts), e.g. SFT dirs
    backend: str = "vllm"                     # "vllm" | "pt"
    enable_thinking: bool | None = None       # Qwen3.5 etc. hybrid-thinking -> False to disable
    max_pixels: int | None = None             # to align with the test_qwen protocol (else adapter default)
    min_pixels: int | None = None
    vllm_max_model_len: int | None = None     # vLLM context cap (--vllm_max_model_len); omit -> model config default
    vllm_tensor_parallel_size: int | None = None   # TP degree; omit -> test.py auto = #CUDA_VISIBLE_DEVICES
    extra: dict[str, Any] = field(default_factory=dict)   # pass-through for extra options

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Model":
        known = {"tag", "path", "subfolder", "backend", "enable_thinking",
                 "max_pixels", "min_pixels", "vllm_max_model_len", "vllm_tensor_parallel_size"}
        return cls(
            **{k: d[k] for k in known if k in d},
            extra={k: v for k, v in d.items() if k not in known},
        )


# ── Adapter registry (name -> class) ─────────────────────────────────────────
# Note: an adapter module must be imported to be registered.
#       benchmarks/__init__.py imports all adapters to populate REGISTRY.
REGISTRY: dict[str, type["BenchmarkAdapter"]] = {}


def register(cls: type["BenchmarkAdapter"]) -> type["BenchmarkAdapter"]:
    """Decorator to attach to an adapter class."""
    if not getattr(cls, "name", ""):
        raise ValueError(f"{cls.__name__}: a `name` class attribute is required.")
    if cls.name in REGISTRY:
        raise ValueError(f"duplicate benchmark name: {cls.name!r}")
    REGISTRY[cls.name] = cls
    return cls


def get_adapter(name: str) -> "BenchmarkAdapter":
    if name not in REGISTRY:
        raise KeyError(f"unregistered benchmark: {name!r}. registered: {list_adapters()}")
    return REGISTRY[name]()


def list_adapters() -> list[str]:
    return sorted(REGISTRY)


def resolve(names: list[str] | None, all_: bool = False) -> list["BenchmarkAdapter"]:
    """Runner selector. --benchmarks a,b  or  --all. (guards against running everything by mistake: error if neither)"""
    if all_:
        chosen = list_adapters()
    elif names:
        chosen = names
    else:
        raise SystemExit("specify a benchmark (--benchmarks a,b or --all).")
    return [get_adapter(n) for n in chosen]


# ── Adapter base ─────────────────────────────────────────────────────────────
class BenchmarkAdapter(ABC):
    """
    A box that holds 'how to handle' one benchmark.
    Only the per-bench parts (raw loading / prompts / scoring) are implemented via
    the 4 methods below.
    """

    name: str = ""                # unique name (@register key). must be set in the subclass.
    # Note: scoring runs in whatever conda env is active when evaluate.py is invoked.
    # The adapter does NOT hardcode an env name; each bench's scoring deps are documented
    # in the README — activate an env that has them before scoring.

    # ── Implemented by each bench (the only place things diverge) ────────────
    @abstractmethod
    def load_raw(self) -> list[dict[str, Any]]:
        """Load raw data as a list of dicts. (called only in env1 -> heavy imports go inside)"""

    @abstractmethod
    def to_messages(self, row: dict[str, Any]) -> dict[str, Any]:
        """
        One raw row -> one ms-swift jsonl row. (model-agnostic)
          - build messages(+ <image>) / images(absolute paths) + inject the prompt
          - carry scoring meta (id / gt / sub_task) as extra keys (--remove_unused_columns false)
          - return None to drop this sample (preprocess filters it out; e.g. skip video)
        NOTE: per-model settings like enable_thinking / max_pixels do NOT go here.
          test.py handles them at inference time via models.yaml -> swift infer flags
          (the input jsonl is model-agnostic).
        """

    @abstractmethod
    def reshape(self, preds_path: Path, out_dir: Path) -> None:
        """Prediction output (preds jsonl) -> reshape into this bench's scorer schema and write to out_dir."""

    @abstractmethod
    def score(self, in_dir: Path) -> dict[str, Any]:
        """
        Score -> return a dict of metrics. (called by evaluate.py; runs in whatever conda
        env is active — activate one with this bench's scoring deps, see README)
        Shell out if there is an official harness (e.g. SpatialScore evaluate_results.py),
        otherwise use a custom metric (e.g. RoboRefer IoU).
        """

    # ── Data preparation (called by data_preparation.py) ─────────────────────
    @property
    def data_dir(self) -> Path:
        """This bench's raw/preprocessed data folder. benchmarks/data/<name>/"""
        return DATA_DIR / self.name

    def ensure_data(self) -> None:
        """
        Download raw data (e.g. from HF) into data_dir if missing. (idempotent)
        Heavy deps (huggingface_hub, ...) are lazy-imported inside.
        """
        raise NotImplementedError(f"{self.name}: ensure_data() not implemented")

    def preprocess(self) -> Path:
        """
        raw -> to_messages -> write ms-swift jsonl to data_dir/<name>.jsonl. (model-agnostic)

        Store the fingerprint (md5) of the processed recs in a sidecar (.<name>.jsonl.sha).
        If data/prompts change, the fingerprint changes and it auto-regenerates ->
        no manual cache clearing / --force needed.
        No per-model settings here (the input jsonl is model-agnostic); test.py handles
        those via swift flags at inference time.
        """
        raw = self.load_raw()
        recs = [rec for r in raw if (rec := self.to_messages(r)) is not None]
        skipped = len(raw) - len(recs)
        blob = json.dumps(recs, sort_keys=True, ensure_ascii=False).encode("utf-8")
        fp = hashlib.md5(blob).hexdigest()

        out = self.data_dir / f"{self.name}.jsonl"
        sha = self.data_dir / f".{self.name}.jsonl.sha"
        if out.exists() and sha.exists() and sha.read_text().strip() == fp:
            print(f"[preprocess skip] {out}  ({len(recs)} samples, fp={fp[:12]})")
            return out

        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".jsonl.tmp")            # atomic write: guard against half-written files
        with open(tmp, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, out)
        sha.write_text(fp)
        skip_note = f", {skipped} skipped" if skipped else ""
        print(f"[preprocess ok ] {out}  ({len(recs)} samples{skip_note}, fp={fp[:12]})")
        return out

    # ── Path/completion contract (test.py <-> evaluate.py communicate only via these files) ──
    def preds_path(self, model: Model) -> Path:
        return PREDS_DIR / model.tag / f"{self.name}.jsonl"

    def done_flag(self, model: Model) -> Path:
        return PREDS_DIR / model.tag / f"{self.name}.done.json"

    def results_dir(self, model: Model) -> Path:
        return RESULTS_DIR / model.tag / self.name

    def mark_done(self, model: Model, n: int) -> None:
        """Called by test.py when inference finishes cleanly -> mark done (+ record expected sample count)."""
        p = self.done_flag(model)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"n": n, "ts": time.time()}), encoding="utf-8")

    def is_complete(self, model: Model) -> bool:
        """
        evaluate.py decides 'is it OK to score'.
        done.flag exists + preds line count == recorded n  ->  complete
        (avoids mis-scoring partial output left by a crash mid-run).
        """
        preds, flag = self.preds_path(model), self.done_flag(model)
        if not (preds.exists() and flag.exists()):
            return False
        try:
            expected = json.loads(flag.read_text())["n"]
        except Exception:
            return False
        return _count_lines(preds) == expected


def _count_lines(p: Path) -> int:
    with open(p, "rb") as f:
        return sum(1 for _ in f)
