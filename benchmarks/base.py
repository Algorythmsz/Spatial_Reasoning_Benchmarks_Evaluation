from __future__ import annotations

import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── Root paths (absolute paths injected only via runtime env) ──
ROOT        = Path(os.environ.get("POST_CRISP_ROOT", ".")).resolve()
CACHE_DIR   = Path(os.environ.get("CACHE_DIR",   ROOT / "cache"))
PREDS_DIR   = Path(os.environ.get("PREDS_DIR",   ROOT / "preds"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))


def _expand_path(p: str) -> str:
    """models.yaml `path` -> concrete path, with ~ and $VARS expanded.

    Lets a LOCAL checkpoint be written as `${POST_CRISP_ROOT}/sft/merged/...` instead of one
    machine's absolute path, so models.yaml stays committable — same rule as the module docstring
    above: absolute paths are injected via runtime env, never hardcoded. HF repo ids
    ("Qwen/Qwen3-VL-4B-Instruct") contain no $ or ~, so this is a no-op for them.

    POST_CRISP_ROOT falls back to ROOT so an unset env behaves like the rest of the repo
    (cwd) rather than leaving a literal "${POST_CRISP_ROOT}" in the path.
    """
    p = p.replace("${POST_CRISP_ROOT}", str(ROOT)) if "POST_CRISP_ROOT" not in os.environ else p
    return os.path.expanduser(os.path.expandvars(p))


# Raw/preprocessed data lives under benchmarks/data/<name>/ (anchored to the real
# folder).
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


# ── Model registry entry; it loads from 'models.yaml' ─────────────────────────────────────────────────────
@dataclass
class Model:
    tag: str                                  # unique alias (used in paths/result folders), e.g. "qwen3.5-27b"
    path: str                                 # HF repo id or local ckpt path (absolute, runtime)
    subfolder: str | None = None              # subfolder within `path` (HF repo with multiple ckpts), e.g. SFT dirs
    model_type: str | None = None             # force ms-swift model_type (e.g. "qwen3_vl") when auto-match is ambiguous
    backend: str = "vllm"                     # "vllm" | "pt"
    enable_thinking: bool | None = None       # Qwen3.5 etc. hybrid-thinking -> False to disable
    max_pixels: int | None = None             # to align with the test_qwen protocol (else adapter default)
    min_pixels: int | None = None
    vllm_max_model_len: int | None = None     # vLLM context cap (--vllm_max_model_len); omit -> model config default
    vllm_tensor_parallel_size: int | None = None   # TP degree; omit -> infer.py auto = #CUDA_VISIBLE_DEVICES
    # Pass-through to vLLM's EngineArgs (swift: --vllm_engine_kwargs). For knobs swift has no
    # dedicated flag for, e.g. max_num_batched_tokens, which also sizes the multimodal encoder
    # cache (vLLM: encoder_cache_size = max(max_num_batched_tokens, max tokens per mm item)).
    vllm_engine_kwargs: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)   # pass-through for extra options

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Model": # models.yaml -> Model
        known = {"tag", "path", "subfolder", "model_type", "backend", "enable_thinking",
                 "max_pixels", "min_pixels", "vllm_max_model_len", "vllm_tensor_parallel_size",
                 "vllm_engine_kwargs"}
        kw = {k: d[k] for k in known if k in d}
        if "path" in kw:
            kw["path"] = _expand_path(kw["path"])
        return cls(
            **kw,
            extra={k: v for k, v in d.items() if k not in known},
        )


# ── models.yaml -> [Model] (shared by infer.py + evaluate.py) ────────────────
MODELS_YAML = Path(os.environ.get("MODELS_YAML", Path(__file__).resolve().parent.parent / "models.yaml"))


def load_models(tags: list[str] | None = None) -> list["Model"]:
    """Load models.yaml -> [Model]. With `tags`, keep only those (request order preserved,
    unknown tags error out); with tags=None, return every model in the file."""
    import yaml                                                 # lazy: keep base.py top-level stdlib-only

    spec = yaml.safe_load(MODELS_YAML.read_text())
    models = [Model.from_dict(m) for m in spec.get("models", [])]
    if tags:                                                    # keep only requested tags, preserving request order
        by_tag = {m.tag: m for m in models}
        missing = [t for t in tags if t not in by_tag]
        if missing:
            raise SystemExit(f"unknown model tag(s): {missing}. known: {sorted(by_tag)}")
        models = [by_tag[t] for t in tags]
    return models


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


def resolve(names: list[str] | None) -> list["BenchmarkAdapter"]:
    """Runner selector. --benchmarks a,b  or  --benchmarks all. (error if none given, to guard against running everything by mistake)"""
    if not names:
        raise SystemExit("specify --benchmarks (comma-separated names, or 'all').")
    chosen = list_adapters() if names == ["all"] else names
    return [get_adapter(n) for n in chosen]


# ── Adapter base ─────────────────────────────────────────────────────────────
class BenchmarkAdapter(ABC):
    """
    A box that holds 'how to handle' one benchmark.
    Only the per-bench parts (raw loading / prompts / scoring) are implemented via
    the 4 methods below.
    """

    name: str = ""                # unique name (@register key). must be set in the subclass.
    # When True, to_messages bakes a model-specific prompt (branches on the Model), so
    # preprocess writes a per-model jsonl (<name>__<tag>.jsonl) instead of the shared one.
    # Default False -> the input jsonl is model-agnostic and reused across all models.
    MODEL_SPECIFIC_PROMPT: bool = False
    # Per-benchmark inference settings, applied by infer.py. Use when a benchmark's official
    # protocol differs from this repo's defaults — the numbers only mean something if the
    # generation settings match the harness we're comparing against.
    #   max_new_tokens: int   -> overrides infer.py's 512 (a --max-new-tokens flag still wins)
    #   pin_pixels: bool      -> False leaves image resolution at the model default, i.e.
    #                            ignores models.yaml's min/max_pixels for this bench
    #   min_pixels / max_pixels: int -> pin THESE instead of models.yaml's, for a bench whose
    #                            official protocol specifies its own image budget
    INFER_DEFAULTS: dict[str, Any] = {}
    # Note: scoring runs in whatever conda env is active when evaluate.py is invoked.
    # The adapter does NOT hardcode an env name; each bench's scoring deps are documented
    # in the README — activate an env that has them before scoring.

    # ── Implemented by each bench ────────────
    @abstractmethod
    def load_raw(self) -> list[dict[str, Any]]:
        """Load raw data as a list of dicts. (called only in env1 -> heavy imports go inside)"""

    @abstractmethod
    def to_messages(self, row: dict[str, Any], model: "Model | None" = None) -> dict[str, Any] | None:
        """
        One raw row -> one ms-swift jsonl row.
          - build messages(+ <image>) / images(absolute paths) + inject the prompt
          - carry scoring meta (id / gt / sub_task) as extra keys (--remove_unused_columns false)
          - return None to drop this sample (preprocess filters it out; e.g. skip video)
        `model` is passed only so a bench can branch the PROMPT per model (e.g. an official
          per-model grounding prompt); set MODEL_SPECIFIC_PROMPT=True when you use it. Most
          benches ignore it and stay model-agnostic.
        NOTE: per-model INFERENCE settings (enable_thinking / max_pixels / backend) do NOT
          go here — infer.py applies them via models.yaml -> swift infer flags. Only prompt
          TEXT that legitimately differs per model belongs here.
        """

    @abstractmethod
    def reshape(self, preds_path: Path, out_dir: Path) -> None:
        """Prediction output (preds jsonl) -> reshape into this bench's scorer schema and write to out_dir."""

    @abstractmethod
    def score(self, in_dir: Path, **opts: Any) -> dict[str, Any]:
        """
        Score -> return a dict of metrics. (called by evaluate.py; runs in whatever conda
        env is active — activate one with this bench's scoring deps, see README)
        Shell out if there is an official harness (e.g. SpatialScore evaluate_results.py),
        otherwise use a custom metric (e.g. RoboRefer IoU).
        `**opts` are USER-supplied scoring options forwarded from evaluate.py's CLI (e.g.
          refspatial_expand's parse_function). Most benches ignore them.
        """

    # ── Data preparation (ensure_data: data_preparation.py; preprocess: infer.py) ────
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

    def _input_paths(self, model: "Model | None" = None) -> tuple[Path, Path]:
 
        if self.MODEL_SPECIFIC_PROMPT and model is None:
            raise ValueError(
                f"{self.name}: the prompt depends on the model (MODEL_SPECIFIC_PROMPT), so "
                f"the input jsonl cannot be built without one. Pass a Model.")
        stem = f"{self.name}__{model.tag}" if self.MODEL_SPECIFIC_PROMPT else self.name
        return self.data_dir / f"{stem}.jsonl", self.data_dir / f".{stem}.jsonl.sha"

    def input_fingerprint(self, model: "Model | None" = None) -> str | None:
        """md5 of the CURRENT input records, read from the sidecar preprocess wrote.
        None when the input has never been built. Cheap — no re-preprocessing."""
        _, sha = self._input_paths(model)
        try:
            return sha.read_text().strip()
        except OSError:
            return None

    def preprocess(self, model: "Model | None" = None) -> Path:
        """
        raw -> to_messages -> write the ms-swift jsonl under data_dir.

        Store the fingerprint (md5) of the processed recs in a sidecar (.<stem>.jsonl.sha).
        If data/prompts change, the fingerprint changes and it auto-regenerates ->
        no manual cache clearing / --force needed.

        Model handling:
          - MODEL_SPECIFIC_PROMPT=False (default): the prompt is model-agnostic, so one
            shared <name>.jsonl is built once and reused across every model (`model` unused).
          - MODEL_SPECIFIC_PROMPT=True: to_messages bakes a per-model prompt, so we write a
            model-scoped <name>__<tag>.jsonl. This keeps each model's fingerprint separate
            (no regeneration thrashing when alternating models in one infer run).
        Per-model INFERENCE settings (pixels / thinking / backend) are NOT applied here;
        infer.py injects them as swift flags at inference time.
        """
        raw = self.load_raw()
        recs = [rec for r in raw if (rec := self.to_messages(r, model)) is not None]
        skipped = len(raw) - len(recs)
        blob = json.dumps(recs, sort_keys=True, ensure_ascii=False).encode("utf-8")
        fp = hashlib.md5(blob).hexdigest()

        out, sha = self._input_paths(model)                    # per-model jsonl when the prompt varies
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

    # ── Path/completion contract (infer.py <-> evaluate.py communicate only via these files) ──
    def preds_path(self, model: Model) -> Path:
        return PREDS_DIR / model.tag / f"{self.name}.jsonl"

    def done_flag(self, model: Model) -> Path:
        return PREDS_DIR / model.tag / f"{self.name}.done.json"

    def results_dir(self, model: Model) -> Path:
        return RESULTS_DIR / model.tag / self.name

    def mark_done(self, model: Model, n: int, config: dict[str, Any] | None = None) -> None:
        """Called by infer.py when inference finishes cleanly -> mark done.

        Records the expected sample count AND the settings that produced these predictions
        (generation budget, pixel bounds, resolved checkpoint, input fingerprint). Without
        that, nothing downstream can tell WHICH protocol a metrics.json came from — the only
        evidence left is file mtimes and job logs.
        """
        p = self.done_flag(model)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {"n": n, "ts": time.time()}
        if config:
            payload["config"] = config
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

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
            recorded = json.loads(flag.read_text())
            expected = recorded["n"]
        except Exception:
            return False
        if _count_lines(preds) != expected:
            return False

        # Stale-input guard. If these predictions recorded the fingerprint of the input they
        # were generated from and the current input hashes differently (the prompt or the
        # data changed), they are not complete FOR THE CURRENT PROTOCOL — scoring them would
        # silently report numbers from the old one. Runs written before this field existed
        # carry no fingerprint and keep the previous count-only behaviour.
        was = (recorded.get("config") or {}).get("input_fingerprint")
        now = self.input_fingerprint(model)
        if was and now and was != now:
            print(f"[{self.name}/{model.tag}] stale predictions: input fingerprint "
                  f"{was[:12]} != current {now[:12]} — re-run inference")
            return False
        return True


def _count_lines(p: Path) -> int:
    with open(p, "rb") as f:
        return sum(1 for _ in f)
