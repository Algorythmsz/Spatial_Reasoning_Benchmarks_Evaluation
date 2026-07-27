"""swift_backend.py — run the vendored official evaluator on ms-swift's engine.

`benchmark_qwen_vllm.py` next to this file is a byte-identical copy of the official
MultihopSpatial evaluator and must stay that way (see README.md). It talks to vLLM through
exactly two names, both looked up on its own module at call time:

    llm = LLM(**llm_kwargs)                                   # module-level `LLM`
    sp  = SamplingParams(**params)                            # module-level `SamplingParams`
    out = llm.chat(messages=convos, sampling_params=sp)       # out[i].outputs[0].text

So instead of editing it, we rebind those two names to the shims below. Everything the
protocol consists of — prompt, retry rounds, response parsing, bbox scaling, IoU, the
hop x view metric table — runs upstream's code untouched.

What the shims add on top of plain vLLM, because the official script is Qwen3-VL-only and
our registry is not:
  - model_type      FT checkpoints match several ms-swift types -> must be forced
  - enable_thinking a Template ctor arg (swift/template/base.py); Qwen3.5 needs it False
Set both via `configure()` before calling upstream's `run_benchmark`.

NOT set here, deliberately: min_pixels / max_pixels. The official script leaves image
resolution at the model default, so we do too — pinning it (as our other benchmarks do for
the SpatialScore protocol) would change image-token counts and stop being a reproduction.
"""
from __future__ import annotations

from typing import Any

_CONFIG: dict[str, Any] = {"model_type": None, "enable_thinking": None}


def configure(*, model_type: str | None = None, enable_thinking: bool | None = None) -> None:
    """Per-model ms-swift settings the official (Qwen3-VL-only) script has no notion of."""
    _CONFIG["model_type"] = model_type
    _CONFIG["enable_thinking"] = enable_thinking


class SamplingParams:
    """Stand-in for vllm.SamplingParams: upstream only constructs it and hands it back to
    `chat`, so carrying the kwargs through to a RequestConfig is enough."""

    def __init__(self, **params: Any) -> None:
        self.params = params

    def __repr__(self) -> str:                                 # upstream logs the object
        return f"SamplingParams({self.params})"

    def to_request_config(self):
        from swift import RequestConfig

        p = dict(self.params)
        return RequestConfig(
            max_tokens=p.pop("max_tokens", None),
            temperature=p.pop("temperature", None),
            top_p=p.pop("top_p", None),
            top_k=p.pop("top_k", None),
            repetition_penalty=p.pop("repetition_penalty", None),
            seed=p.pop("seed", None),
        )


def _to_infer_request(conversation: list[dict[str, Any]]):
    """Upstream's OpenAI-style turn -> ms-swift InferRequest.

    Upstream builds exactly one user turn whose content is a list of parts:
        [{"type": "image_url", "image_url": {"url": "file:///abs.jpg"}},
         {"type": "text", "text": prompt}]
    ms-swift wants plain-string content with one `<image>` placeholder per image, plus a
    separate `images` list. Placeholders go first, matching upstream's part order (image
    before text) so the image precedes the question in the rendered prompt.
    """
    from swift import InferRequest

    images: list[str] = []
    messages: list[dict[str, str]] = []
    for turn in conversation:
        content = turn.get("content")
        if isinstance(content, str):
            messages.append({"role": turn["role"], "content": content})
            continue
        texts: list[str] = []
        n_img = 0
        for part in content or []:
            if part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                images.append(url[len("file://"):] if url.startswith("file://") else url)
                n_img += 1
            elif part.get("type") == "text":
                texts.append(part.get("text", ""))
        messages.append({"role": turn["role"], "content": "<image>" * n_img + "".join(texts)})
    return InferRequest(messages=messages, images=images)


class _Choice:
    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = text


class _Output:
    """Mimics vLLM's RequestOutput far enough for upstream's `output.outputs[0].text`."""

    __slots__ = ("outputs",)

    def __init__(self, text: str) -> None:
        self.outputs = [_Choice(text)]


class LLM:
    """Stand-in for vllm.LLM backed by `swift.VllmEngine`.

    Accepts the kwargs upstream passes and maps the ones ms-swift exposes; the rest are
    handled by ms-swift itself (`trust_remote_code`, local media access) and dropped with a
    log rather than silently, so a future upstream kwarg can't vanish unnoticed.
    """

    _MAPPED = {"model", "max_model_len", "gpu_memory_utilization", "tensor_parallel_size",
               "enforce_eager", "max_num_seqs", "dtype"}

    def __init__(self, **llm_kwargs: Any) -> None:
        from swift import VllmEngine, get_model_processor, get_template

        model_path = llm_kwargs["model"]
        dropped = sorted(set(llm_kwargs) - self._MAPPED)
        if dropped:
            print(f"[swift_backend] vLLM kwargs handled by ms-swift, not forwarded: {dropped}")

        model_type = _CONFIG["model_type"]
        enable_thinking = _CONFIG["enable_thinking"]

        # Build the template ourselves only when we must: enable_thinking is a Template ctor
        # arg, so VllmEngine's default template cannot receive it.
        template = None
        if enable_thinking is not None:
            _, processor = get_model_processor(model_path, load_model=False, use_hf=True,
                                               model_type=model_type)
            template = get_template(processor, enable_thinking=enable_thinking,
                                    remove_unused_columns=False)

        kwargs: dict[str, Any] = {"use_hf": True}
        if template is not None:
            kwargs["template"] = template
        if model_type is not None:
            kwargs["model_type"] = model_type
        for src, dst in (("max_model_len", "max_model_len"),
                         ("gpu_memory_utilization", "gpu_memory_utilization"),
                         ("tensor_parallel_size", "tensor_parallel_size"),
                         ("enforce_eager", "enforce_eager"),
                         ("max_num_seqs", "max_num_seqs")):
            if llm_kwargs.get(src) is not None:
                kwargs[dst] = llm_kwargs[src]
        if llm_kwargs.get("dtype"):
            import torch

            dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                     "float32": torch.float32}.get(llm_kwargs["dtype"])
            if dtype is not None:
                kwargs["torch_dtype"] = dtype

        print(f"[swift_backend] VllmEngine({model_path}, "
              f"{dict(kwargs, template='<Template>' if template is not None else None)})")
        self.engine = VllmEngine(model_path, **kwargs)

    def chat(self, messages: list, sampling_params: SamplingParams, **_: Any) -> list[_Output]:
        requests = [_to_infer_request(c) for c in messages]
        responses = self.engine.infer(requests, sampling_params.to_request_config())
        return [_Output(r.choices[0].message.content or "") for r in responses]

    def release(self) -> None:
        """Free the GPU so the next model can load."""
        import gc

        self.engine = None
        gc.collect()
        try:
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )
            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception:
            pass
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass


def install() -> Any:
    """Rebind `LLM` / `SamplingParams` on the vendored module and return it."""
    import importlib

    upstream = importlib.import_module(
        "benchmarks.scorers.multihopspatial.benchmark_qwen_vllm")
    upstream.LLM = LLM
    upstream.SamplingParams = SamplingParams
    return upstream
