#!/usr/bin/env python
"""evaluate.py — scoring orchestrator.

Drives the SCORING half of the pipeline. For each (benchmark, model):
  1) is_complete(model)          — gate: only score cleanly-finished inference
  2) reshape(preds -> results)   — swift preds jsonl -> the scorer's input schema
  3) score(results) -> metrics   — bench-specific scoring (may shell out)
  4) write metrics.json          — persisted under results_dir(model)

★ Activate a scoring env YOURSELF before running (see README.md)
  This script runs scoring in whatever conda env is currently active — it does not
  switch or check envs. Each benchmark needs certain deps (README lists them); make
  sure the active env has them before running.

Usage:
    conda activate <an env with the scorer's deps>
    python evaluate.py --benchmarks spatialscore --models qwen3.5-27b,qwen3vl-8b
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from benchmarks import base  # importing the package registers all adapters

MODELS_YAML = Path(os.environ.get("MODELS_YAML", Path(__file__).resolve().parent / "models.yaml"))


# ── models.yaml -> [Model] ───────────────────────────────────────────────────
def load_models(tags: list[str] | None = None) -> list[base.Model]:
    import yaml

    spec = yaml.safe_load(MODELS_YAML.read_text())
    models = [base.Model.from_dict(m) for m in spec.get("models", [])]
    if tags:                                                   # keep only requested tags, preserving request order
        by_tag = {m.tag: m for m in models}
        missing = [t for t in tags if t not in by_tag]
        if missing:
            raise SystemExit(f"unknown model tag(s): {missing}. known: {sorted(by_tag)}")
        models = [by_tag[t] for t in tags]
    return models


# ── reshape + score for one (adapter, model), in the current env ─────────────
def score_one(adapter: base.BenchmarkAdapter, model: base.Model) -> dict:
    preds = adapter.preds_path(model)                        # swift infer output for this (model, bench)
    results = adapter.results_dir(model)                     # where reshape/score write their artifacts
    results.mkdir(parents=True, exist_ok=True)               # ensure it exists (idempotent)

    adapter.reshape(preds, results)                          # swift preds -> scorer input (e.g. all_results.json)
    metrics = adapter.score(results)                         # bench-specific scoring -> metrics dict

    (results / "metrics.json").write_text(                   # persist the metrics next to the artifacts
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[evaluate] {adapter.name}/{model.tag} metrics -> {results / 'metrics.json'}")
    return metrics


# ── entry point ──────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="Score benchmark predictions (activate an env with the scorer deps; see README).")
    ap.add_argument("--benchmarks", help="comma-separated bench names, or 'all' for every bench")
    ap.add_argument("--models", help="comma-separated model tags from models.yaml, or 'all' for every model")
    args = ap.parse_args()

    names = args.benchmarks.split(",") if args.benchmarks else None
    adapters = base.resolve(names)                            # selected adapters (or error if none specified)
    if not args.models:
        raise SystemExit("specify --models (comma-separated tags from models.yaml, or 'all').")
    tags = None if args.models == "all" else args.models.split(",")  # 'all' -> every model in models.yaml
    models = load_models(tags)                               # selected models

    failures: list[str] = []
    for adapter in adapters:
        for model in models:
            if not adapter.is_complete(model):               # gate: skip partial/absent inference
                print(f"[evaluate] skip {adapter.name}/{model.tag}: inference not complete")
                continue
            try:
                score_one(adapter, model)                    # reshape + score in the current env
            except Exception as e:
                print(f"[evaluate] FAIL {adapter.name}/{model.tag}: {type(e).__name__}: {e}")
                failures.append(f"{adapter.name}/{model.tag}")

    if failures:
        print(f"[evaluate] failures: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
