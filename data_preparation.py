#!/usr/bin/env python
"""data_preparation.py — benchmark data download + ms-swift preprocessing orchestrator.

For each bench:
  1) ensure_data()  — download raw data from HF into benchmarks/data/<name>/ .
  2) preprocess()   — raw -> ms-swift jsonl (benchmarks/data/<name>/<name>.jsonl).

Usage:
    python data_preparation.py spatialscore
    python data_preparation.py multihopspatial
    python data_preparation.py refspatial_expand
    python data_preparation.py all              # all three benches
"""

from __future__ import annotations

import argparse
import sys

from benchmarks import base 

CHOICES = ["spatialscore", "multihopspatial", "refspatial_expand"]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Download and preprocess benchmark datasets"
    )
    ap.add_argument("benchmark", choices=CHOICES + ["all"], help="benchmark names ['spatialscore', 'multihopspatial', 'refspatial_expand'](or all)")
    args = ap.parse_args()

    names = CHOICES if args.benchmark == "all" else [args.benchmark]
    failures: list[str] = []

    for name in names:
        print(f"\n===== {name} =====")
        adapter = base.get_adapter(name)
        try:
            adapter.ensure_data()      # download if missing
            adapter.preprocess()       # build ms-swift jsonl (auto-regenerated when fingerprint changes)
        except Exception as e:
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
            failures.append(name)

    if failures:
        print(f"Failed: {failures}")
        return 1
    print(f"Success: {names}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
