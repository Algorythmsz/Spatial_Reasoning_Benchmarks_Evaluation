#!/usr/bin/env python
"""
data_preparation.py — benchmark data downloader.

Downloads raw data.

Usage:
    python data_preparation.py spatialscore
    python data_preparation.py multihopspatial
    python data_preparation.py refspatial_bench
    python data_preparation.py refspatial_expand
    python data_preparation.py all
"""

from __future__ import annotations

import argparse
import sys

from benchmarks import base  # importing the package registers every adapter in base.REGISTRY


def main() -> int:
    choices = base.list_adapters()                # dynamic: any registered adapter is valid
    ap = argparse.ArgumentParser(
        description="Download benchmark datasets (infer.py builds the input jsonl)"
    )
    ap.add_argument("benchmark", choices=choices + ["all"],
                    help=f"benchmark name ({' | '.join(choices)}) or 'all'")
    args = ap.parse_args()

    names = choices if args.benchmark == "all" else [args.benchmark]
    failures: list[str] = []

    for name in names:
        print(f"\n===== {name} =====")
        adapter = base.get_adapter(name)
        try:
            adapter.ensure_data()      # download if missing
        except Exception as e:
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
            failures.append(name)

    if failures:
        print(f"Failed: {failures}")
        return 1
    print(f"Success: {names}  (input jsonl is built by infer.py at inference time)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
