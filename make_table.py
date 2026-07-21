#!/usr/bin/env python3
"""Collect scored benchmark results into a table.

Scans results/<model_tag>/<bench>/metrics.json (only models whose scoring
finished — a metrics.json is written last by evaluate.py, so its presence means
the run completed). Models that OOM'd during inference or failed in the judge
have no metrics.json and are listed as skipped, not silently dropped.

Usage:
    POST_CRISP_ROOT="your directory" python make_table.py --bench spatialscore
    python make_table.py --bench spatialscore --breakdown category --csv out.csv

Env: POST_CRISP_ROOT (default ".") / RESULTS_DIR override the results location,
matching benchmarks/base.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("POST_CRISP_ROOT", ".")).resolve()
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
# Generated tables land here (POST_CRISP_ROOT/table/); a bare --csv name resolves into it.
TABLE_DIR = Path(os.environ.get("TABLE_DIR", ROOT / "table"))

# metrics.json groups accuracies under these keys; each maps name -> {accuracy, count}.
BREAKDOWNS = ("category", "task", "sub_task", "source_dataset")


def load_metrics(bench: str) -> tuple[dict[str, dict], list[str]]:
    """Return {model_tag: metrics} for models with a metrics.json, and the list of
    model dirs that exist but were not (successfully) scored."""
    scored: dict[str, dict] = {}
    skipped: list[str] = []
    if not RESULTS_DIR.is_dir():
        sys.exit(f"results dir not found: {RESULTS_DIR} (set POST_CRISP_ROOT/RESULTS_DIR)")
    for model_dir in sorted(RESULTS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        mpath = model_dir / bench / "metrics.json"
        if mpath.is_file():
            try:
                scored[model_dir.name] = json.loads(mpath.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                skipped.append(f"{model_dir.name} (bad metrics.json: {e})")
        elif (model_dir / bench).is_dir():
            skipped.append(model_dir.name)
    return scored, skipped


def pct(x: float) -> str:
    return f"{x * 100:.2f}"


def build_rows(scored: dict[str, dict], breakdown: str | None) -> tuple[list[str], list[list[str]]]:
    """Header + rows. Always includes Overall; optionally a per-breakdown column set."""
    # Column order for the breakdown = union of names across models, keeping first-seen order.
    sub_cols: list[str] = []
    if breakdown:
        for m in scored.values():
            for name in m.get(breakdown, {}):
                if name not in sub_cols:
                    sub_cols.append(name)

    header = ["model", "n", "overall"] + sub_cols
    rows: list[list[str]] = []
    for tag, m in sorted(scored.items(), key=lambda kv: kv[1].get("overall", {}).get("accuracy", 0), reverse=True):
        overall = m.get("overall", {})
        row = [tag, str(overall.get("count", "")), pct(overall.get("accuracy", 0.0))]
        for name in sub_cols:
            cell = m.get(breakdown, {}).get(name)
            row.append(pct(cell["accuracy"]) if cell else "-")
        rows.append(row)
    return header, rows


def print_table(header: list[str], rows: list[list[str]]) -> None:
    widths = [len(h) for h in header]
    for r in rows:
        widths = [max(w, len(c)) for w, c in zip(widths, r)]
    line = lambda cols: "  ".join(c.ljust(w) if i == 0 else c.rjust(w)
                                  for i, (c, w) in enumerate(zip(cols, widths)))
    print(line(header))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(line(r))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True, help="benchmark name (results/<tag>/<bench>/)")
    ap.add_argument("--breakdown", choices=BREAKDOWNS, help="add per-group columns (default: overall only)")
    ap.add_argument("--csv", help="also write the table to this CSV path")
    args = ap.parse_args()

    scored, skipped = load_metrics(args.bench)
    if not scored:
        print(f"[table] no scored models found under {RESULTS_DIR}/*/{args.bench}/metrics.json")
        if skipped:
            print(f"[table] present but unscored: {', '.join(skipped)}")
        return 1

    header, rows = build_rows(scored, args.breakdown)
    print(f"# {args.bench} — accuracy (%), {len(scored)} model(s) scored, sorted by overall\n")
    print_table(header, rows)
    if skipped:
        print(f"\n[table] present but not scored ({len(skipped)}): {', '.join(skipped)}")

    if args.csv:
        out = Path(args.csv)
        if not out.is_absolute():                 # bare name -> under POST_CRISP_ROOT/table/
            out = TABLE_DIR / out
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        print(f"\n[table] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
