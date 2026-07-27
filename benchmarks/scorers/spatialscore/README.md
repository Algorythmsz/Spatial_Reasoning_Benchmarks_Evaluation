# Vendored: official SpatialScore scorer

`evaluate_results.py` (+ `utils/`) is a **byte-identical copy** of the official SpatialScore
evaluation code, vendored so scoring is self-contained and reproducible without an upstream
checkout. Upstream's MIT `LICENSE` sits next to it.

**Treat these files as read-only.** Editing them silently changes what our numbers mean and
breaks the "identical to upstream" guarantee that makes them comparable to the published
results. To score with a modified scorer, point the harness at your own checkout instead:

```bash
export SS_SCORER=/path/to/your/evaluate_results.py
```

Resolution logic: `benchmarks/spatialscore.py` uses `$SS_SCORER` if set, else this copy.
