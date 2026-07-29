"""RefSpatial-Bench — the original release accompanying RoboRefer: Location, Placement, Unseen.

Everything below the dataset identity lives in refspatial_base.py: the per-model grounding
prompt, the five coordinate parsers, and point-in-mask scoring.
"""
from __future__ import annotations

from .base import register
from .refspatial_base import RefSpatialBase


@register
class RefSpatialBenchAdapter(RefSpatialBase):
    name = "refspatial_bench"
    HF_REPO = "BAAI/RefSpatial-Bench"
    SUBSETS = ("Location", "Placement", "Unseen")
    # The official repo warns: "If your model is not trained with RefSpatial, Unseen set
    # should not be used for evaluation." None of the models here is, so Unseen is scored and
    # reported per-subset but left out of `overall` (200 samples, not 277). Its effect is
    # large — 2-6 pp across our models — so this is not a rounding-level choice.
    OVERALL_EXCLUDE_SUBSETS = ("Unseen",)
