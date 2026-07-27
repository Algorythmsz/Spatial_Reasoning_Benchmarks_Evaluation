"""RefSpatial-Bench — the original release accompanying RoboRefer: Location, Placement, Unseen.

Everything below the dataset identity lives in refspatial.py: the per-model grounding
prompt, the five coordinate parsers, and point-in-mask scoring.
"""
from __future__ import annotations

from .base import register
from .refspatial import RefSpatialBase


@register
class RefSpatialBenchAdapter(RefSpatialBase):
    name = "refspatial_bench"
    HF_REPO = "BAAI/RefSpatial-Bench"
    SUBSETS = ("Location", "Placement", "Unseen")
