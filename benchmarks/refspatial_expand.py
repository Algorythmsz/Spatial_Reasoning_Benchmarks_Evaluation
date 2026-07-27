"""RefSpatial-Expand — the expanded release: Location (241) + Placement (200), no Unseen.

Everything below the dataset identity lives in refspatial.py: the per-model grounding
prompt, the five coordinate parsers, and point-in-mask scoring.
"""
from __future__ import annotations

from .base import register
from .refspatial_base import RefSpatialBase


@register
class RefSpatialExpandAdapter(RefSpatialBase):
    name = "refspatial_expand"
    HF_REPO = "JingkunAn/RefSpatial-Expand-Bench"
    SUBSETS = ("Location", "Placement")
