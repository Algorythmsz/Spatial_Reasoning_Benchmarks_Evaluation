"""benchmarks package.

Adapter modules must be imported here so that @register runs and populates base.REGISTRY.
When adding a new bench, add one line to this list.
"""

from . import base  # noqa: F401
from . import spatialscore  # noqa: F401
from . import multihopspatial  # noqa: F401
from . import refspatial_expand  # noqa: F401

__all__ = ["base", "spatialscore", "multihopspatial", "refspatial_expand"]
