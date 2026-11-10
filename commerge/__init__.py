"""Commerge: communication-efficient multi-robot LiDAR map merging."""
__version__ = "2.0.0"

from .selection import STEPS, Selection, cosine_affinity, enhance_affinity, extract_regions, select  # noqa: E402,F401
from .session import Session, read_pcd, write_pcd  # noqa: E402,F401
from .merge import POLICIES, alignment_error, choose_regions, register, register_exchanged, register_grid  # noqa: E402,F401
