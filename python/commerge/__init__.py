"""Commerge: communication-efficient multi-robot map merging (LiDAR, radar, sonar, any place descriptor)."""
__version__ = "2.1.0"

from .selection import STEPS, Selection, cosine_affinity, enhance_affinity, extract_regions, select  # noqa: E402,F401
from .session import Session, read_pcd, write_pcd  # noqa: E402,F401
from .merge import POLICIES, alignment_error, choose_regions, register, register_exchanged, register_grid, register_pairs  # noqa: E402,F401
