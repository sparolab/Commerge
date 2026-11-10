"""Commerge: only a small subset of carefully selected scans is sufficient for map merging."""
__version__ = "1.1.0"

from .selection import STEPS, Selection, cosine_affinity, extract_regions, select  # noqa: E402,F401
from .session import Session, read_pcd, write_pcd  # noqa: E402,F401
from .merge import POLICIES, alignment_error, choose_regions, register  # noqa: E402,F401
