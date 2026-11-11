"""Commerge C++ core for Python: the same scan selection (steps A, B, C) as ``commerge``, in C++.

    import commerge_cpp as cc
    sel = cc.select(affinity, pos_a, pos_b, threshold=0.96, min_area=2000)
    sel.keyframes_a, sel.keyframes_b, sel.pairs

Wheels from PyPI use HiGHS; ``registration_available()`` is True only for source builds with
KISS-Matcher and small_gicp (``pip install ./cpp``), ``gurobi_available()`` when Gurobi was found at build time.
"""
from ._core import *  # noqa: F401,F403
from ._core import __version__  # noqa: F401
