"""
sppa — Sparse Projection Pursuit Analysis
==========================================

Usage
-----
>>> from sppa import sppa
>>> T, V, Var, kurt = sppa(X, dim=2, nvars=5, meth='uni')
"""

from .core import sppa

__all__ = ["sppa"]
