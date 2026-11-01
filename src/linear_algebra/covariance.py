"""
src/linear_algebra/covariance.py
Phase 3 – Covariance computation uses matrix_ops primitives.

This module will contain compute_covariance_matrix etc in later tasks.
For Task 3.1, it re-exports matrix utilities for convenience.
"""

from .matrix_ops import (
    center_columns,
    transpose,
    matrix_multiply,
    quadratic_form,
    is_symmetric,
)

__all__ = ["center_columns", "transpose", "matrix_multiply", "quadratic_form", "is_symmetric"]