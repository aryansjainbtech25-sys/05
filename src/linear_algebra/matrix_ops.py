"""
src/linear_algebra/matrix_ops.py
Phase 3.1 – Matrix Utilities: Centering, Transpose, and Quadratic Form

Primitives operating on numpy.ndarray for math layer, with thin adapters for pd.DataFrame.
All shapes documented, dimension checks explicit.

Mathematical definitions:
- Transpose: (X^T)_{ij} = X_{ji}. If X is (T×N), X^T is (N×T), X^T X is (N×N).
- Centering: X̃_{t,i} = r_{t,i} - μ_i, μ_i = (1/T) Σ_t r_{t,i}. Column-wise centering.
  Required because covariance is about mean; X^T X without centering is second moment about zero.
- Quadratic form: w^T Σ w = Σ_i Σ_j w_i Σ_{ij} w_j, portfolio variance for symmetric PSD Σ.
- Matrix multiply: (AB)_{ij} = Σ_k A_{ik} B_{kj}
"""

from __future__ import annotations

from typing import Union
import numpy as np
import pandas as pd

__all__ = [
    "center_columns",
    "transpose",
    "matrix_multiply",
    "quadratic_form",
    "is_symmetric",
]


def is_symmetric(M: np.ndarray, tol: float = 1e-8) -> bool:
    """
    Check if matrix is symmetric within tolerance.

    Args:
        M: (N,N) ndarray
        tol: absolute tolerance

    Returns:
        bool
    """
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        return False
    return np.allclose(M, M.T, atol=tol)


def center_columns(X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
    """
    Return X - mean(X, axis=0), column-wise centering.

    Mathematical: X̃_{t,i} = r_{t,i} - μ_i, μ_i = (1/T) Σ_t r_{t,i}

    Args:
        X: (T,N) ndarray or DataFrame, T observations, N tickers

    Shape:
        in: (T,N) -> out: (T,N), each column mean ≈ 0

    Returns:
        Centered matrix same type as input (DataFrame preserves index/columns)

    Raises:
        ValueError if not 2D or empty
    """
    if isinstance(X, pd.DataFrame):
        if X.empty:
            raise ValueError("center_columns received empty DataFrame")
        if X.ndim != 2:
            raise ValueError("center_columns expects 2D input")
        means = X.mean(axis=0)
        centered = X - means
        # Verify
        assert np.allclose(centered.mean(axis=0).values, 0, atol=1e-12)
        return centered

    X = np.asarray(X)
    if X.size == 0:
        raise ValueError("center_columns received empty array")
    if X.ndim != 2:
        raise ValueError(f"center_columns expects 2D array, got {X.ndim}D")
    means = np.mean(X, axis=0)
    centered = X - means
    # Verify column means ~0
    col_means = np.mean(centered, axis=0)
    if not np.allclose(col_means, 0, atol=1e-12):
        raise RuntimeError(f"center_columns failed: column means not zero {col_means}")
    return centered


def transpose(X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
    """
    Return X^T.

    Mathematical: (X^T)_{ij} = X_{ji}

    Args:
        X: (T,N) ndarray or DataFrame

    Shape:
        in: (T,N) -> out: (N,T)

    Returns:
        Transposed matrix

    Raises:
        ValueError if not 2D
    """
    if isinstance(X, pd.DataFrame):
        if X.ndim != 2:
            raise ValueError("transpose expects 2D DataFrame")
        result = X.T
        # Assert shape (T,N) -> (N,T)
        assert result.shape[0] == X.shape[1] and result.shape[1] == X.shape[0]
        return result

    X = np.asarray(X)
    if X.ndim == 1:
        raise ValueError(f"transpose expects 2D array, got 1D of shape {X.shape} – use reshape")
    if X.ndim != 2:
        raise ValueError(f"transpose expects 2D array, got {X.ndim}D")
    T, N = X.shape
    XT = X.T
    assert XT.shape == (N, T), f"transpose shape mismatch: ({T},{N}) -> {XT.shape}, expected ({N},{T})"
    return XT


def matrix_multiply(A: Union[np.ndarray, pd.DataFrame], B: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
    """
    Return A @ B with dimension compatibility assertion.

    Mathematical: (AB)_{ij} = Σ_k A_{ik} B_{kj}

    Args:
        A: (m,k) ndarray/DataFrame
        B: (k,n) ndarray/DataFrame

    Shape:
        in: (m,k) , (k,n) -> out: (m,n)

    Returns:
        Product matrix, DataFrame if both inputs are DataFrames with compatible labels

    Raises:
        ValueError on dimension mismatch or non-2D
    """
    # Handle DataFrame case
    if isinstance(A, pd.DataFrame) and isinstance(B, pd.DataFrame):
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"matrix_multiply dimension mismatch: A {A.shape} columns {A.shape[1]} != B {B.shape} rows {B.shape[0]}")
        # For DataFrames, use @ which aligns on columns? Use values to avoid label issues, but preserve?
        # We'll use numpy for computation and return DataFrame with appropriate index/columns
        result_values = A.values @ B.values
        return pd.DataFrame(result_values, index=A.index, columns=B.columns)

    # Convert to ndarray for check
    A_arr = A.values if isinstance(A, pd.DataFrame) else np.asarray(A)
    B_arr = B.values if isinstance(B, pd.DataFrame) else np.asarray(B)

    if A_arr.ndim != 2 or B_arr.ndim != 2:
        raise ValueError(f"matrix_multiply expects 2D arrays, got {A_arr.ndim}D and {B_arr.ndim}D")

    m, k1 = A_arr.shape
    k2, n = B_arr.shape
    if k1 != k2:
        raise ValueError(f"matrix_multiply dimension mismatch: A ({m},{k1}) @ B ({k2},{n}) – inner dimensions {k1} != {k2}")

    result = A_arr @ B_arr
    assert result.shape == (m, n)

    # If one input was DataFrame, return DataFrame with sensible labels? Keep ndarray for math layer simplicity
    if isinstance(A, pd.DataFrame) or isinstance(B, pd.DataFrame):
        # Return ndarray to avoid label confusion unless both DataFrames
        return result
    return result


def quadratic_form(w: Union[np.ndarray, pd.Series], M: Union[np.ndarray, pd.DataFrame], check_symmetric: bool = True, tol: float = 1e-8) -> float:
    """
    Return w^T M w.

    Mathematical: w^T M w = Σ_i Σ_j w_i M_{ij} w_j, portfolio variance for symmetric PSD M.

    Args:
        w: (N,) 1-D ndarray or Series, length N
        M: (N,N) symmetric ndarray or DataFrame
        check_symmetric: if True, assert M is symmetric within tol
        tol: tolerance for symmetry check

    Shape:
        in: w (N,) , M (N,N) -> out: scalar float

    Returns:
        float scalar

    Raises:
        ValueError if dimensions mismatch or M not symmetric (when check_symmetric=True) or not 2D/square
    """
    # Convert w
    if isinstance(w, pd.Series):
        w_arr = w.values
    else:
        w_arr = np.asarray(w)

    if w_arr.ndim != 1:
        raise ValueError(f"quadratic_form expects 1D w, got {w_arr.ndim}D shape {w_arr.shape}")

    N = w_arr.shape[0]

    # Convert M
    if isinstance(M, pd.DataFrame):
        M_arr = M.values
        if M.shape[0] != M.shape[1]:
            raise ValueError(f"quadratic_form expects square M, got {M.shape}")
        if M.shape[0] != N:
            raise ValueError(f"quadratic_form dimension mismatch: w length {N} != M shape {M.shape}")
    else:
        M_arr = np.asarray(M)
        if M_arr.ndim != 2:
            raise ValueError(f"quadratic_form expects 2D M, got {M_arr.ndim}D")
        if M_arr.shape[0] != M_arr.shape[1]:
            raise ValueError(f"quadratic_form expects square M, got {M_arr.shape}")
        if M_arr.shape[0] != N:
            raise ValueError(f"quadratic_form dimension mismatch: w length {N} != M shape {M_arr.shape}")

    if check_symmetric:
        if not is_symmetric(M_arr, tol=tol):
            max_asym = np.max(np.abs(M_arr - M_arr.T))
            raise ValueError(f"quadratic_form expects symmetric M, max asymmetry {max_asym} > tol {tol}")

    # Compute w^T M w = w @ M @ w
    result = float(w_arr @ M_arr @ w_arr)

    # Cross-check via sum_{ij} w_i M_ij w_j
    # result2 = np.sum(np.outer(w_arr, w_arr) * M_arr)
    # assert np.isclose(result, result2)

    return result