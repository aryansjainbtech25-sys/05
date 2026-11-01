"""
src/linear_algebra/covariance.py
Phase 3.1-3.2 – Matrix Utilities and Covariance via X^T X / (T-1)

This module implements covariance using explicit linear algebra identity,
not pandas.DataFrame.cov().

Formulas:
- X̃ = X - 1_T μ^T, μ_i = (1/T) Σ_t r_{t,i}
- X̃^T X̃ is (N×T)·(T×N) = (N×N), entry (i,j) = Σ_t (r_{t,i}-μ_i)(r_{t,j}-μ_j)
- Σ_daily = X̃^T X̃ / (T-1)  (Bessel's correction, unbiased)
- Equivalent: Σ_{ij} = (1/(T-1)) Σ_t (r_{t,i}-μ_i)(r_{t,j}-μ_j)
- Annualization: Σ_annual = 252 * Σ_daily (assumes i.i.d. daily returns, approximation)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.data.stock_data import TRADING_DAYS_PER_YEAR

from .matrix_ops import (
    center_columns,
    transpose,
    matrix_multiply,
    quadratic_form,
    is_symmetric,
)

__all__ = [
    "center_columns",
    "transpose",
    "matrix_multiply",
    "quadratic_form",
    "is_symmetric",
    "compute_covariance_matrix",
    "compute_correlation_matrix",
]


def compute_covariance_matrix(
    returns: pd.DataFrame,
    annualize: bool = False,
) -> pd.DataFrame:
    """
    Compute covariance matrix via X̃^T X̃ / (T-1), where X̃ is column-centered returns.

    Mathematical:
        Define X̃ = X - 1_T μ^T, where μ is (N,) vector of column means, 1_T length-T ones.
        Then Σ_daily = X̃^T X̃ / (T-1)
        Entry (i,j) = Σ_t (r_{t,i}-μ_i)(r_{t,j}-μ_j) / (T-1)
        Annualized: Σ_annual = 252 * Σ_daily

    Args:
        returns: (T,N) DataFrame, T trading days, N tickers, no NaN, numeric
        annualize: if True multiply by TRADING_DAYS_PER_YEAR (252)

    Shape:
        in: (T,N) -> out: (N,N) DataFrame indexed/columned by ticker

    Returns:
        pd.DataFrame (N,N), index=returns.columns, columns=returns.columns

    Raises:
        ValueError if not 2D, has NaN, <2 rows/cols, non-numeric

    Note:
        Does NOT call np.cov, DataFrame.cov, DataFrame.corr – uses explicit identity.
        np.cov may be used in tests as oracle only.
    """
    if returns.empty:
        raise ValueError("compute_covariance_matrix received empty DataFrame")

    if returns.shape[0] < 2:
        raise ValueError(f"compute_covariance_matrix requires at least 2 rows, got {returns.shape[0]}")

    if returns.shape[1] < 1:
        raise ValueError(f"compute_covariance_matrix requires at least 1 column, got {returns.shape[1]}")

    if returns.isna().any().any():
        raise ValueError("compute_covariance_matrix: returns contain NaN")

    # Numeric dtype check
    for col in returns.columns:
        if not pd.api.types.is_numeric_dtype(returns[col]):
            raise ValueError(f"compute_covariance_matrix: column {col} is non-numeric")

    # Convert to numpy X (T,N)
    X = returns.values  # (T,N)
    T, N = X.shape

    # Center columns: X̃ = X - μ, Task 3.1 primitive
    # Use matrix_ops center_columns on ndarray
    Xc = center_columns(X)  # (T,N), each column mean ≈0

    # Verify centered
    assert Xc.shape == (T, N)
    col_means = np.mean(Xc, axis=0)
    if not np.allclose(col_means, 0, atol=1e-12):
        raise RuntimeError(f"Centering failed: column means {col_means}")

    # Compute Σ = X̃^T X̃ / (T-1)
    # X̃^T is (N,T), X̃ is (T,N) -> (N,N)
    XT = transpose(Xc)  # (N,T), uses Task 3.1, asserts shape
    assert XT.shape == (N, T)

    # Use matrix_multiply for dimension compatibility check
    XTX = matrix_multiply(XT, Xc)  # (N,T) @ (T,N) = (N,N)
    assert XTX.shape == (N, N)

    Sigma_daily = XTX / (T - 1)

    if annualize:
        Sigma = Sigma_daily * TRADING_DAYS_PER_YEAR
    else:
        Sigma = Sigma_daily

    # Return as DataFrame with ticker labels preserved
    Sigma_df = pd.DataFrame(
        Sigma,
        index=returns.columns,
        columns=returns.columns,
    )

    # Basic sanity: diagonal positive (variance >0)
    diag = np.diag(Sigma_df.values)
    if not np.all(diag > 0):
        # Could be zero if constant returns, but warn
        non_pos = [c for c, v in zip(Sigma_df.columns, diag) if v <= 0]
        if len(non_pos) > 0:
            # Allow but log – for synthetic constant data might be zero
            # Raise only if negative (impossible for covariance)
            if np.any(diag < -1e-12):
                raise ValueError(f"compute_covariance_matrix: negative variance in {non_pos}")

    return Sigma_df


def compute_correlation_matrix(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute correlation matrix from covariance (helper for later tasks).

    Correlation is scale-invariant, annualization cancels.

    Returns:
        DataFrame (N,N) correlation
    """
    Sigma = compute_covariance_matrix(returns, annualize=False)
    # Use standard formula: corr = D^{-1} Sigma D^{-1}, D = diag(sigma)
    diag = np.sqrt(np.diag(Sigma.values))
    # Avoid division by zero
    diag[diag == 0] = 1e-12
    D_inv = np.diag(1.0 / diag)
    Corr = D_inv @ Sigma.values @ D_inv
    return pd.DataFrame(Corr, index=returns.columns, columns=returns.columns)