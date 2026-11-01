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
    "is_positive_semidefinite",
    "sigma_from_covariance",
    "verify_covariance_properties",

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

def is_positive_semidefinite(M: pd.DataFrame | np.ndarray, tol: float = 1e-10) -> bool:
    """
    Check if matrix is positive semidefinite (PSD).

    Mathematical: M PSD iff all eigenvalues >= 0.
    For covariance: Σ = (1/(T-1)) X̃^T X̃ = (1/(T-1)) ||X̃ w||^2 >=0 is Gram matrix, so PSD by construction.
    Do NOT use determinant as test – det>0 necessary but not sufficient.

    Args:
        M: (N,N) ndarray or DataFrame, symmetric
        tol: allow eigenvalues >= -tol for float round-off

    Returns:
        bool: True if all eigenvalues >= -tol

    Raises:
        ValueError if not square
    """
    arr = M.values if isinstance(M, pd.DataFrame) else __import__("numpy").asarray(M)
    import numpy as np
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"is_positive_semidefinite expects square matrix, got {arr.shape}")
    # Use eigvalsh for Hermitian (symmetric) – more stable than eig
    eigs = np.linalg.eigvalsh(arr)
    return np.all(eigs >= -tol)


def sigma_from_covariance(Sigma_annual: pd.DataFrame) -> pd.Series:
    """
    Return σ = sqrt(diag(Σ_annual)).

    Mathematical: Σ_ii = Var(r_i), sqrt = σ_i
    Annualized: sqrt(252 * Σ_ii_daily) = sqrt(252)*σ_daily = σ_annual (Phase 2 contract)

    Args:
        Sigma_annual: (N,N) DataFrame, annualized covariance

    Returns:
        pd.Series indexed by ticker
    """
    import numpy as np
    import pandas as pd
    if Sigma_annual.empty:
        raise ValueError("sigma_from_covariance received empty DataFrame")
    diag = np.diag(Sigma_annual.values if isinstance(Sigma_annual, pd.DataFrame) else Sigma_annual)
    if np.any(diag < -1e-12):
        raise ValueError(f"sigma_from_covariance: negative variance {diag}")
    # Clip tiny negative due to float
    diag = np.maximum(diag, 0.0)
    sigma = np.sqrt(diag)
    if isinstance(Sigma_annual, pd.DataFrame):
        return pd.Series(sigma, index=Sigma_annual.index, name="sigma_from_cov")
    return pd.Series(sigma, name="sigma_from_cov")


def verify_covariance_properties(
    Sigma_daily: pd.DataFrame,
    Sigma_annual: pd.DataFrame,
    sigma_series: pd.Series,
    tol_sym: float = 1e-10,
    tol_psd: float = 1e-10,
    tol_sigma: float = 1e-10,
) -> dict:
    """
    Verify structural properties: symmetry, PSD, diag matches Phase 2 σ.

    Hard requirements: symmetric True, psd True, diag_matches_sigma True.
    Raises AssertionError if any fails.

    Args:
        Sigma_daily: (N,N) daily covariance
        Sigma_annual: (N,N) annualized = daily*252
        sigma_series: annualized σ from Phase 2 compute_volatility
        tol_sym: symmetry tolerance
        tol_psd: PSD tolerance
        tol_sigma: sigma match tolerance

    Returns:
        dict {symmetric, psd, diag_matches_sigma, min_eig, max_eig, condition_number, max_diag_sigma_error}

    Raises:
        AssertionError if hard requirement fails
    """
    import numpy as np

    # Symmetry checks
    def _is_symmetric(M, tol):
        arr = M.values if isinstance(M, pd.DataFrame) else np.asarray(M)
        return np.max(np.abs(arr - arr.T)) <= tol

    symmetric_daily = _is_symmetric(Sigma_daily, tol_sym)
    symmetric_annual = _is_symmetric(Sigma_annual, tol_sym)
    symmetric = symmetric_daily and symmetric_annual

    # PSD checks via eigvalsh
    def _eigs(M):
        arr = M.values if isinstance(M, pd.DataFrame) else np.asarray(M)
        return np.linalg.eigvalsh(arr)

    eigs_daily = _eigs(Sigma_daily)
    min_eig = float(np.min(eigs_daily))
    max_eig = float(np.max(eigs_daily))
    psd_daily = np.all(eigs_daily >= -tol_psd)

    eigs_annual = _eigs(Sigma_annual)
    psd_annual = np.all(eigs_annual >= -tol_psd)
    psd = psd_daily and psd_annual

    # Condition number
    try:
        cond = float(np.linalg.cond(Sigma_daily.values if isinstance(Sigma_daily, pd.DataFrame) else Sigma_daily))
    except Exception:
        cond = float("inf")

    # diag matches sigma: sqrt(diag(Sigma_annual)) == sigma_series
    sigma_from_cov = sigma_from_covariance(Sigma_annual)
    # Align indices
    if isinstance(sigma_series, pd.Series):
        # Ensure same order
        sigma_aligned = sigma_series.reindex(sigma_from_cov.index)
    else:
        sigma_aligned = pd.Series(sigma_series, index=sigma_from_cov.index)

    diff = np.abs(sigma_from_cov.values - sigma_aligned.values)
    max_err = float(np.max(diff))
    diag_matches_sigma = max_err <= tol_sigma

    result = {
        "symmetric": bool(symmetric),
        "psd": bool(psd),
        "diag_matches_sigma": bool(diag_matches_sigma),
        "min_eig": min_eig,
        "max_eig": max_eig,
        "min_eig_annual": float(np.min(eigs_annual)),
        "max_eig_annual": float(np.max(eigs_annual)),
        "condition_number": cond,
        "max_diag_sigma_error": max_err,
        "symmetric_daily": bool(symmetric_daily),
        "symmetric_annual": bool(symmetric_annual),
        "psd_daily": bool(psd_daily),
        "psd_annual": bool(psd_annual),
    }

    # Hard requirements – raise AssertionError if fails
    if not symmetric:
        raise AssertionError(f"verify_covariance_properties: symmetry failed daily={symmetric_daily} annual={symmetric_annual}")
    if not psd:
        raise AssertionError(f"verify_covariance_properties: PSD failed min_eig={min_eig} (tol {tol_psd})")
    if not diag_matches_sigma:
        raise AssertionError(f"verify_covariance_properties: diag vs sigma mismatch max_err={max_err} > tol {tol_sigma}")

    # Soft warning for condition number
    if cond > 1e6:
        # Do not fail, just note – suspect multicollinearity
        result["warning"] = f"High condition number {cond} >1e6, suspect multicollinearity"

    return result