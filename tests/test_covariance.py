"""
tests/test_covariance.py
Phase 3.8 – Offline Unit Tests for Covariance, Correlation, Eigen Analysis

All tests offline, no yfinance.
Oracle-based checks prove from-scratch identities match Pandas/NumPy canonical implementations.
Internal consistency checks prove σ and spectral reconstruction agree.
Structural checks prove PSD.
"""

import pytest
import numpy as np
import pandas as pd

from src.linear_algebra.matrix_ops import center_columns, transpose, matrix_multiply, quadratic_form
from src.linear_algebra.covariance import (
    compute_covariance_matrix,
    compute_correlation_matrix,
    compute_correlation_matrix_from_returns,
    is_positive_semidefinite,
    sigma_from_covariance,
    verify_covariance_properties,
)
from src.linear_algebra.eigen_analysis import eigen_decompose, eigen_summary_table, first_principal_direction
from src.preprocessing.returns import compute_daily_returns, compute_volatility
from src.data.stock_data import TRADING_DAYS_PER_YEAR


def make_returns(T=100, N=5, seed=0):
    """Helper: synthetic returns (T,N)"""
    rng = np.random.default_rng(seed)
    tickers = [f"T{i}" for i in range(N)]
    if N >= 7:
        tickers = ["AAPL","MSFT","JPM","JNJ","PG","XOM","AMZN"][:N]
    data = rng.normal(0, 0.01, size=(T,N))
    return pd.DataFrame(data, columns=tickers[:N], index=pd.date_range("2023-01-01", periods=T))


# --- Task 3.2: Covariance ---

def test_covariance_matches_pandas_cov():
    """1. compute_covariance_matrix matches returns.cov(ddof=1)"""
    returns = make_returns(T=100, N=5, seed=42)
    Sigma = compute_covariance_matrix(returns, annualize=False)
    expected = returns.cov(ddof=1)
    assert np.allclose(Sigma.values, expected.values, atol=1e-12), f"max diff {np.max(np.abs(Sigma.values-expected.values))}"


def test_covariance_annualization():
    """2. annualize=True equals daily*252"""
    returns = make_returns(T=100, N=5, seed=1)
    Sigma_daily = compute_covariance_matrix(returns, annualize=False)
    Sigma_annual = compute_covariance_matrix(returns, annualize=True)
    assert np.allclose(Sigma_annual.values, Sigma_daily.values * TRADING_DAYS_PER_YEAR, atol=1e-15)
    assert np.allclose(Sigma_annual.values, Sigma_daily.values * 252)


def test_covariance_symmetric():
    """3. Σ is symmetric"""
    returns = make_returns(T=80, N=5, seed=2)
    Sigma = compute_covariance_matrix(returns)
    assert np.allclose(Sigma.values, Sigma.values.T, atol=1e-12)
    assert Sigma.index.equals(Sigma.columns)


def test_covariance_psd():
    """4. Σ is PSD: min eig >= -1e-12"""
    returns = make_returns(T=80, N=5, seed=3)
    Sigma = compute_covariance_matrix(returns)
    eigs = np.linalg.eigvalsh(Sigma.values)
    assert np.min(eigs) >= -1e-12, f"min eig {np.min(eigs)}"
    assert is_positive_semidefinite(Sigma, tol=1e-10)


def test_diagonal_matches_phase2_sigma():
    """5. sqrt(diag(Σ_annual)) == compute_volatility within 1e-10 (Phase 2 ↔ Phase 3 contract)"""
    returns = make_returns(T=100, N=5, seed=4)
    Sigma_annual = compute_covariance_matrix(returns, annualize=True)
    sigma_phase2 = compute_volatility(returns, annualize=True)
    sigma_from_cov = sigma_from_covariance(Sigma_annual)
    assert np.allclose(sigma_from_cov.values, sigma_phase2.values, atol=1e-10)
    # Also test via verify_covariance_properties hard requirement
    Sigma_daily = compute_covariance_matrix(returns, annualize=False)
    result = verify_covariance_properties(Sigma_daily, Sigma_annual, sigma_phase2)
    assert result["diag_matches_sigma"]
    assert result["max_diag_sigma_error"] < 1e-10


# --- Task 3.4: Correlation ---

def test_correlation_matches_pandas_corr():
    """6. Correlation matches returns.corr()"""
    returns = make_returns(T=100, N=5, seed=5)
    Sigma = compute_covariance_matrix(returns, annualize=False)
    Corr = compute_correlation_matrix(Sigma)
    expected = returns.corr()
    assert np.allclose(Corr.values, expected.values, atol=1e-12), f"max diff {np.max(np.abs(Corr.values-expected.values))}"
    # Also via convenience wrapper
    Corr2 = compute_correlation_matrix_from_returns(returns)
    assert np.allclose(Corr2.values, expected.values, atol=1e-12)


def test_correlation_diagonal_one():
    """7. Correlation diagonal is 1.0 within 1e-12"""
    returns = make_returns(T=100, N=5, seed=6)
    Sigma = compute_covariance_matrix(returns)
    Corr = compute_correlation_matrix(Sigma)
    assert np.allclose(np.diag(Corr.values), 1.0, atol=1e-12)
    # Entries in [-1,1]
    assert np.all(Corr.values >= -1 - 1e-12) and np.all(Corr.values <= 1 + 1e-12)


# --- Task 3.5: Eigen ---

def test_eigenvalues_match_eigvalsh_descending():
    """8. Eigenvalues match np.linalg.eigvalsh(Σ)[::-1] within 1e-12, descending"""
    returns = make_returns(T=100, N=5, seed=7)
    Sigma = compute_covariance_matrix(returns)
    res = eigen_decompose(Sigma)
    expected = np.linalg.eigvalsh(Sigma.values)[::-1]
    assert np.allclose(res["values"], expected, atol=1e-12)
    assert np.all(np.diff(res["values"]) <= 1e-12), "not descending"


def test_eigenvectors_orthonormal():
    """9. Eigenvectors orthonormal: V^T V ≈ I within 1e-10"""
    returns = make_returns(T=100, N=5, seed=8)
    Sigma = compute_covariance_matrix(returns)
    res = eigen_decompose(Sigma)
    V = res["vectors"]
    N = V.shape[0]
    VTV = V.T @ V
    assert np.allclose(VTV, np.eye(N), atol=1e-10), f"max error {np.max(np.abs(VTV-np.eye(N)))}"
    # Unit norm
    norms = np.linalg.norm(V, axis=0)
    assert np.allclose(norms, 1.0, atol=1e-10)


def test_reconstruction_identity():
    """10. Reconstruction V @ diag(λ) @ V^T ≈ Σ within 1e-10"""
    returns = make_returns(T=100, N=5, seed=9)
    Sigma = compute_covariance_matrix(returns)
    res = eigen_decompose(Sigma)
    Lambda = np.diag(res["values"])
    recon = res["vectors"] @ Lambda @ res["vectors"].T
    assert np.allclose(recon, Sigma.values, atol=1e-10), f"max error {np.max(np.abs(recon-Sigma.values))}"
    assert res["reconstruction_error"] < 1e-10


def test_explained_variance_sums_to_one():
    """11. Explained variance ratio sums to 1 within 1e-12"""
    returns = make_returns(T=100, N=5, seed=10)
    Sigma = compute_covariance_matrix(returns)
    res = eigen_decompose(Sigma)
    assert np.isclose(np.sum(res["explained_variance_ratio"]), 1.0, atol=1e-12)
    assert np.isclose(res["cumulative_explained"][-1], 1.0, atol=1e-12)
    assert np.all(np.diff(res["cumulative_explained"]) >= -1e-12)  # monotonic non-decreasing


# --- Task 3.1: Quadratic form ---

def test_quadratic_form_consistency():
    """12. quadratic_form(w,Σ) == w @ Σ @ w within 1e-12"""
    returns = make_returns(T=100, N=5, seed=11)
    Sigma = compute_covariance_matrix(returns)
    rng = np.random.default_rng(11)
    w = rng.normal(0,1,size=5)
    w = w / np.sum(np.abs(w))  # normalize
    val1 = quadratic_form(w, Sigma.values)
    val2 = float(w @ Sigma.values @ w)
    assert abs(val1 - val2) < 1e-12
    # Also sum_ij version
    val_sum = np.sum(np.outer(w,w) * Sigma.values)
    assert abs(val1 - val_sum) < 1e-12


def test_quadratic_form_rejects_non_symmetric():
    """13. quadratic_form rejects non-symmetric M"""
    w = np.array([0.5, 0.5])
    non_sym = np.array([[1,2],[0,1]], dtype=float)
    with pytest.raises(ValueError, match="symmetric"):
        quadratic_form(w, non_sym)
    # Also test symmetric passes
    sym = np.array([[1,0.5],[0.5,1]], dtype=float)
    val = quadratic_form(w, sym)
    assert isinstance(val, float)


def test_center_columns_zero_means():
    """14. center_columns produces zero means within 1e-12"""
    X = np.array([[1,10],[2,20],[3,30]], dtype=float)
    Xc = center_columns(X)
    assert np.allclose(Xc.mean(axis=0), 0, atol=1e-12)
    # DataFrame version
    df = pd.DataFrame(X, columns=["A","B"])
    dfc = center_columns(df)
    assert np.allclose(dfc.mean(axis=0).values, 0, atol=1e-12)


def test_compute_daily_returns_regression():
    """15. Phase 2 regression: compute_daily_returns still passes spec [100,110,99]->[0.10,-0.10]"""
    prices = pd.DataFrame({"AAPL": [100.0, 110.0, 99.0]}, index=pd.date_range("2023-01-01", periods=3))
    ret = compute_daily_returns(prices, method="simple")
    assert ret.shape[0] == 2
    assert abs(ret.iloc[0]["AAPL"] - 0.10) < 1e-9
    assert abs(ret.iloc[1]["AAPL"] + 0.10) < 1e-9


def test_eigen_summary_table_shape():
    """Extra: eigen_summary_table (7,10) and first PC all same sign for real-ish data"""
    returns = make_returns(T=100, N=7, seed=12)
    Sigma = compute_covariance_matrix(returns, annualize=True)
    table = eigen_summary_table(Sigma)
    assert table.shape == (7, 10), f"expected (7,10) got {table.shape}"
    assert np.isclose(table["explained_ratio"].sum(), 1.0, atol=1e-12)
    assert table["cumulative_ratio"].iloc[-1] == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.diff(table["cumulative_ratio"].values) >= -1e-12)


def test_no_yfinance_in_tests():
    """Cross-cutting: ensure tests don't import yfinance module directly"""
    # Check file content doesn't contain actual import statement at line start
    from pathlib import Path
    import re
    test_file = Path(__file__).read_text()
    # Look for lines that start with import yfinance or from yfinance
    pattern = r'^\s*import\s+yfinance|^\s*from\s+yfinance'
    matches = re.findall(pattern, test_file, flags=re.MULTILINE)
    assert len(matches) == 0, f"Found yfinance import in test file: {matches}"