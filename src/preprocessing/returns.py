"""
src/preprocessing/returns.py
Phase 2.1–2.2 – Return Convention, Annualization, and Daily Returns

Convention chosen: SIMPLE returns.

Why simple, not log:
- Markowitz portfolio return R_p = w^T mu is exactly linear under simple returns.
  Simple return r_t = P_t / P_{t-1} - 1 aggregates linearly: R_p,t = sum_i w_i * r_{i,t}
  Log return r_t = ln(P_t / P_{t-1}) composes multiplicatively; portfolio log return is
  only approximately w^T mu.
- Phase 3 teaches covariance as Sigma = X^T X / (T-1) where X is centered simple-return matrix.
  That identity is exactly sample covariance of simple returns, used in w^T Sigma w.
- Linear algebra focus is quadratic forms w^T Sigma w with linear weights.

Formulas:
Simple: r_t = (P_t - P_{t-1}) / P_{t-1} = P_t / P_{t-1} - 1
Log:    r_t = ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})

Annualization identities (verbatim in Phase 3,4,5,6):
mu_annual = mu_daily * 252
Sigma_annual = Sigma_daily * 252
sigma_annual = sigma_daily * sqrt(252)

Constants centralized: TRADING_DAYS_PER_YEAR from src.data.stock_data.
"""

from __future__ import annotations

from typing import Literal
import numpy as np
import pandas as pd

from src.data.stock_data import TRADING_DAYS_PER_YEAR

RETURN_TYPE = "simple"

__all__ = [
    "RETURN_TYPE",
    "TRADING_DAYS_PER_YEAR",
    "compute_daily_returns",
    "compute_returns",
    "compute_log_returns",
]


def compute_daily_returns(
    prices: pd.DataFrame,
    method: str = "simple",
) -> pd.DataFrame:
    """
    Convert price DataFrame to daily returns DataFrame.

    Input: price DataFrame from Phase 1 – DatetimeIndex sorted, one column per ticker, values >0.
    Output: returns DataFrame, same column order, DatetimeIndex, first row dropped.
            Does NOT fillna(0) – that fabricates fake zero return.

    Args:
        prices: DataFrame with DatetimeIndex, columns=tickers.
        method: "simple" (must be exact) or optionally "log".

    Returns:
        DataFrame of daily returns, shape (T-1, N), no NaN.

    Raises:
        ValueError: for non-positive, unsorted index, duplicate dates, non-numeric, empty, NaN.

    Math:
        r_t = P_t / P_{t-1} - 1 realized between day t-1 and t.
        Row t in output corresponds to return between trading day t-1 and t.
        Matrix X in Phase 3 is this DataFrame mean-centered.
    """
    if prices.empty:
        raise ValueError("compute_daily_returns received empty DataFrame")

    if not isinstance(prices.index, pd.DatetimeIndex):
        try:
            prices = prices.copy()
            prices.index = pd.to_datetime(prices.index)
        except Exception as e:
            raise ValueError(f"Index must be convertible to DatetimeIndex: {e}")

    # Duplicate dates check
    if prices.index.duplicated().any():
        raise ValueError("compute_daily_returns: duplicate dates in price index")

    # Sorted check
    if not prices.index.is_monotonic_increasing:
        raise ValueError("compute_daily_returns: price index must be sorted ascending")

    # Non-numeric columns
    for col in prices.columns:
        if not pd.api.types.is_numeric_dtype(prices[col]):
            try:
                pd.to_numeric(prices[col], errors="raise")
            except Exception:
                raise ValueError(f"compute_daily_returns: column {col} is non-numeric")

    # NaN check – prices from Phase 1 should be clean
    if prices.isna().any().any():
        raise ValueError("compute_daily_returns: prices contain NaN – run clean_price_data first")

    # Non-positive check
    if (prices <= 0).any().any():
        bad = [c for c in prices.columns if (prices[c] <= 0).any()]
        raise ValueError(f"compute_daily_returns: non-positive prices in {bad}")

    if method == "simple":
        returns = prices.pct_change()
    elif method == "log":
        returns = np.log(prices / prices.shift(1))
    else:
        raise ValueError(f"Unsupported method {method}, expected 'simple' or 'log'")

    # Drop first row (pct_change produces NaN) – do NOT fillna(0)
    returns = returns.iloc[1:].copy()
    returns.index.name = "Date"

    # Ensure no NaN remains
    if returns.isna().any().any():
        raise ValueError("compute_daily_returns: returns contain NaN after calc – check price gaps")

    return returns


def compute_returns(
    prices: pd.DataFrame,
    return_type: Literal["simple", "log"] = "simple",
) -> pd.DataFrame:
    """Backward-compatible alias – delegates to compute_daily_returns."""
    method = return_type if return_type in ("simple", "log") else "simple"
    return compute_daily_returns(prices, method=method)


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Separate helper for log returns, not wired to main pipeline."""
    return compute_daily_returns(prices, method="log")