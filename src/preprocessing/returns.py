from __future__ import annotations
"""
src/preprocessing/returns.py
Phase 2.1–2.2 – Return Convention, Annualization, and Return Calculation

Convention chosen: SIMPLE returns.

Why simple, not log:
- Markowitz portfolio return R_p = w^T mu is exactly linear under simple returns.
  Simple return r_t = P_t / P_{t-1} - 1 aggregates linearly: R_p,t = sum_i w_i * r_{i,t}
  Log return r_t = ln(P_t / P_{t-1}) composes multiplicatively; portfolio log return is
  only approximately w^T mu, requiring approximation error.
- Phase 3 teaches covariance as Sigma = X^T X / (T-1) where X is centered simple-return matrix.
  That identity is exactly the sample covariance of simple returns, directly used in w^T Sigma w.
- Linear algebra focus of project is quadratic forms w^T Sigma w with linear weights w.
  Linearity of simple returns keeps risk decomposition exact.
- Log returns are useful for time-aggregation and risk, but brief does not require them.
  If needed, provide separate helper `compute_log_returns` with own tests, but pipeline uses simple.

Formulas:
Simple: r_t = (P_t - P_{t-1}) / P_{t-1} = P_t / P_{t-1} - 1
Log:    r_t = ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})  (optional helper)

Annualization identities (must be used verbatim in Phase 3,4,5,6):
mu_annual = mu_daily * 252
Sigma_annual = Sigma_daily * 252
sigma_annual = sigma_daily * sqrt(252)

Constants are centralized: TRADING_DAYS_PER_YEAR imported from src.data.stock_data,
not redefined. RETURN_TYPE is the single source of truth for downstream phases.
"""

from typing import Literal
import numpy as np
import pandas as pd

from src.data.stock_data import TRADING_DAYS_PER_YEAR

RETURN_TYPE = "simple"

__all__ = ["RETURN_TYPE", "TRADING_DAYS_PER_YEAR", "compute_returns", "compute_log_returns"]


def compute_returns(
    prices: pd.DataFrame,
    return_type: Literal["simple", "log"] = "simple",
) -> pd.DataFrame:
    """
    Compute daily returns from price DataFrame.

    Args:
        prices: DataFrame with DatetimeIndex, columns=tickers, values>0, no NaN (cleaned).
        return_type: "simple" (default) or "log".

    Returns:
        DataFrame of same columns, DatetimeIndex, first row NaN dropped.
        Simple: (P_t - P_{t-1}) / P_{t-1}
        Log: ln(P_t / P_{t-1})

    Raises:
        ValueError: if prices empty, has NaN, or non-positive.
    """
    if prices.empty:
        raise ValueError("compute_returns received empty DataFrame")

    if not isinstance(prices.index, pd.DatetimeIndex):
        prices = prices.copy()
        prices.index = pd.to_datetime(prices.index)

    if prices.isna().any().any():
        raise ValueError("compute_returns: prices contains NaN – run clean_price_data first")

    if (prices <= 0).any().any():
        bad = [c for c in prices.columns if (prices[c] <= 0).any()]
        raise ValueError(f"compute_returns: non-positive prices in {bad}")

    if return_type == "simple":
        returns = prices.pct_change()
    elif return_type == "log":
        returns = np.log(prices / prices.shift(1))
    else:
        raise ValueError(f"Unsupported return_type {return_type}, expected simple or log")

    returns = returns.iloc[1:].copy()
    returns.index.name = "Date"

    if returns.isna().any().any():
        raise ValueError("compute_returns: returns contain NaN after calculation – check price gaps")

    return returns


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Separate helper for log returns, not wired to main pipeline."""
    return compute_returns(prices, return_type="log")