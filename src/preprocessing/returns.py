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
from pathlib import Path

from typing import Literal
import numpy as np
import pandas as pd

from src.data.stock_data import TRADING_DAYS_PER_YEAR

RETURN_TYPE = "simple"

__all__ = [
    "compute_expected_returns",
    "compute_volatility",

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


def compute_expected_returns(
    returns: pd.DataFrame,
    annualize: bool = True,
) -> pd.Series:
    """
    Compute expected returns vector mu from daily returns.

    mu_i = (1/T) sum_t r_{i,t}  (sample mean)
    If annualize: mu_annual = mu_daily * 252 (arithmetic, not geometric)

    Args:
        returns: DataFrame of daily returns, shape (T, N), no NaN, no inf.
        annualize: if True multiply by TRADING_DAYS_PER_YEAR.

    Returns:
        pd.Series indexed by ticker – mu vector for R_p = w^T mu.
        Labeled Series prevents column-order drift in dot products.

    Raises:
        ValueError if empty, has NaN, or infinite.
    """
    if returns.empty:
        raise ValueError("compute_expected_returns received empty DataFrame")

    if returns.isna().any().any():
        raise ValueError("compute_expected_returns: returns contain NaN")

    if not np.isfinite(returns.values).all():
        raise ValueError("compute_expected_returns: returns contain infinite values")

    daily_mean = returns.mean()

    if annualize:
        mu = daily_mean * TRADING_DAYS_PER_YEAR
    else:
        mu = daily_mean

    mu.name = "expected_return"
    return mu


def compute_volatility(
    returns: pd.DataFrame,
    annualize: bool = True,
) -> pd.Series:
    """
    Compute volatility sigma per ticker from daily returns.

    sigma_i = sqrt( (1/(T-1)) sum_t (r_{i,t} - mu_i)^2 )  – sample std, ddof=1.
    sigma_annual = sigma_daily * sqrt(252)

    DDOF CONTRACT: This function uses ddof=1 (sample std, pandas default).
    This matches Phase 3 covariance Sigma = X^T X / (T-1) which uses T-1 denominator.
    Using ddof=0 would cause sqrt(diag(Sigma_annual)) to disagree with sigma_annual
    by factor sqrt((T-1)/T) – small but real bug if instructor checks.

    Why sqrt(252) not 252? Variances add linearly under independence:
    Var(annual) ≈ 252 * Var(daily). Std = sqrt(Var), so
    sqrt(252 * Var_daily) = sqrt(252) * sqrt(Var_daily) = sqrt(252) * sigma_daily.
    Covariance annualizes by *252, volatility by *sqrt(252).

    Args:
        returns: DataFrame of daily returns, no NaN, no inf.
        annualize: if True multiply by sqrt(TRADING_DAYS_PER_YEAR).

    Returns:
        pd.Series indexed by ticker – sigma vector, sqrt(diag(Sigma)) should match this.

    Raises:
        ValueError if empty, NaN, infinite.
    """
    if returns.empty:
        raise ValueError("compute_volatility received empty DataFrame")

    if returns.isna().any().any():
        raise ValueError("compute_volatility: returns contain NaN")

    if not np.isfinite(returns.values).all():
        raise ValueError("compute_volatility: returns contain infinite values")

    daily_vol = returns.std()

    if annualize:
        vol = daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)
    else:
        vol = daily_vol

    vol.name = "volatility"
    return vol


def process_returns_pipeline(
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict]:
    """
    One-call pipeline: prices -> returns -> (mu, sigma, summary).

    Guarantees mu and sigma come from same returns matrix used in Phase 3 covariance,
    preventing bug of mu from one window and Sigma from another.

    Args:
        prices: Clean price DataFrame from Phase 1 (DatetimeIndex, >0, no NaN).

    Returns:
        returns_df: daily simple returns, shape (T-1, N)
        mu_series: annualized expected returns, pd.Series indexed by ticker
        sigma_series: annualized volatility, pd.Series indexed by ticker
        summary_dict: dict with n_assets, n_obs, convention, annualization, min/max, best/worst.

    Math: mu and sigma derived from T-1 degrees of freedom (T trading days -> T-1 returns).
    """
    returns_df = compute_daily_returns(prices, method="simple")
    mu_series = compute_expected_returns(returns_df, annualize=True)
    sigma_series = compute_volatility(returns_df, annualize=True)

    n_assets = len(prices.columns)
    n_obs_daily = len(returns_df)

    best_return_ticker = mu_series.idxmax()
    best_return_value = float(mu_series.max())
    lowest_vol_ticker = sigma_series.idxmin()
    lowest_vol_value = float(sigma_series.min())

    summary = {
        "n_assets": n_assets,
        "n_obs_daily": n_obs_daily,
        "n_obs_prices": len(prices),
        "return_convention": RETURN_TYPE,
        "annualization": TRADING_DAYS_PER_YEAR,
        "annualization_vol_factor": f"sqrt({TRADING_DAYS_PER_YEAR})",
        "mu_min": float(mu_series.min()),
        "mu_max": float(mu_series.max()),
        "sigma_min": float(sigma_series.min()),
        "sigma_max": float(sigma_series.max()),
        "best_return_ticker": best_return_ticker,
        "best_return_value": best_return_value,
        "lowest_vol_ticker": lowest_vol_ticker,
        "lowest_vol_value": lowest_vol_value,
        "tickers": list(prices.columns),
        "degrees_of_freedom_note": "mu and sigma derived from T-1 degrees (T prices -> T-1 returns)",
    }

    return returns_df, mu_series, sigma_series, summary


def summary_to_markdown(mu: pd.Series, sigma: pd.Series, summary: dict) -> str:
    lines = []
    lines.append("# Returns Summary Report")
    lines.append("")
    lines.append(f"**Generated:** {pd.Timestamp.now().date()}")
    lines.append(f"**Convention:** {summary['return_convention']} returns, annualized with {summary['annualization']} trading days")
    lines.append(f"**Note:** mu and sigma derived from T-1 degrees of freedom (T prices -> T-1 returns)")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| n_assets | {summary['n_assets']} |")
    lines.append(f"| n_obs_prices | {summary['n_obs_prices']} |")
    lines.append(f"| n_obs_daily_returns | {summary['n_obs_daily']} |")
    lines.append(f"| mu_min (annual) | {summary['mu_min']:.4f} |")
    lines.append(f"| mu_max (annual) | {summary['mu_max']:.4f} |")
    lines.append(f"| sigma_min (annual) | {summary['sigma_min']:.4f} |")
    lines.append(f"| sigma_max (annual) | {summary['sigma_max']:.4f} |")
    lines.append(f"| best_return_ticker | {summary['best_return_ticker']} ({summary['best_return_value']:.4f}) |")
    lines.append(f"| lowest_vol_ticker | {summary['lowest_vol_ticker']} ({summary['lowest_vol_value']:.4f}) |")
    lines.append("")
    lines.append("### Per-Ticker mu_annual and sigma_annual")
    lines.append("")
    lines.append("| Ticker | mu_annual | sigma_annual |")
    lines.append("|--------|-----------|--------------|")
    for ticker in summary["tickers"]:
        mu_val = mu[ticker] if ticker in mu.index else float("nan")
        sigma_val = sigma[ticker] if ticker in sigma.index else float("nan")
        lines.append(f"| {ticker} | {mu_val:.4f} | {sigma_val:.4f} |")
    lines.append("")
    lines.append("**Sanity:** AAPL sigma typically 0.20-0.40, JNJ 0.12-0.22, XOM 0.20-0.35. If 0.01 or 4.0, annualization bug.")
    lines.append("")
    lines.append("**Annualization:** mu_annual = mu_daily * 252, sigma_annual = sigma_daily * sqrt(252)")
    return "\n".join(lines)


if __name__ == "__main__":
    from src.data.stock_data import load_prices, PROCESSED_DATA_DIR

    # Find latest processed CSV
    processed_dir = Path(PROCESSED_DATA_DIR)
    csv_files = list(processed_dir.glob("prices_clean_*.csv"))
    if not csv_files:
        # Fallback: check data/processed directly
        alt_dir = Path("data/processed")
        csv_files = list(alt_dir.glob("prices_clean_*.csv"))
    if not csv_files:
        print("No processed price CSV found. Run: python -m src.data.stock_data first")
        print(f"Searched in {processed_dir} and data/processed/")
        raise SystemExit(1)

    latest = max(csv_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading latest processed prices: {latest}")
    prices = load_prices(latest)
    print(f"Prices shape: {prices.shape}, tickers: {list(prices.columns)}")

    returns_df, mu_series, sigma_series, summary = process_returns_pipeline(prices)

    print(f"\nReturns shape: {returns_df.shape}")
    print("\nMu annual:")
    print(mu_series)
    print("\nSigma annual:")
    print(sigma_series)
    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    md = summary_to_markdown(mu_series, sigma_series, summary)
    print("\n" + md)

    docs_path = Path("docs/return_summary.md")
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(md)
    print(f"\nReport saved to {docs_path}")

    # Sanity warnings
    for ticker, sigma_val in sigma_series.items():
        if sigma_val < 0.05 or sigma_val > 1.0:
            print(f"WARNING: {ticker} sigma_annual {sigma_val:.4f} outside typical 0.05-1.0 range – check annualization")