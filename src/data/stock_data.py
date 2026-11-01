"""
src/data/stock_data.py
Phase 1.1-1.2 – Data Universe Constants + Download Function

Defines deterministic defaults used by all downstream phases
and provides primary fetch via yfinance.

yfinance 1.4.1 behavior note:
- With auto_adjust=True (recommended since yf 0.2.31+), the API does NOT return a separate "Adj Close" column.
  The "Close" column itself is already adjusted for splits and dividends.
- yf.download returns DataFrame with columns = MultiIndex (Field, Ticker) when >1 ticker,
  or Field Index when 1 ticker, if group_by='ticker' not used.
  Example shape for 2 tickers: columns = [('Close','AAPL'),('Close','MSFT'), ...] or level order depends on auto_adjust.
  We flatten to single level: one column per ticker containing adjusted close.
- With auto_adjust=False, you'd get separate "Adj Close" field, but we use auto_adjust=True to avoid confusion.

Prices: positive real-valued time series, indexed by trading date (discrete ordered set).
Adjusted close is critical: raw close is discontinuous at splits/dividends, injecting artificial -50% returns that corrupt covariance.
"""

from typing import List, Optional
import pandas as pd
import yfinance as yf

# Universe of large-cap liquid US tickers across >=3 sectors, >=10y history
DEFAULT_TICKERS = [
    "AAPL",  # Apple - Technology (Consumer Electronics)
    "MSFT",  # Microsoft - Technology (Software)
    "JPM",   # JPMorgan Chase - Financial Services
    "JNJ",   # Johnson & Johnson - Healthcare (Pharma)
    "PG",    # Procter & Gamble - Consumer Staples
    "XOM",   # Exxon Mobil - Energy
    "AMZN",  # Amazon - Consumer Discretionary
]

# Default time window - deterministic for all phases
DEFAULT_START_DATE = "2018-01-01"
DEFAULT_END_DATE = None  # None resolves to today at call time (yfinance end = now)

# Annualization constant - reused in Phase 2,3,4,5
TRADING_DAYS_PER_YEAR = 252


def download_price_data(
    tickers: List[str],
    start: str,
    end: Optional[str] = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Primary fetch: download adjusted closing prices via yfinance batch API.

    Args:
        tickers: List of ticker symbols, e.g. ["AAPL", "MSFT"]
        start: Start date string "YYYY-MM-DD" (inclusive)
        end: End date string "YYYY-MM-DD" (exclusive, yfinance convention) or None for today
        auto_adjust: If True (default), Close is already split/dividend-adjusted.
                     Recommended for yfinance >=0.2.31. yf 1.4.1 uses this convention.

    Returns:
        pd.DataFrame: 
            - DatetimeIndex sorted ascending, named "Date"
            - One column per ticker (no MultiIndex)
            - Values: float > 0, adjusted close prices
            - Shape: (n_trading_days, n_tickers)

    Raises:
        Let yfinance exceptions surface – caller (Task 1.3/1.4) handles retries/caching.
        ValueError if empty result.

    Notes:
        - Uses yf.download(..., auto_adjust=auto_adjust, progress=False, threads=True)
        - Flattens MultiIndex columns if present: extracts "Close" field
        - Does NOT do cleaning/missing-value handling – Task 1.3 responsibility
    """
    if not tickers:
        raise ValueError("tickers list must be non-empty")

    # yfinance batch download – recommended API
    # progress=False to avoid stdout spam in notebooks/app
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
        threads=True,
        group_by="column",  # ensures columns = Field x Ticker (consistent)
    )

    if raw.empty:
        raise ValueError(f"yfinance returned empty DataFrame for tickers={tickers}, start={start}, end={end}")

    # --- Flatten logic for yfinance 1.4.1 ---
    # Cases:
    # 1) Multiple tickers: raw.columns is MultiIndex [(Field, Ticker), ...] e.g. ('Close','AAPL')
    #    or (Ticker, Field) depending on group_by – we handle both.
    # 2) Single ticker: raw.columns may be single-level Index ['Close','High',...] or MultiIndex with single ticker
    # We want DataFrame with columns = tickers, values = Close (adjusted)

    if isinstance(raw.columns, pd.MultiIndex):
        # Determine level order: check if 'Close' in level 0
        level0 = raw.columns.get_level_values(0)
        if "Close" in level0:
            # columns = (Field, Ticker) -> select Field=Close
            # raw["Close"] returns DataFrame with columns = tickers
            try:
                df = raw["Close"]
            except KeyError:
                # fallback: xs
                df = raw.xs("Close", axis=1, level=0, drop_level=True)
        elif "Close" in raw.columns.get_level_values(1):
            # columns = (Ticker, Field)
            df = raw.xs("Close", axis=1, level=1, drop_level=True)
        else:
            # Unexpected – fallback to Close if exists else first field
            raise ValueError(f"yfinance result missing 'Close' field. Columns: {raw.columns.tolist()[:10]}")
    else:
        # Single ticker, single-level columns
        if "Close" in raw.columns:
            df = raw[["Close"]].copy()
            # Rename column from 'Close' to ticker symbol
            df.columns = [tickers[0]] if len(tickers) == 1 else raw.columns
            # If multi-ticker but not MultiIndex (rare), handle
            if len(tickers) > 1:
                # raw might already be wide? Actually single-level with tickers as columns when auto_adjust?
                # Try to interpret: if raw has ticker names as columns directly
                df = raw.copy()
        else:
            raise ValueError(f"yfinance result missing 'Close' field. Columns: {raw.columns.tolist()}")

    # Ensure DataFrame (when single ticker, df may be Series after selection)
    if isinstance(df, pd.Series):
        df = df.to_frame(name=tickers[0])

    # Ensure columns are exactly tickers (in provided order if possible)
    # yfinance may return columns in requested order, but sort to be safe? Keep provided order.
    # If df columns don't match tickers exactly (e.g., due to missing ticker), keep what we have but ensure order
    # Reindex columns to tickers list, missing will be NaN (let cleaning phase handle)
    existing_cols = list(df.columns)
    # If df columns are superset/subset, align
    if set(tickers).issubset(set(existing_cols)):
        df = df[tickers]
    # else keep as is (e.g., single ticker)

    # Clean index: ensure DatetimeIndex, sorted ascending, named Date
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.index.name = "Date"

    # Basic sanity: ensure float dtype
    df = df.astype(float)

    return df