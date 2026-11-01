
"""
src/data/stock_data.py
Phase 1.1-1.3 – Data Universe + Download + Cleaning

yfinance 1.4.1 with auto_adjust=True returns Close already adjusted.
Raw download returns MultiIndex (Field, Ticker) – flattened to ticker columns.
Adjusted close avoids artificial split/dividend jumps that corrupt covariance.
"""

from typing import List, Optional, Tuple
import pandas as pd
import yfinance as yf

DEFAULT_TICKERS = [
    "AAPL",
    "MSFT",
    "JPM",
    "JNJ",
    "PG",
    "XOM",
    "AMZN",
]

DEFAULT_START_DATE = "2018-01-01"
DEFAULT_END_DATE = None
TRADING_DAYS_PER_YEAR = 252


def download_price_data(
    tickers: List[str],
    start: str,
    end: Optional[str] = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    if not tickers:
        raise ValueError("tickers list must be non-empty")

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
        threads=True,
        group_by="column",
    )

    if raw.empty:
        raise ValueError(f"yfinance returned empty DataFrame for tickers={tickers}")

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        if "Close" in level0:
            df = raw["Close"]
        elif "Close" in raw.columns.get_level_values(1):
            df = raw.xs("Close", axis=1, level=1, drop_level=True)
        else:
            raise ValueError(f"Missing Close field. Columns: {raw.columns.tolist()[:10]}")
    else:
        if "Close" in raw.columns:
            df = raw[["Close"]].copy()
            df.columns = [tickers[0]]
        else:
            df = raw.copy()

    if isinstance(df, pd.Series):
        df = df.to_frame(name=tickers[0])

    existing_cols = list(df.columns)
    if set(tickers).issubset(set(existing_cols)):
        df = df[tickers]

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.index.name = "Date"
    df = df.astype(float)

    return df


def clean_price_data(
    raw: pd.DataFrame,
    min_coverage: float = 0.95,
) -> Tuple[pd.DataFrame, List[str]]:
    if raw.empty:
        raise ValueError("clean_price_data received empty DataFrame")

    if not isinstance(raw.index, pd.DatetimeIndex):
        raw = raw.copy()
        raw.index = pd.to_datetime(raw.index)

    dropped: List[str] = []
    df = raw.copy()

    # Entirely failed tickers appear as all-NaN columns
    for col in df.columns.tolist():
        if df[col].isna().all():
            dropped.append(col)

    if dropped:
        df = df.drop(columns=dropped)

    if df.empty:
        raise ValueError(f"All tickers dropped as all-NaN. Dropped={dropped}")

    # Drop tickers with low coverage: > (1-min_coverage) NaNs
    n_rows = len(df)
    for col in df.columns.tolist():
        nan_frac = df[col].isna().sum() / n_rows if n_rows > 0 else 1.0
        if nan_frac > (1 - min_coverage):
            dropped.append(col)

    if dropped:
        df = df.drop(columns=[c for c in dropped if c in df.columns])

    if df.empty:
        raise ValueError(f"No tickers survive coverage threshold {min_coverage}. Dropped={dropped}")

    # Sanity: drop non-positive or non-finite price columns
    for col in df.columns.tolist():
        vals = df[col].dropna()
        if not vals.empty:
            if (vals <= 0).any():
                dropped.append(col)
            elif (vals.apply(lambda x: x == float('inf') or x == float('-inf')).any()):
                dropped.append(col)

    if dropped:
        df = df.drop(columns=[c for c in dropped if c in df.columns], errors='ignore')

    if df.empty:
        raise ValueError(f"All tickers dropped after sanity. Dropped={dropped}")

    # Forward-fill short gaps (exchange halts) then full-case deletion to preserve PSD
    df = df.sort_index()
    df = df.ffill(limit=5)
    df = df.dropna(how='any')

    if df.empty:
        raise ValueError(f"No rows survive after ffill+dropna. Dropped={dropped}")

    if not (df > 0).all().all():
        bad = [c for c in df.columns if (df[c] <= 0).any()]
        raise ValueError(f"Non-positive prices remain in {bad}")

    df.index.name = "Date"
    df = df.sort_index()

    seen = set()
    dropped_unique = []
    for t in dropped:
        if t not in seen:
            seen.add(t)
            dropped_unique.append(t)

    return df, dropped_unique