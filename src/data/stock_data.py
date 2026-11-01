"""
src/data/stock_data.py
Phase 1.1-1.4 – Universe + Download + Cleaning + Persistence

yfinance 1.4.1 with auto_adjust=True returns Close already adjusted.
Adjusted close avoids split/dividend jumps that corrupt covariance.
Covariance requires rectangular complete matrix – cleaning uses full-case deletion to preserve PSD.
CSV round-trip must preserve float precision – no rounding on save.
"""

from typing import List, Optional, Tuple
from pathlib import Path
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

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")


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

    for col in df.columns.tolist():
        if df[col].isna().all():
            dropped.append(col)

    if dropped:
        df = df.drop(columns=dropped)

    if df.empty:
        raise ValueError(f"All tickers dropped as all-NaN. Dropped={dropped}")

    n_rows = len(df)
    for col in df.columns.tolist():
        nan_frac = df[col].isna().sum() / n_rows if n_rows > 0 else 1.0
        if nan_frac > (1 - min_coverage):
            dropped.append(col)

    if dropped:
        df = df.drop(columns=[c for c in dropped if c in df.columns])

    if df.empty:
        raise ValueError(f"No tickers survive coverage threshold {min_coverage}. Dropped={dropped}")

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


def _build_filename(prefix: str, tickers: List[str], start: str, end: Optional[str]) -> str:
    # Filename convention includes tickers and date range to avoid overwrites
    # Example: prices_clean_AAPL_MSFT_JPM_2018-01-01_2024-12-31.csv
    tickers_part = "_".join(tickers[:5])
    if len(tickers) > 5:
        tickers_part += f"_plus{len(tickers)-5}"
    end_part = end if end is not None else "today"
    return f"{prefix}_{tickers_part}_{start}_{end_part}.csv"


def save_prices(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    # Ensure parent directories exist for reproducibility
    path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        raise ValueError(f"Cannot save empty DataFrame to {path}")

    # Preserve exact float precision – default to_csv writes 17 significant digits
    df.to_csv(path, index=True)


def load_prices(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found at {path}")

    df = pd.read_csv(path, index_col=0, parse_dates=True)

    if df.empty:
        raise ValueError(f"Loaded empty DataFrame from {path}")

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    if df.index.duplicated().any():
        raise ValueError(f"Duplicate dates found in {path}")

    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    df.index.name = "Date"

    # Validate numeric columns
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            try:
                df[col] = pd.to_numeric(df[col], errors='raise')
            except Exception as e:
                raise ValueError(f"Column {col} in {path} is not numeric: {e}")

    return df


def fetch_and_cache(
    tickers: List[str],
    start: str,
    end: Optional[str] = None,
    min_coverage: float = 0.95,
    use_cache: bool = True,
    raw_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> pd.DataFrame:
    # Check cache first for offline-friendly fast loads
    processed_filename = _build_filename("prices_clean", tickers, start, end)
    processed_path = Path(processed_dir) / processed_filename

    raw_filename = _build_filename("prices_raw", tickers, start, end)
    raw_path = Path(raw_dir) / raw_filename

    if use_cache and processed_path.exists():
        return load_prices(processed_path)

    # Download, clean, save both raw and processed for reproducibility
    raw_df = download_price_data(tickers, start, end)
    clean_df, dropped = clean_price_data(raw_df, min_coverage)

    # Save raw and cleaned for Phase 2+ offline use
    save_prices(raw_df, raw_path)
    save_prices(clean_df, processed_path)

    return clean_df