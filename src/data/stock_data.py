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


def validate_price_data(df: pd.DataFrame) -> dict:
    if df.empty:
        raise ValueError("validate_price_data received empty DataFrame")

    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    n_tickers = df.shape[1]
    n_rows = df.shape[0]
    start_date = df.index.min()
    end_date = df.index.max()

    # Trading days per year sanity check
    years = (end_date - start_date).days / 365.25
    years = years if years > 0 else 1
    n_trading_days_per_year_avg = n_rows / years

    all_positive = bool((df > 0).all().all())
    has_nan = bool(df.isna().any().any())
    duplicate_dates = bool(df.index.duplicated().any())

    # Count gaps >5 calendar days between adjacent trading dates
    gaps = (df.index.to_series().diff().dropna().dt.days > 5).sum()

    per_ticker_first = {}
    per_ticker_last = {}
    per_ticker_missing = {}

    for col in df.columns:
        s = df[col].dropna()
        per_ticker_first[col] = s.index.min() if not s.empty else None
        per_ticker_last[col] = s.index.max() if not s.empty else None
        per_ticker_missing[col] = int(df[col].isna().sum())

    return {
        "n_tickers": n_tickers,
        "n_rows": n_rows,
        "start_date": str(start_date.date()) if hasattr(start_date, 'date') else str(start_date),
        "end_date": str(end_date.date()) if hasattr(end_date, 'date') else str(end_date),
        "n_trading_days_per_year_avg": round(n_trading_days_per_year_avg, 1),
        "all_positive": all_positive,
        "has_nan": has_nan,
        "duplicate_dates": duplicate_dates,
        "date_gaps_over_5_days": int(gaps),
        "per_ticker_first_date": {k: str(v.date()) if hasattr(v, 'date') else str(v) for k, v in per_ticker_first.items()},
        "per_ticker_last_date": {k: str(v.date()) if hasattr(v, 'date') else str(v) for k, v in per_ticker_last.items()},
        "per_ticker_missing_count": per_ticker_missing,
    }


def summary_to_markdown(summary: dict) -> str:
    lines = []
    lines.append("# Data Summary Report")
    lines.append("")
    lines.append(f"**Generated:** {pd.Timestamp.now().date()}")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| n_tickers | {summary['n_tickers']} |")
    lines.append(f"| n_rows | {summary['n_rows']} |")
    lines.append(f"| start_date | {summary['start_date']} |")
    lines.append(f"| end_date | {summary['end_date']} |")
    lines.append(f"| trading_days/year avg | {summary['n_trading_days_per_year_avg']} |")
    lines.append(f"| all_positive | {summary['all_positive']} |")
    lines.append(f"| has_nan | {summary['has_nan']} |")
    lines.append(f"| duplicate_dates | {summary['duplicate_dates']} |")
    lines.append(f"| date_gaps_over_5_days | {summary['date_gaps_over_5_days']} |")
    lines.append("")
    lines.append("### Per-Ticker Coverage")
    lines.append("")
    lines.append("| Ticker | First | Last | Missing |")
    lines.append("|--------|-------|------|---------|")
    for t in summary["per_ticker_first_date"]:
        first = summary["per_ticker_first_date"][t]
        last = summary["per_ticker_last_date"][t]
        miss = summary["per_ticker_missing_count"][t]
        lines.append(f"| {t} | {first} | {last} | {miss} |")
    lines.append("")
    lines.append("**Sanity:** trading_days/year ~252 indicates correct annualization. has_nan=False, all_positive=True required for downstream math.")
    return "\n".join(lines)


if __name__ == "__main__":
    # One-command refresh for both team members
    tickers = DEFAULT_TICKERS
    start = DEFAULT_START_DATE
    end = DEFAULT_END_DATE

    if start and end and start > end:
        raise ValueError(f"Invalid date range: start {start} > end {end}")

    print(f"Downloading {tickers} from {start} to {end or 'today'}...")
    raw = download_price_data(tickers, start, end)
    print(f"Raw shape: {raw.shape}")

    clean, dropped = clean_price_data(raw)
    print(f"Clean shape: {clean.shape}, Dropped: {dropped}")

    raw_file = RAW_DATA_DIR / _build_filename("prices_raw", tickers, start, end)
    clean_file = PROCESSED_DATA_DIR / _build_filename("prices_clean", tickers, start, end)

    save_prices(raw, raw_file)
    save_prices(clean, clean_file)
    print(f"Saved raw to {raw_file}")
    print(f"Saved clean to {clean_file}")

    summary = validate_price_data(clean)
    print("\nValidation Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    md = summary_to_markdown(summary)
    print("\n" + md)

    docs_path = Path("docs/data_summary.md")
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(md)
    print(f"\nMarkdown report saved to {docs_path}")