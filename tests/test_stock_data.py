import pytest
import pandas as pd
from pathlib import Path

from src.data.stock_data import (
    clean_price_data,
    save_prices,
    load_prices,
    validate_price_data,
)


def test_cleaning_failed_ticker_all_nan():
    df = pd.DataFrame(
        {"AAPL": [100.0, 101.0], "BAD": [None, None]},
        index=pd.date_range("2023-01-01", periods=2),
    )
    clean, dropped = clean_price_data(df)
    assert "BAD" in dropped
    assert "AAPL" in clean.columns
    assert "BAD" not in clean.columns


def test_cleaning_interior_gap_forward_filled():
    # 20 rows, 1 NaN = 5% missing -> passes 95% coverage, then ffill fills it
    idx = pd.date_range("2023-01-01", periods=20)
    values = [100.0 + i for i in range(20)]
    values[5] = None
    df = pd.DataFrame({"AAPL": values}, index=idx)
    clean, dropped = clean_price_data(df)
    assert clean.isna().sum().sum() == 0
    assert dropped == []
    assert len(clean) == 20



def test_cleaning_long_gap_over_limit_drops_column():
    # 10 consecutive NaNs > limit 5, coverage 1/12 ~8% <95% -> column dropped
    df = pd.DataFrame(
        {"AAPL": [100.0] + [None] * 10 + [110.0]},
        index=pd.date_range("2023-01-01", periods=12),
    )
    # With single ticker that fails coverage, should raise empty after drop
    with pytest.raises(ValueError, match="No tickers survive|All tickers dropped"):
        clean_price_data(df, min_coverage=0.95)


def test_cleaning_long_gap_policy_documented():
    # Two tickers: one with long gap, one clean -> long gap ticker dropped
    df = pd.DataFrame(
        {
            "AAPL": [100.0] + [None] * 10 + [110.0],
            "MSFT": list(range(100, 112)),
        },
        index=pd.date_range("2023-01-01", periods=12),
    )
    clean, dropped = clean_price_data(df, min_coverage=0.95)
    assert "AAPL" in dropped
    assert "MSFT" in clean.columns
    assert clean.isna().sum().sum() == 0


def test_cleaning_empty_input_raises():
    with pytest.raises(ValueError, match="empty DataFrame"):
        clean_price_data(pd.DataFrame())


def test_roundtrip_save_load(tmp_path):
    df_orig = pd.DataFrame(
        {"AAPL": [100.123456789012345, 101.1]},
        index=pd.date_range("2023-01-01", periods=2),
    )
    df_orig.index.name = "Date"
    path = tmp_path / "test_roundtrip.csv"
    save_prices(df_orig, path)
    assert path.exists()
    df_loaded = load_prices(path)
    assert df_orig.equals(df_loaded)
    pd.testing.assert_frame_equal(df_orig, df_loaded, check_freq=False)
    assert isinstance(df_loaded.index, pd.DatetimeIndex)
    assert df_loaded.index.is_monotonic_increasing


def test_load_invalid_file_raises():
    with pytest.raises(FileNotFoundError):
        load_prices("nonexistent/path.csv")


def test_validate_happy_path():
    df = pd.DataFrame(
        {
            "AAPL": [100.0, 101.0, 102.0],
            "MSFT": [200.0, 201.0, 202.0],
            "JPM": [50.0, 51.0, 52.0],
        },
        index=pd.date_range("2023-01-01", periods=3),
    )
    df.index.name = "Date"
    summary = validate_price_data(df)
    assert summary["n_tickers"] == 3
    assert summary["n_rows"] == 3
    assert summary["all_positive"] is True
    assert summary["has_nan"] is False
    assert summary["duplicate_dates"] is False
    assert "per_ticker_first_date" in summary
    assert "per_ticker_last_date" in summary
    assert "per_ticker_missing_count" in summary
    assert all(v == 0 for v in summary["per_ticker_missing_count"].values())


def test_validate_duplicate_index_detection():
    idx = pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-02"])
    df = pd.DataFrame({"AAPL": [100.0, 100.0, 101.0]}, index=idx)
    df.index.name = "Date"
    summary = validate_price_data(df)
    # validate should flag duplicate_dates True
    assert summary["duplicate_dates"] is True


def test_load_duplicate_dates_raises(tmp_path):
    # Create CSV with duplicate dates manually
    path = tmp_path / "dup.csv"
    path.write_text("Date,AAPL\n2023-01-01,100\n2023-01-01,100\n2023-01-02,101\n")
    with pytest.raises(ValueError, match="Duplicate dates"):
        load_prices(path)