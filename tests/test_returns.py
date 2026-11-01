import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

from src.preprocessing.returns import (
    RETURN_TYPE,
    TRADING_DAYS_PER_YEAR,
    compute_daily_returns,
    compute_returns,
    compute_expected_returns,
    compute_volatility,
    process_returns_pipeline,
)


def test_return_type_and_trading_days():
    assert RETURN_TYPE in {"simple", "log"}
    assert TRADING_DAYS_PER_YEAR == 252


def test_compute_daily_returns_simple_known_values():
    # Spec: [100,110,99] -> [0.10, -0.10]
    prices = pd.DataFrame({"AAPL": [100.0, 110.0, 99.0]}, index=pd.date_range("2023-01-01", periods=3))
    ret = compute_daily_returns(prices, method="simple")
    assert ret.shape[0] == 2
    assert ret.columns.equals(prices.columns)
    assert ret.isna().sum().sum() == 0
    assert abs(ret.iloc[0]["AAPL"] - 0.10) < 1e-9
    assert abs(ret.iloc[1]["AAPL"] - (-0.10)) < 1e-9


def test_compute_daily_returns_no_fillna_zero():
    # Ensure first row dropped, not filled with 0
    prices = pd.DataFrame({"AAPL": [100.0, 101.0]}, index=pd.date_range("2023-01-01", periods=2))
    ret = compute_daily_returns(prices)
    assert ret.shape[0] == 1
    assert ret.iloc[0]["AAPL"] != 0.0 or abs(ret.iloc[0]["AAPL"] - 0.01) < 1e-9


def test_compute_daily_returns_validations():
    # Unsorted index
    bad = pd.DataFrame({"AAPL": [100.0, 101.0]}, index=pd.to_datetime(["2023-01-02", "2023-01-01"]))
    with pytest.raises(ValueError, match="sorted ascending"):
        compute_daily_returns(bad)

    # Duplicate dates
    dup = pd.DataFrame({"AAPL": [100.0, 100.0]}, index=pd.to_datetime(["2023-01-01", "2023-01-01"]))
    with pytest.raises(ValueError, match="duplicate dates"):
        compute_daily_returns(dup)

    # Non-positive
    neg = pd.DataFrame({"AAPL": [100.0, 0.0]}, index=pd.date_range("2023-01-01", periods=2))
    with pytest.raises(ValueError, match="non-positive"):
        compute_daily_returns(neg)

    # Non-numeric
    non_num = pd.DataFrame({"AAPL": ["a", "b"]}, index=pd.date_range("2023-01-01", periods=2))
    with pytest.raises(ValueError, match="non-numeric"):
        compute_daily_returns(non_num)

    # Empty
    with pytest.raises(ValueError, match="empty DataFrame"):
        compute_daily_returns(pd.DataFrame())


def test_compute_daily_returns_log_variant():
    prices = pd.DataFrame({"AAPL": [100.0, 110.0]}, index=pd.date_range("2023-01-01", periods=2))
    ret_log = compute_daily_returns(prices, method="log")
    assert ret_log.shape[0] == 1
    expected = np.log(110.0 / 100.0)
    assert abs(ret_log.iloc[0]["AAPL"] - expected) < 1e-9


def test_compute_expected_returns_spec_case():
    # Spec: [0.01, -0.02, 0.03] -> mean 0.0066667, annualized 1.68
    returns = pd.DataFrame({"AAPL": [0.01, -0.02, 0.03]})
    mu = compute_expected_returns(returns, annualize=True)
    assert isinstance(mu, pd.Series)
    assert mu.index.equals(returns.columns)
    daily_mean = (0.01 - 0.02 + 0.03) / 3
    assert abs(daily_mean - 0.006666666666666667) < 1e-9
    assert abs(mu["AAPL"] - daily_mean * 252) < 1e-9
    assert abs(mu["AAPL"] - 1.68) < 0.01

    # Cross-check vs pandas
    assert abs(mu["AAPL"] - returns.mean()["AAPL"] * 252) < 1e-12


def test_compute_volatility_ddof1_and_annualization():
    returns = pd.DataFrame({"AAPL": [0.01, -0.02, 0.03]})
    vol = compute_volatility(returns, annualize=True)
    assert isinstance(vol, pd.Series)
    assert vol.index.equals(returns.columns)

    manual_std = np.std([0.01, -0.02, 0.03], ddof=1)
    manual_annual = manual_std * np.sqrt(252)
    assert abs(vol["AAPL"] - manual_annual) < 1e-12

    # Cross-check vs returns.std(ddof=1)*sqrt(252)
    assert abs(vol["AAPL"] - returns.std(ddof=1)["AAPL"] * np.sqrt(252)) < 1e-12

    # Consistency sqrt(var*252)
    var_check = np.var([0.01, -0.02, 0.03], ddof=1) * 252
    assert np.isclose(vol["AAPL"], np.sqrt(var_check))


def test_compute_volatility_validations():
    # NaN
    bad = pd.DataFrame({"AAPL": [0.01, None]})
    with pytest.raises(ValueError, match="NaN"):
        compute_volatility(bad)

    # Inf
    inf_df = pd.DataFrame({"AAPL": [0.01, float("inf")]})
    with pytest.raises(ValueError, match="infinite"):
        compute_volatility(inf_df)


def test_process_returns_pipeline_one_call():
    prices = pd.DataFrame(
        {"AAPL": [100.0, 110.0, 99.0, 105.0], "MSFT": [200.0, 202.0, 201.0, 205.0]},
        index=pd.date_range("2023-01-01", periods=4),
    )
    rets, mu, sigma, summary = process_returns_pipeline(prices)

    assert rets.shape[0] == len(prices) - 1
    assert mu.index.equals(prices.columns)
    assert sigma.index.equals(prices.columns)
    assert summary["n_assets"] == 2
    assert summary["n_obs_daily"] == 3
    assert summary["return_convention"] == "simple"
    assert summary["annualization"] == 252
    assert summary["best_return_ticker"] == mu.idxmax()
    assert summary["lowest_vol_ticker"] == sigma.idxmin()
    assert "degrees_of_freedom_note" in summary


def test_backward_compat_compute_returns_alias():
    prices = pd.DataFrame({"AAPL": [100.0, 110.0]}, index=pd.date_range("2023-01-01", periods=2))
    ret1 = compute_daily_returns(prices, method="simple")
    ret2 = compute_returns(prices, return_type="simple")
    pd.testing.assert_frame_equal(ret1, ret2)