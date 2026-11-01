# Returns Summary Report

**Generated:** 2026-11-01
**Convention:** simple returns, annualized with 252 trading days
**Note:** mu and sigma derived from T-1 degrees of freedom (T prices -> T-1 returns)

| Metric | Value |
|--------|-------|
| n_assets | 7 |
| n_obs_prices | 2220 |
| n_obs_daily_returns | 2219 |
| mu_min (annual) | 0.1131 |
| mu_max (annual) | 0.2900 |
| sigma_min (annual) | 0.1976 |
| sigma_max (annual) | 0.3445 |
| best_return_ticker | AAPL (0.2900) |
| lowest_vol_ticker | JNJ (0.1976) |

### Per-Ticker mu_annual and sigma_annual

| Ticker | mu_annual | sigma_annual |
|--------|-----------|--------------|
| AAPL | 0.2900 | 0.3034 |
| MSFT | 0.2571 | 0.2902 |
| JPM | 0.1967 | 0.2844 |
| JNJ | 0.1233 | 0.1976 |
| PG | 0.1131 | 0.2011 |
| XOM | 0.1783 | 0.2993 |
| AMZN | 0.2179 | 0.3445 |

**Sanity:** AAPL sigma typically 0.20-0.40, JNJ 0.12-0.22, XOM 0.20-0.35. If 0.01 or 4.0, annualization bug.

**Annualization:** mu_annual = mu_daily * 252, sigma_annual = sigma_daily * sqrt(252)