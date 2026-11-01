# Eigen Analysis Report

**Generated:** 2026-11-01
**Tickers:** AAPL, MSFT, JPM, JNJ, PG, XOM, AMZN
**n_obs:** 2219 daily returns

## Verification

- Symmetric: True
- PSD: True, min_eig daily 7.289999e-05, annual 1.837080e-02
- Condition number: 14.38
- Diag matches σ: True, max error 5.55e-17

## Σ_annual Eigen Summary

Total variance (trace): 0.544974

| | eigenvalue | explained_ratio | cumulative_ratio | AAPL | MSFT | JPM | JNJ | PG | XOM | AMZN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PC1 | 0.264143 | 0.484689 | 0.484689 | 0.484255 | 0.470569 | 0.365726 | 0.156414 | 0.169707 | 0.279310 | 0.528229 |
| PC2 | 0.104158 | 0.191124 | 0.675813 | -0.088695 | -0.209832 | 0.429454 | 0.197247 | 0.126696 | 0.684461 | -0.490130 |
| PC3 | 0.053663 | 0.098469 | 0.774282 | 0.343110 | 0.088409 | -0.067570 | 0.460535 | 0.538946 | -0.426722 | -0.430404 |
| PC4 | 0.039754 | 0.072947 | 0.847229 | -0.525834 | -0.031074 | 0.683462 | 0.140506 | 0.112488 | -0.433263 | 0.187888 |
| PC5 | 0.036392 | 0.066778 | 0.914006 | 0.457671 | 0.079746 | 0.445020 | -0.429834 | -0.413523 | -0.276164 | -0.392568 |
| PC6 | 0.028493 | 0.052284 | 0.966291 | -0.391373 | 0.846302 | -0.104358 | -0.088107 | -0.030077 | 0.072218 | -0.325311 |
| PC7 | 0.018371 | 0.033709 | 1.000000 | 0.027746 | 0.056276 | -0.034559 | 0.715733 | -0.692911 | -0.047004 | -0.016110 |

## Correlation Matrix Eigen Summary

Total variance (trace = N): 7.000000

| | eigenvalue | explained_ratio | cumulative_ratio | AAPL | MSFT | JPM | JNJ | PG | XOM | AMZN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PC1 | 3.120445 | 0.445778 | 0.445778 | 0.447898 | 0.444864 | 0.396244 | 0.323450 | 0.333034 | 0.295460 | 0.376358 |
| PC2 | 1.257575 | 0.179654 | 0.625431 | 0.239035 | 0.362543 | -0.181278 | -0.497135 | -0.386919 | -0.341015 | 0.515191 |
| PC3 | 0.966610 | 0.138087 | 0.763519 | -0.063263 | -0.080212 | 0.430312 | -0.351210 | -0.503252 | 0.652297 | -0.047876 |
| PC4 | 0.467622 | 0.066803 | 0.830322 | -0.306928 | -0.005542 | 0.603626 | 0.381278 | -0.366181 | -0.496870 | 0.122719 |
| PC5 | 0.463093 | 0.066156 | 0.896478 | 0.038259 | 0.017274 | -0.495108 | 0.612809 | -0.521654 | 0.272861 | 0.176053 |
| PC6 | 0.407675 | 0.058239 | 0.954717 | 0.741460 | -0.061814 | 0.101963 | 0.062938 | -0.273462 | -0.220812 | -0.555445 |
| PC7 | 0.316979 | 0.045283 | 1.000000 | -0.304656 | 0.812450 | -0.071189 | 0.004417 | -0.071597 | 0.028800 | -0.485867 |

## First Principal Directions

### Σ_annual PC1

| Ticker | Loading |
| --- | --- |
| AMZN | 0.528229 |
| AAPL | 0.484255 |
| MSFT | 0.470569 |
| JPM | 0.365726 |
| XOM | 0.279310 |
| PG | 0.169707 |
| JNJ | 0.156414 |

### Corr PC1

| Ticker | Loading |
| --- | --- |
| AAPL | 0.447898 |
| MSFT | 0.444864 |
| JPM | 0.396244 |
| AMZN | 0.376358 |
| PG | 0.333034 |
| JNJ | 0.323450 |
| XOM | 0.295460 |

## Interpretation

The first principal component explains 48.5% of total variance in Σ_annual and 44.6% in the correlation matrix. The first three PCs together explain 77.4% (Σ) and 76.4% (Corr), indicating the portfolio has effectively ~3 independent risk factors out of 7 tickers. For Σ_annual, PC1 loadings are all same sign (AMZN:0.528, AAPL:0.484, MSFT:0.471...), consistent with a market-factor interpretation where all equities move together. For Corr, PC1 loadings are all same sign, also suggesting market mode. Eigenvalues near zero would indicate near-collinear tickers or overfit noise – here min eigenvalue 7.29e-05 (daily) and 1.84e-02 (annual) are safely positive, so no near-zero direction. Condition number 14.4 is moderate (10-200 typical for 7 tickers over 5 years), so no severe multicollinearity. Note that Σ and Corr have different eigenproblems: Σ's eigenvalues are variances in return units (scale matters for portfolio variance w^T Σ w), while Corr's eigenvalues are variances of standardized returns (unit-free, used for heatmap).
