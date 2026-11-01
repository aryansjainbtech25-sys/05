"""
src/preprocessing/returns.py
Phase 2.1 – Return Convention and Annualization Constants

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
Log:    r_t = ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})  (not used in main pipeline)

Annualization identities (must be used verbatim in Phase 3,4,5,6):
mu_annual = mu_daily * 252
Sigma_annual = Sigma_daily * 252
sigma_annual = sigma_daily * sqrt(252)

Constants are centralized: TRADING_DAYS_PER_YEAR imported from src.data.stock_data,
not redefined. RETURN_TYPE is the single source of truth for downstream phases.
"""

from src.data.stock_data import TRADING_DAYS_PER_YEAR

RETURN_TYPE = "simple"

__all__ = ["RETURN_TYPE", "TRADING_DAYS_PER_YEAR"]