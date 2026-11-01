"""
src/data/stock_data.py
Phase 1.1 – Data Universe Constants

Defines deterministic defaults used by all downstream phases.
These constants are the single source of truth for ticker universe,
date window, and annualization factor.
"""

# Universe of large-cap liquid US tickers across >=3 sectors, >=10y history
# All have clean Adj Close history back to at least 2008.
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
# 252 trading days/year is industry standard (not 250 or 365)
# Formulas: mu_annual = mu_daily * 252, Sigma_annual = Sigma_daily * 252, sigma_annual = sigma_daily * sqrt(252)
TRADING_DAYS_PER_YEAR = 252