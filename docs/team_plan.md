# Team Ownership Plan – Phase 0-8

**Repository:** 05-Financial-Portfolio-Optimization-and-Risk-Allocator  
**Python:** 3.14.2 (verified)  
**Team:** 2 members – shared ownership with primary responsibilities

## Ownership Matrix

| Phase / Folder | Primary Owner | Support | Notes |
|----------------|---------------|---------|-------|
| Phase 0: Setup (0.1-0.9) | Both | Both | Skeleton, requirements, devcontainer, README, LICENSE, env verification – completed together |
| `src/data/` | Member A | Member B | yfinance download, data loading |
| `src/preprocessing/` | Member A | Member B | Returns calculation, cleaning |
| `src/linear_algebra/` | Member B | Member A | Covariance, expected returns |
| `src/optimization/` | Member B | Member A | Mean-variance, Max Sharpe, Risk Parity |
| `src/risk/` | Member A | Member B | VaR, CVaR, Sharpe, drawdown |
| `src/visualization/` | Member B | Member A | Plotly charts |
| `app/` (Streamlit) | Both | Both | Integration, UI |
| `tests/` | Both | Both | pytest for each module |
| `docs/` | Both | Both | Screenshots, final README |

## Collaboration Rules

1. **Main branch protected** – PRs for Phase 1+ features
2. **Environment parity** – Both use Codespaces with same devcontainer.json, Python 3.14.2, pinned requirements
3. **Git hygiene** – .venv/, __pycache__/, *.csv ignored; .gitkeep tracks empty data folders
4. **Verification before merge** – pip list matches pins, smoke test passes, git status clean

## Phase 0 Completion Evidence

- Skeleton matches reference tree, all __init__.py present
- requirements.txt pinned: numpy 2.5.4, pandas 3.1.0, scipy 1.18.1, yfinance 1.4.1, plotly 7.1.0, streamlit 1.66.1, pytest 9.2.0
- .venv/ created at repo root, interpreter path: /workspaces/.../.venv/bin/python
- devcontainer rebuild succeeds, auto-installs requirements, forwards 8501
- .gitignore verified: !! .venv/, !! __pycache__/, !! *.csv
- README and LICENSE render on GitHub, setup verified end-to-end
- Smoke test: all 7 imports OK
- Initial commits pushed: a606e60 HEAD -> main

## Next Phase

Phase 1: Data Ingestion – yfinance download to data/raw/



---

## Data Universe (Phase 1.1)

### Constants Location
`src/data/stock_data.py` - single source of truth for all phases

```python
DEFAULT_TICKERS = ["AAPL", "MSFT", "JPM", "JNJ", "PG", "XOM", "AMZN"]
DEFAULT_START_DATE = "2018-01-01"
DEFAULT_END_DATE = None  # resolves to today at call time
TRADING_DAYS_PER_YEAR = 252
```

### Ticker Universe - 7 Tickers Across 6 Sectors

| Ticker | Company | Sector (GICS) | Market Cap | Rationale |
|--------|---------|---------------|------------|-----------|
| AAPL | Apple Inc. | Technology (Consumer Electronics) | Large-cap >$2T | Most liquid US stock, 10y+ clean history, tech anchor |
| MSFT | Microsoft Corp. | Technology (Software - Infrastructure) | Large-cap >$2T | Diversified tech (cloud, software), different sub-sector from AAPL |
| JPM | JPMorgan Chase & Co. | Financial Services | Large-cap >$500B | Largest US bank, represents financials, interest-rate sensitive |
| JNJ | Johnson & Johnson | Healthcare (Pharma) | Large-cap >$350B | Defensive healthcare, low beta, dividend aristocrat |
| PG | Procter & Gamble | Consumer Staples | Large-cap >$350B | Defensive consumer staples, low correlation with tech |
| XOM | Exxon Mobil Corp. | Energy | Large-cap >$400B | Largest US energy, provides inflation/commodity hedge, low correlation with tech |
| AMZN | Amazon.com Inc. | Consumer Discretionary (Internet Retail) | Large-cap >$1.5T | E-commerce/cloud, high growth, consumer discretionary exposure |

**Sector Diversity:** 6 distinct sectors (Technology x2 different industries, Financials, Healthcare, Consumer Staples, Energy, Consumer Discretionary) – exceeds requirement of ≥3 sectors.

**Validation:**
- All 7 have Adj Close data back to 2008+ (well before 2018-01-01 start)
- No penny stocks, no ETFs with structure changes, no recent IPOs
- Average daily volume >10M shares (high liquidity)
- Manual Yahoo Finance check: https://finance.yahoo.com/quote/<TICKER>/history/

### Date Range

- `DEFAULT_START_DATE = "2018-01-01"` – 8+ years of data, includes COVID crash (2020), inflation cycle (2022), rate hikes, provides regime diversity
- `DEFAULT_END_DATE = None` – at runtime resolves to `datetime.today()` or `yfinance` default (today), ensuring always up-to-date but deterministic in code
- `TRADING_DAYS_PER_YEAR = 252` – industry standard annualization constant

### Annualization Convention (for Phases 2-5)

```
mu_annual = mu_daily × 252
Sigma_annual = Sigma_daily × 252
sigma_annual = sigma_daily × √252
Sharpe_annual = Sharpe_daily × √252
```

Fixing 252 here avoids classic bugs of using 250, 252, 260, or 365 inconsistently across modules.