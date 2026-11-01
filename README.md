# 05 - Financial Portfolio Optimization and Risk Allocator

A quantitative finance application that downloads historical market data, estimates risk-return characteristics, and constructs optimal portfolios using Modern Portfolio Theory. The app allows users to select tickers, estimate expected returns and covariance, run optimization (Mean-Variance, Risk Parity, Max Sharpe), analyze risk metrics (VaR, CVaR, drawdown), and visualize results through an interactive Streamlit dashboard.

**Target audience:** Students and practitioners learning portfolio optimization, risk management, and applied linear algebra / optimization in Python. Built as an 8-phase academic project.

## Tech Stack

Pinned versions verified on Python 3.14.2 (Oct 2026):

- **numpy==2.5.4** – Linear algebra, vectorized calculations
- **pandas==3.1.0** – Time series handling, data manipulation
- **scipy==1.18.1** – Optimization solvers (minimize), statistical functions
- **yfinance==1.4.1** – Market data download from Yahoo Finance
- **plotly==7.1.0** – Interactive visualizations (approved viz library)
- **streamlit==1.66.1** – Web app framework for portfolio allocator
- **pytest==9.2.0** – Unit and integration testing

> No databases, cloud SDKs, or unapproved frameworks. Matplotlib omitted intentionally – Plotly is the approved visualization library.

## Repository Structure

Reference structure from Section 3 of project brief:

```
.
├── .devcontainer/
│   └── devcontainer.json
├── data/
│   ├── raw/              # Downloaded CSVs (ignored, kept via .gitkeep)
│   │   └── .gitkeep
│   └── processed/        # Cleaned / processed datasets (ignored)
│       └── .gitkeep
├── notebooks/            # Exploratory notebooks
├── src/
│   ├── __init__.py
│   ├── data/
│   │   └── __init__.py   # Data download & loading
│   ├── preprocessing/
│   │   └── __init__.py   # Returns, cleaning
│   ├── linear_algebra/
│   │   └── __init__.py   # Covariance, expected returns
│   ├── optimization/
│   │   └── __init__.py   # Mean-variance, risk parity, etc.
│   ├── risk/
│   │   └── __init__.py   # VaR, CVaR, drawdown, Sharpe
│   └── visualization/
│       └── __init__.py   # Plotly charts
├── app/                  # Streamlit app (Phase 6)
├── tests/                # pytest suite
│   └── __init__.py
├── docs/
│   └── screenshots/      # Screenshots for final README
├── .gitignore
├── requirements.txt
└── README.md
```

Every Python package folder contains an empty `__init__.py` so imports like `from src.linear_algebra import covariance` work.

## Setup Instructions

### Option A: GitHub Codespaces (Recommended)

1. Clone the repo:
   ```bash
   git clone https://github.com/aryansjainbtech25-sys/05-Financial-Portfolio-Optimization-and-Risk-Allocator.git
   cd 05-Financial-Portfolio-Optimization-and-Risk-Allocator
   ```
2. Open in Codespace: `Code` → `Create codespace on main`
3. Wait for container build – `postCreateCommand` automatically runs:
   ```bash
   python -m pip install --upgrade pip && pip install -r requirements.txt
   ```
4. Verify:
   ```bash
   python --version
   # Python 3.14.2
   pip list | grep -E "numpy|pandas|scipy|yfinance|plotly|streamlit|pytest"
   ```

No manual pip install needed – environment is auto-provisioned.

### Option B: Local Setup

Tested on Python 3.14.2:

```bash
git clone https://github.com/aryansjainbtech25-sys/05-Financial-Portfolio-Optimization-and-Risk-Allocator.git
cd 05-Financial-Portfolio-Optimization-and-Risk-Allocator

# 1. Create venv at repo root
python3 -m venv .venv

# 2. Activate
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Upgrade pip and install pinned deps
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify
which python
# Should point to .venv/bin/python
python -c "import sys; print(sys.executable)"
pip list

# 5. Test persistence
deactivate
source .venv/bin/activate
python --version
```

`.venv/` is ignored by `.gitignore`. Generated CSVs in `data/raw/*.csv` and `data/processed/*.csv` are also ignored – folders stay tracked via `.gitkeep`.

## How to Run

**Current status:** Phase 0 complete (environment setup).

**Phase 6 placeholder:** Streamlit app will be added in `app/`:

```bash
# Future command (not yet available):
streamlit run app/app.py
# Will be served on port 8501 (forwarded in devcontainer)
```

Until then, run notebooks or `pytest`:

```bash
pytest -v
```

## Team Members and Course Context

- **Repository:** `aryansjainbtech25-sys/05-Financial-Portfolio-Optimization-and-Risk-Allocator`
- **Course/Project:** Financial Portfolio Optimization – 8-phase implementation
  - Phase 0: Environment & Skeleton (Tasks 0.1–0.6) – ✅ Complete
  - Phase 1: Data Ingestion (yfinance)
  - Phase 2: Preprocessing & Returns
  - Phase 3: Linear Algebra (Covariance, Expected Returns)
  - Phase 4: Optimization
  - Phase 5: Risk Metrics
  - Phase 6: Visualization & Streamlit App
  - Phase 7: Documentation & Final README expansion
  - Phase 8: Testing & Polish

## License

Academic project – License to be added in Phase 7. All market data via Yahoo Finance is for research/educational purposes per Yahoo terms of use.

---
*Setup verified on Python 3.14.2 | numpy 2.5.4 | pandas 3.1.0 | scipy 1.18.1 | yfinance 1.4.1 | plotly 7.1.0 | streamlit 1.66.1 | pytest 9.2.0*