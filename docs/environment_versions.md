# Environment Versions Snapshot

**Verified on:** 2026-11-01 (Phase 0.8)  
**Python:** 3.14.2  
**Environment:** `.venv/` at `/workspaces/05-Financial-Portfolio-Optimization-and-Risk-Allocator/.venv/bin/python` + Codespace devcontainer `mcr.microsoft.com/devcontainers/python:3.14-bullseye`

## Pinned Dependencies (from requirements.txt)

| Package | Pinned Version | Import Verified | Source |
|---------|----------------|-----------------|--------|
| numpy | 2.5.4 | ✅ | `import numpy; numpy.__version__` |
| pandas | 3.1.0 | ✅ | `import pandas; pandas.__version__` |
| scipy | 1.18.1 | ✅ | `import scipy; scipy.__version__` |
| yfinance | 1.4.1 | ✅ | `import yfinance; yfinance.__version__` |
| plotly | 7.1.0 | ✅ | `import plotly; plotly.__version__` |
| streamlit | 1.66.1 | ✅ | `import streamlit; streamlit.__version__` |
| pytest | 9.2.0 | ✅ | `import pytest; pytest.__version__` |

## Smoke Test Output

```
Python: 3.14.2
numpy: 2.5.4 - OK
pandas: 3.1.0 - OK
yfinance: 1.4.1 - OK
scipy: 1.18.1 - OK
plotly: 7.1.0 - OK
streamlit: 1.66.1 - OK
pytest: 9.2.0 - OK
```

All imports succeed with no ModuleNotFoundError. Versions match pins in requirements.txt.

## How to Reproduce

```bash
source .venv/bin/activate
python -c "import numpy, pandas, yfinance, scipy, plotly, streamlit, pytest; print('All imports OK')"
python -c "import numpy, pandas, yfinance, scipy, plotly, streamlit, pytest; [print(f'{m.__name__}: {getattr(m, "__version__", "unknown")}') for m in [numpy, pandas, yfinance, scipy, plotly, streamlit, pytest]]"
```

Or run temporary script:

```bash
python smoke_test_temp.py
# then delete: rm smoke_test_temp.py
```