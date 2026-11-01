# Environment Versions Snapshot
Python: 3.14.2 (Codespace) / 3.9.25 (CI)
- numpy==2.5.4
- pandas==3.1.0
- scipy==1.18.1
- yfinance==1.4.1
- plotly==6.3.0
- kaleido==0.2.1
- tabulate==0.9.0
- streamlit==1.66.1
- pytest==9.2.0

## Pin History & Rationale

- **Phase 0**: `plotly==7.1.0` pinned. Plotly 7.x requires `kaleido>=1.0.0` for static image export (`fig.write_image`).
- **Phase 3 Task 3.7**: `kaleido>=1.0.0` bundles its own Chrome via `choreographer` and fails in Codespace sandbox (browser closes immediately, missing libnss3, needs --no-sandbox). Brave vs Chrome irrelevant — kaleido v1 uses its own chrome at `~/.local/share/choreographer/deps/chrome-linux64/chrome`.
- **Fix**: Downgrade to `plotly==6.3.0` + `kaleido==0.2.1`. Plotly 6.x still supports kaleido 0.2.1 (deprecated but functional), which bundles chromium inside pip wheel and works headless without system Chrome.
- `tabulate==0.9.0` added for pretty-printing eigen_summary_table in CLI and docs (used in `src/linear_algebra/eigen_analysis.py` `print(tabulate(...))` fallback). Legit runtime dependency for docs generation, belongs in requirements.txt.
- Verified: `python -m src.visualization.charts` saves PNGs 148KB/130KB via kaleido 0.2.1 without Chrome.

Smoke test: all imports OK, `fig.write_image(engine="kaleido")` works without Chrome.

## Live Verification

```bash
pip show plotly kaleido tabulate
# plotly 6.3.0, kaleido 0.2.1, tabulate 0.9.0
python - << 'PY'
import plotly.graph_objects as go
fig = go.Figure(go.Scatter(x=[1,2], y=[3,4]))
fig.write_image("/tmp/test.png", engine="kaleido")
print("kaleido 0.2.1 works ✅")
PY
pytest tests/test_covariance.py -v  # 17 passed
```

This matches `pip freeze` and `requirements.txt`.