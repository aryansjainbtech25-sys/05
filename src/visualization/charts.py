"""
src/visualization/charts.py
Phase 3.7 – Matrix Visualization (Correlation Heatmap + Eigenvalue Bar)

Plotly-only (approved stack), no seaborn/matplotlib.
Functions return go.Figure, do not call .show() inside – caller decides display/save/Streamlit.

Mathematical:
- Heatmap visualizes Corr_{ij} – off-diagonal near ±1 = near-collinear, near 0 = diversification potential.
  Directly relevant to why some eigenvalues near zero.
- Eigenvalue bar chart visualizes risk budget across independent directions. Steep drop-off = covariance effectively low-rank,
  common feature of equity data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

__all__ = ["plot_correlation_heatmap", "plot_eigenvalue_spectrum", "plot_covariance_heatmap"]


def plot_correlation_heatmap(Corr: pd.DataFrame) -> go.Figure:
    """
    Plot correlation matrix heatmap.

    Args:
        Corr: (N,N) DataFrame, values in [-1,1], index/columns tickers

    Returns:
        go.Figure: square heatmap, ticker labels both axes, colorscale RdBu centered at 0, range [-1,1],
                   annotations with numeric correlation rounded to 2 decimals

    Title: "Correlation Matrix — Daily Returns"
    """
    if Corr.empty:
        raise ValueError("plot_correlation_heatmap received empty DataFrame")

    if Corr.shape[0] != Corr.shape[1]:
        raise ValueError(f"Correlation must be square, got {Corr.shape}")

    # Ensure values in [-1,1]
    arr = Corr.values
    # Clip for display but keep original for annotations? Use clipped for color scale
    # Annotations from original

    fig = go.Figure(
        data=go.Heatmap(
            z=arr,
            x=list(Corr.columns),
            y=list(Corr.index),
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlation"),
            hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
        )
    )

    # Add annotations rounded to 2 decimals
    annotations = []
    for i, row_label in enumerate(Corr.index):
        for j, col_label in enumerate(Corr.columns):
            val = arr[i, j]
            # Choose text color based on background intensity
            # Dark background for strong correlations
            text_color = "white" if abs(val) > 0.5 else "black"
            annotations.append(
                dict(
                    x=col_label,
                    y=row_label,
                    text=f"{val:.2f}",
                    showarrow=False,
                    font=dict(color=text_color, size=10),
                )
            )

    fig.update_layout(
        title="Correlation Matrix — Daily Returns",
        xaxis_title="Ticker",
        yaxis_title="Ticker",
        width=600,
        height=600,
        annotations=annotations,
        template="plotly_white",
        yaxis=dict(autorange="reversed"),  # Match matrix display
    )

    # Square aspect
    fig.update_layout(
        xaxis=dict(constrain="domain"),
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )

    return fig


def plot_covariance_heatmap(Sigma: pd.DataFrame, title: str = "Covariance Matrix — Daily Returns") -> go.Figure:
    """
    Optional helper: covariance heatmap (not required but useful for debugging).

    Args:
        Sigma: (N,N) covariance DataFrame
        title: chart title

    Returns:
        go.Figure
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=Sigma.values,
            x=list(Sigma.columns),
            y=list(Sigma.index),
            colorscale="Viridis",
            colorbar=dict(title="Covariance"),
            hovertemplate="%{y} vs %{x}: %{z:.6f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Ticker",
        yaxis_title="Ticker",
        width=600,
        height=600,
        template="plotly_white",
        yaxis=dict(autorange="reversed"),
    )

    return fig


def plot_eigenvalue_spectrum(eigen_dict: dict) -> go.Figure:
    """
    Plot eigenvalue spectrum bar chart + cumulative explained variance line.

    Args:
        eigen_dict: dict from eigen_decompose, containing values, explained_variance_ratio, cumulative_explained

    Returns:
        go.Figure: bar chart eigenvalues descending x=PC1..PCN y=eigenvalue,
                   second trace cumulative explained variance on secondary y-axis line+markers

    Title: "Eigenvalue Spectrum and Cumulative Explained Variance"
    """
    values = np.asarray(eigen_dict["values"])
    explained = np.asarray(eigen_dict.get("explained_variance_ratio", values / values.sum()))
    cumulative = np.asarray(eigen_dict.get("cumulative_explained", np.cumsum(explained)))
    N = len(values)

    pc_labels = [f"PC{i+1}" for i in range(N)]

    fig = go.Figure()

    # Bar chart eigenvalues
    fig.add_trace(
        go.Bar(
            x=pc_labels,
            y=values,
            name="Eigenvalue",
            marker_color="steelblue",
            yaxis="y",
            text=[f"{v:.4f}" for v in values],
            textposition="auto",
            hovertemplate="%{x}: %{y:.6f}<extra></extra>",
        )
    )

    # Line cumulative explained variance on secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=pc_labels,
            y=cumulative,
            name="Cumulative Explained",
            mode="lines+markers",
            marker=dict(color="indianred", size=8),
            line=dict(color="indianred", width=2),
            yaxis="y2",
            hovertemplate="%{x}: %{y:.2%}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Eigenvalue Spectrum and Cumulative Explained Variance",
        xaxis_title="Principal Component",
        yaxis=dict(title="Eigenvalue", side="left"),
        yaxis2=dict(
            title="Cumulative Explained Variance",
            side="right",
            overlaying="y",
            range=[0, 1.05],
            tickformat=".0%",
        ),
        legend=dict(x=0.7, y=0.2),
        template="plotly_white",
        height=500,
        width=700,
        hovermode="x unified",
    )

    return fig



if __name__ == "__main__":
    from pathlib import Path
    import sys
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        import yfinance
    except ModuleNotFoundError:
        import types
        sys.modules["yfinance"] = types.ModuleType("yfinance")

    from src.data.stock_data import load_prices
    from src.preprocessing.returns import process_returns_pipeline
    from src.linear_algebra.covariance import compute_covariance_matrix, compute_correlation_matrix
    from src.linear_algebra.eigen_analysis import eigen_decompose

    processed_dir = Path("data/processed")
    if not processed_dir.exists():
        processed_dir = repo_root / "data" / "processed"

    csv_files = list(processed_dir.glob("prices_clean_*.csv"))
    if not csv_files:
        print("No processed CSV found")
        raise SystemExit(1)

    latest = max(csv_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading {latest}")
    prices = load_prices(latest)
    rets, _, _, _ = process_returns_pipeline(prices)

    Sigma = compute_covariance_matrix(rets)
    Corr = compute_correlation_matrix(Sigma)
    eig = eigen_decompose(Sigma)

    fig1 = plot_correlation_heatmap(Corr)
    fig2 = plot_eigenvalue_spectrum(eig)

    print(f"fig1 7x7: {fig1.data[0].z.shape}, type {type(fig1)}")
    print(f"fig2 bars {len(fig2.data[0].y)}, cum ends {fig2.data[1].y[-1]}")

    screenshots_dir = Path("docs/screenshots")
    if not screenshots_dir.exists():
        screenshots_dir = repo_root / "docs" / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    def save_fig(fig, name, **kwargs):
        png_path = screenshots_dir / f"{name}.png"
        html_path = screenshots_dir / f"{name}.html"
        # Try kaleido 0.2.1 first (plotly 6.x compatible)
        try:
            fig.write_image(str(png_path), engine="kaleido", **kwargs)
            print(f"Saved {png_path} via kaleido 0.2.1")
            return True
        except Exception as e:
            print(f"kaleido 0.2.1 failed for {name}: {e}")

        # Try default (kaleido v1)
        try:
            fig.write_image(str(png_path), **kwargs)
            print(f"Saved {png_path} via default engine")
            return True
        except Exception as e:
            print(f"default write_image failed for {name}: {e}")

        # Try with chrome auto-install
        try:
            import kaleido
            if hasattr(kaleido, "get_chrome_sync"):
                kaleido.get_chrome_sync()
            fig.write_image(str(png_path), **kwargs)
            print(f"Saved {png_path} after get_chrome_sync")
            return True
        except Exception as e:
            print(f"chrome fallback failed for {name}: {e}")

        # Last resort HTML
        try:
            fig.write_html(str(html_path))
            print(f"Saved fallback HTML {html_path}")
        except Exception as e:
            print(f"HTML fallback also failed: {e}")
        return False

    ok1 = save_fig(fig1, "corr_heatmap", width=600, height=600, scale=2)
    ok2 = save_fig(fig2, "eigen_spectrum", width=700, height=500, scale=2)

    if ok1 and ok2:
        print("All PNGs saved successfully")
    else:
        print("Some PNGs failed - check kaleido version")
    print("Done")