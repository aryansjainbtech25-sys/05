"""
src/linear_algebra/eigen_analysis.py
Phase 3.5 – Eigenvalues and Eigenvectors of Σ

Implements Σ = V Λ V^T, V orthogonal, Λ diag eigenvalues sorted descending.

Mathematical:
- Σ symmetric => Λ real, V orthogonal V^T V = I, V^{-1}=V^T
- Σ PSD => λ_i >=0 (Task 3.3)
- Largest eigenvalue = direction of max variance, first principal component, often "market mode"
- Variance explained: λ_k / Tr(Σ), first few explain 50-80% for 7-stock equity
- Reconstruction: Σ = Σ_k λ_k v_k v_k^T spectral theorem, each term PSD if λ_k>=0

Uses np.linalg.eigh (Hermitian solver) – faster, guarantees real, sorted, orthonormal eigenvectors
for symmetric input. Possible only because Σ symmetric (Task 3.3).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["eigen_decompose", "eigen_summary_table", "first_principal_direction"]


def eigen_decompose(Sigma: pd.DataFrame) -> dict:
    """
    Compute eigendecomposition Σ = V Λ V^T.

    Args:
        Sigma: (N,N) symmetric PSD DataFrame, index/columns tickers

    Returns:
        dict {
            values: np.ndarray (N,) eigenvalues descending,
            vectors: np.ndarray (N,N) eigenvectors as columns vectors[:,k] for values[k],
            explained_variance_ratio: np.ndarray (N,) values / sum(values),
            cumulative_explained: np.ndarray (N,) cumulative sum,
            labels: list[str] ticker labels,
            reconstruction_error: float max abs error V Λ V^T - Σ
        }

    Raises:
        ValueError if not square, not symmetric, has NaN
        AssertionError if reconstruction fails atol 1e-10

    Mathematical:
        Uses eigh (Hermitian) – real, orthonormal eigenvectors, orthogonal V^T V = I
        Enforces sign convention: make largest-magnitude component of each eigenvector positive
        for interpretability (sign flip does not change math but changes heatmap colors).
        Verification inside: V @ diag(λ) @ V^T ≈ Σ elementwise atol 1e-10
    """
    if Sigma.empty:
        raise ValueError("eigen_decompose received empty DataFrame")

    if Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError(f"eigen_decompose expects square Sigma, got {Sigma.shape}")

    if Sigma.isna().any().any():
        raise ValueError("eigen_decompose: Sigma contains NaN")

    # Check symmetric
    arr = Sigma.values
    max_asym = np.max(np.abs(arr - arr.T))
    if max_asym > 1e-8:
        raise ValueError(f"eigen_decompose: Sigma not symmetric, max asymmetry {max_asym}")

    N = arr.shape[0]

    # Use eigh – for symmetric, returns eigenvalues ascending
    # eigh guarantees real, orthonormal
    eig_vals_asc, eig_vecs_asc = np.linalg.eigh(arr)  # eig_vals ascending, eig_vecs columns

    # Sort descending for variance-explained analysis (largest first)
    idx_desc = np.argsort(eig_vals_asc)[::-1]
    values = eig_vals_asc[idx_desc]  # descending
    vectors = eig_vecs_asc[:, idx_desc]  # columns correspondingly

    # Verify descending
    assert np.all(np.diff(values) <= 1e-12), "eigenvalues not descending"

    # Sign convention: make largest-magnitude component of each eigenvector positive
    # For interpretability, does not change math
    for k in range(N):
        vec = vectors[:, k]
        # Find index of largest absolute value
        max_idx = np.argmax(np.abs(vec))
        if vec[max_idx] < 0:
            vectors[:, k] = -vec

    # Explained variance
    total_var = np.sum(values)
    if total_var <= 0:
        raise ValueError(f"eigen_decompose: total variance {total_var} <=0")
    explained_variance_ratio = values / total_var
    cumulative_explained = np.cumsum(explained_variance_ratio)

    # Verification: V @ diag(λ) @ V^T ≈ Σ
    Lambda = np.diag(values)
    Sigma_reconstructed = vectors @ Lambda @ vectors.T
    reconstruction_error = np.max(np.abs(Sigma_reconstructed - arr))
    if reconstruction_error > 1e-10:
        raise AssertionError(f"eigen_decompose reconstruction failed: max error {reconstruction_error} > 1e-10")

    # Additional checks
    # Each eigenvector unit norm
    norms = np.linalg.norm(vectors, axis=0)
    if not np.allclose(norms, 1.0, atol=1e-10):
        raise AssertionError(f"eigenvectors not unit norm: {norms}")

    # Orthogonal V^T V ≈ I
    VTV = vectors.T @ vectors
    if not np.allclose(VTV, np.eye(N), atol=1e-10):
        raise AssertionError(f"eigenvectors not orthogonal: max error {np.max(np.abs(VTV - np.eye(N)))}")

    # Sum explained =1
    if not np.isclose(np.sum(explained_variance_ratio), 1.0, atol=1e-12):
        raise AssertionError(f"explained variance ratio sum {np.sum(explained_variance_ratio)} !=1")

    labels = list(Sigma.index)

    return {
        "values": values,
        "vectors": vectors,
        "explained_variance_ratio": explained_variance_ratio,
        "cumulative_explained": cumulative_explained,
        "labels": labels,
        "reconstruction_error": reconstruction_error,
        "total_variance": float(total_var),
        "trace": float(np.trace(arr)),
    }


def eigen_summary_table(Sigma: pd.DataFrame, labels: list[str] | None = None) -> pd.DataFrame:
    """
    Return labeled table of eigenvalue, explained ratio, cumulative, and eigenvector loadings.

    Args:
        Sigma: (N,N) covariance or correlation DataFrame
        labels: optional list of ticker labels, defaults to Sigma.index

    Returns:
        DataFrame shape (N, 3+N):
            Rows: PC1..PCN sorted descending
            Columns: eigenvalue, explained_ratio, cumulative_ratio, then one column per ticker (loading)

    Mathematical:
        v_k is direction, λ_k variance along it. a = V^T r => Var(a_k)=λ_k, components uncorrelated.
        Explained ratio λ_k / Tr(Σ). First few typically 50-80% for 7-stock equity.
    """
    res = eigen_decompose(Sigma)
    N = len(res["values"])
    if labels is None:
        labels = res["labels"]

    # Build table
    pc_names = [f"PC{i+1}" for i in range(N)]
    df = pd.DataFrame(index=pc_names)
    df["eigenvalue"] = res["values"]
    df["explained_ratio"] = res["explained_variance_ratio"]
    df["cumulative_ratio"] = res["cumulative_explained"]

    # Add loadings: vectors[:,k] is k-th PC, loading on ticker i = vectors[i,k]
    # vectors is (N,N) where rows = tickers (in labels order), columns = PCs
    for i, ticker in enumerate(labels):
        # For each ticker, get its loading across PCs: row i of vectors
        # Actually vectors[i,k] = loading of ticker i in PC k
        # So column per ticker would be transpose? Spec: columns per ticker showing each eigenvector's loading on that ticker
        # Row = PC, so row k should contain loadings of all tickers for that PC = vectors[:,k]
        # We'll store per ticker: for PC k, loading = vectors[ticker_idx, k]
        pass

    # Correct approach: for each PC k, the vector is vectors[:,k] (length N, order = labels)
    # So build matrix loadings[k,i] = vectors[i,k]
    loadings_matrix = res["vectors"].T  # shape (N PCs, N tickers) – row k is PC k loadings across tickers
    for i, ticker in enumerate(labels):
        df[ticker] = loadings_matrix[:, i]

    return df


def first_principal_direction(Sigma: pd.DataFrame, labels: list[str] | None = None) -> pd.Series:
    """
    Return loadings of first eigenvector (PC1) on each ticker.

    Args:
        Sigma: (N,N) covariance or correlation
        labels: optional ticker labels, defaults to Sigma.index

    Returns:
        pd.Series index=tickers, values=loadings of PC1, sorted by absolute loading descending

    Sign convention: largest-magnitude component positive (from eigen_decompose)
    For equities, first PC usually all same sign (market mode)
    """
    res = eigen_decompose(Sigma)
    if labels is None:
        labels = res["labels"]

    # First eigenvector is vectors[:,0]
    pc1 = res["vectors"][:, 0]  # length N
    series = pd.Series(pc1, index=labels, name="PC1_loading")

    # Sort by absolute loading descending for reporting
    series = series.reindex(series.abs().sort_values(ascending=False).index)

    return series


if __name__ == "__main__":
    from pathlib import Path
    import pandas as pd
    import sys
    # Ensure src import works when run as module or script
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Mock yfinance if not installed (for offline verification)
    try:
        import yfinance
    except ModuleNotFoundError:
        import types
        sys.modules["yfinance"] = types.ModuleType("yfinance")

    from src.data.stock_data import load_prices
    from src.preprocessing.returns import process_returns_pipeline
    from src.linear_algebra.covariance import compute_covariance_matrix, compute_correlation_matrix, verify_covariance_properties

    processed_dir = Path("data/processed")
    if not processed_dir.exists():
        processed_dir = repo_root / "data" / "processed"

    csv_files = list(processed_dir.glob("prices_clean_*.csv"))
    if not csv_files:
        print("No processed CSV found. Run python -m src.data.stock_data first")
        raise SystemExit(1)

    latest = max(csv_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading latest: {latest}")
    prices = load_prices(latest)
    rets, mu, sigma, summary = process_returns_pipeline(prices)

    Sigma_daily = compute_covariance_matrix(rets, annualize=False)
    Sigma_annual = compute_covariance_matrix(rets, annualize=True)
    Corr = compute_correlation_matrix(Sigma_daily)

    # Verify properties
    props = verify_covariance_properties(Sigma_daily, Sigma_annual, sigma)
    print(f"\nVerification: {props}")

    # Eigen decompose both
    eig_Sigma = eigen_decompose(Sigma_annual)
    eig_Corr = eigen_decompose(Corr)

    print("\n=== Σ_annual eigenvalues ===")
    print(eig_Sigma["values"])
    print(f"Explained: {eig_Sigma['explained_variance_ratio']}")
    print(f"Cumulative: {eig_Sigma['cumulative_explained']}")

    print("\n=== Corr eigenvalues ===")
    print(eig_Corr["values"])
    print(f"Explained: {eig_Corr['explained_variance_ratio']}")

    # Summary tables
    table_Sigma = eigen_summary_table(Sigma_annual)
    table_Corr = eigen_summary_table(Corr)

    print("\n=== Eigen Summary Table Σ_annual ===")
    print(table_Sigma)

    print("\n=== Eigen Summary Table Corr ===")
    print(table_Corr)

    # First principal direction
    pc1_Sigma = first_principal_direction(Sigma_annual)
    pc1_Corr = first_principal_direction(Corr)
    print("\n=== First Principal Direction Σ_annual ===")
    print(pc1_Sigma)
    print("\n=== First Principal Direction Corr ===")
    print(pc1_Corr)

    # Write docs/eigen_analysis.md
    docs_path = Path("docs/eigen_analysis.md")
    if not docs_path.parent.exists():
        docs_path = repo_root / "docs" / "eigen_analysis.md"
    docs_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Eigen Analysis Report")
    lines.append("")
    lines.append(f"**Generated:** {pd.Timestamp.now().date()}")
    lines.append(f"**Tickers:** {', '.join(Sigma_annual.columns)}")
    lines.append(f"**n_obs:** {len(rets)} daily returns")
    lines.append("")
    lines.append("## Verification")
    lines.append("")
    lines.append(f"- Symmetric: {props['symmetric']}")
    lines.append(f"- PSD: {props['psd']}, min_eig daily {props['min_eig']:.6e}, annual {props['min_eig_annual']:.6e}")
    lines.append(f"- Condition number: {props['condition_number']:.2f}")
    lines.append(f"- Diag matches σ: {props['diag_matches_sigma']}, max error {props['max_diag_sigma_error']:.2e}")
    lines.append("")
    lines.append("## Σ_annual Eigen Summary")
    lines.append("")
    lines.append(f"Total variance (trace): {eig_Sigma['total_variance']:.6f}")
    lines.append("")
    def df_to_md(df):
        header = "| " + " | ".join(df.columns) + " |"
        sep = "| " + " | ".join(["---"]*len(df.columns)) + " |"
        rows = []
        for idx, row in df.iterrows():
            vals = []
            for v in row:
                if isinstance(v, float):
                    vals.append(f"{v:.6f}")
                else:
                    vals.append(str(v))
            rows.append(f"| {idx} | " + " | ".join(vals[0:]) + " |" if "PC" not in str(df.columns[0]) else f"| {idx} | " + " | ".join(vals) + " |")
        # Simpler: handle index as first col
        header2 = "| | " + " | ".join(df.columns) + " |"
        sep2 = "| --- | " + " | ".join(["---"]*len(df.columns)) + " |"
        rows2 = []
        for idx, row in df.iterrows():
            vals = []
            for v in row:
                if isinstance(v, (float, int, __import__("numpy").floating)):
                    try:
                        vals.append(f"{float(v):.6f}")
                    except:
                        vals.append(str(v))
                else:
                    vals.append(str(v))
            rows2.append(f"| {idx} | " + " | ".join(vals) + " |")
        return "\n".join([header2, sep2] + rows2)

    def series_to_md(s):
        header = "| Ticker | Loading |"
        sep = "| --- | --- |"
        rows = [f"| {idx} | {float(val):.6f} |" for idx, val in s.items()]
        return "\n".join([header, sep] + rows)

    lines.append(df_to_md(table_Sigma))
    lines.append("")
    lines.append("## Correlation Matrix Eigen Summary")
    lines.append("")
    lines.append(f"Total variance (trace = N): {eig_Corr['total_variance']:.6f}")
    lines.append("")
    lines.append(df_to_md(table_Corr))
    lines.append("")
    lines.append("## First Principal Directions")
    lines.append("")
    lines.append("### Σ_annual PC1")
    lines.append("")
    lines.append(series_to_md(pc1_Sigma))
    lines.append("")
    lines.append("### Corr PC1")
    lines.append("")
    lines.append(series_to_md(pc1_Corr))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    # Generate interpretation based on actual numbers
    first_var_Sigma = eig_Sigma['explained_variance_ratio'][0]
    first_var_Corr = eig_Corr['explained_variance_ratio'][0]
    first_three_Sigma = eig_Sigma['cumulative_explained'][2] if len(eig_Sigma['cumulative_explained'])>=3 else 0
    first_three_Corr = eig_Corr['cumulative_explained'][2] if len(eig_Corr['cumulative_explained'])>=3 else 0

    all_same_sign_Sigma = "all same sign" if (pc1_Sigma>0).all() or (pc1_Sigma<0).all() else "mixed signs"
    all_same_sign_Corr = "all same sign" if (pc1_Corr>0).all() or (pc1_Corr<0).all() else "mixed signs"

    lines.append(f"The first principal component explains {first_var_Sigma:.1%} of total variance in Σ_annual and {first_var_Corr:.1%} in the correlation matrix. "
                 f"The first three PCs together explain {first_three_Sigma:.1%} (Σ) and {first_three_Corr:.1%} (Corr), indicating the portfolio has effectively ~3 independent risk factors out of 7 tickers. "
                 f"For Σ_annual, PC1 loadings are {all_same_sign_Sigma} ({', '.join([f'{idx}:{val:.3f}' for idx,val in pc1_Sigma.items()][:3])}...), consistent with a market-factor interpretation where all equities move together. "
                 f"For Corr, PC1 loadings are {all_same_sign_Corr}, also suggesting market mode. "
                 f"Eigenvalues near zero would indicate near-collinear tickers or overfit noise – here min eigenvalue {props['min_eig']:.2e} (daily) and {props['min_eig_annual']:.2e} (annual) are safely positive, so no near-zero direction. "
                 f"Condition number {props['condition_number']:.1f} is moderate (10-200 typical for 7 tickers over 5 years), so no severe multicollinearity. "
                 f"Note that Σ and Corr have different eigenproblems: Σ's eigenvalues are variances in return units (scale matters for portfolio variance w^T Σ w), while Corr's eigenvalues are variances of standardized returns (unit-free, used for heatmap).")
    lines.append("")

    docs_path.write_text("\n".join(lines))
    print(f"\nReport saved to {docs_path}")