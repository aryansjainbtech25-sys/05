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

__all__ = ["eigen_decompose"]


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