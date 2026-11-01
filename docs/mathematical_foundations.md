# Mathematical Foundations — Covariance and Eigen Analysis (Phase 3)

> This document demonstrates meaningful applications of Linear Algebra for Financial Portfolio Optimization.
> Code lives in `src/linear_algebra/`; this doc contains only math. Extended in Phase 4 (Optimization) and Phase 5 (Risk).

---

## 1. Notation

| Symbol | Space / Size | Meaning |
| :--- | :--- | :--- |
| $T$ | $\mathbb{N}$ | Number of trading days (observations) |
| $N$ | $\mathbb{N}$ | Number of assets (tickers), $N=7$ in this project |
| $X \in \mathbb{R}^{T \times N}$ | $T \times N$ | Daily returns matrix, $X_{t,i}=r_{t,i}$ |
| $r_{:,i}$ | $\mathbb{R}^T$ | Return time series of asset $i$ (column $i$ of $X$) |
| $\mu \in \mathbb{R}^N$ | $N$ | Column means $\mu_i = \frac{1}{T}\sum_{t=1}^T r_{t,i}$ |
| $\mathbf{1}_T \in \mathbb{R}^T$ | $T$ | Vector of ones, length $T$ |
| $\tilde{X} \in \mathbb{R}^{T \times N}$ | $T \times N$ | Centered returns $\tilde{X}=X - \mathbf{1}_T \mu^\top$ |
| $\Sigma \in \mathbb{R}^{N \times N}$ | $N \times N$ | Covariance matrix, daily or annualized |
| $\Sigma_{\text{daily}}$ | $N \times N$ | $\tilde{X}^\top \tilde{X}/(T-1)$ |
| $\Sigma_{\text{annual}}$ | $N \times N$ | $252 \cdot \Sigma_{\text{daily}}$ |
| $D = \text{diag}(\sigma_i)$ | $N \times N$ | Diagonal matrix of volatilities |
| $\sigma \in \mathbb{R}^N$ | $N$ | $\sigma_i = \sqrt{\Sigma_{ii}}$ |
| $\text{Corr} \in \mathbb{R}^{N \times N}$ | $N \times N$ | Correlation matrix, unit-free |
| $w \in \mathbb{R}^N$ | $N$ | Portfolio weights |
| $V \in \mathbb{R}^{N \times N}$ | $N \times N$ | Orthogonal eigenvector matrix, $V^\top V = I$ |
| $\Lambda = \text{diag}(\lambda_i)$ | $N \times N$ | Eigenvalue matrix, $\lambda_1 \ge \dots \ge \lambda_N \ge 0$ |
| $\lambda_i$ | $\mathbb{R}_{\ge 0}$ | Eigenvalue = variance along eigenvector $v_i$ |

---

## 2. Covariance Derivation

### 2.1 Population definition

For two random returns $X_i, X_j$:

$$
\text{Cov}(X_i, X_j) = \mathbb{E}\left[(X_i - \mu_i)(X_j - \mu_j)\right]
$$

where $\mu_i = \mathbb{E}[X_i]$.

In particular $\text{Var}(X_i)=\text{Cov}(X_i,X_i)$.

### 2.2 Sample estimator

Given $T$ observations $r_{t,i}$, unbiased sample estimator uses Bessel's correction $T-1$:

$$
\hat{\mu}_i = \frac{1}{T}\sum_{t=1}^T r_{t,i}
$$

$$
\hat{\Sigma}_{ij} = \frac{1}{T-1}\sum_{t=1}^T (r_{t,i}-\hat{\mu}_i)(r_{t,j}-\hat{\mu}_j)
$$

$$
\hat{\Sigma}_{ii} = \frac{1}{T-1}\sum_{t=1}^T (r_{t,i}-\hat{\mu}_i)^2
$$

Division by $T-1$ not $T$ makes $\mathbb{E}[\hat{\Sigma}] = \Sigma$ unbiased.

### 2.3 Matrix form $\Sigma = \frac{1}{T-1}\tilde{X}^\top \tilde{X}$

Define centered matrix:

$$
\tilde{X} = X - \mathbf{1}_T \mu^\top, \quad \tilde{X}_{t,i}=r_{t,i}-\mu_i, \quad \frac{1}{T}\sum_t \tilde{X}_{t,i}=0
$$

Then:

$$
\tilde{X}^\top \tilde{X} \in \mathbb{R}^{N \times N}, \quad (\tilde{X}^\top \tilde{X})_{ij}=\sum_{t=1}^T (r_{t,i}-\mu_i)(r_{t,j}-\mu_j)
$$

Shape contract: $(N \times T) \cdot (T \times N) = N \times N$.

Therefore:

$$
\Sigma_{\text{daily}} = \frac{1}{T-1}\tilde{X}^\top \tilde{X}
$$

$$
\Sigma_{\text{annual}} = 252 \cdot \Sigma_{\text{daily}} \quad \text{(i.i.d. approximation, see §6)}
$$

Implementation uses Task 3.1 primitives: `center_columns`, `transpose`, `matrix_multiply` — no `DataFrame.cov()`.

---

## 3. Symmetry $\Sigma = \Sigma^\top$

From definition:

$$
\Sigma_{ij} = \frac{1}{T-1}\sum_t (r_{t,i}-\mu_i)(r_{t,j}-\mu_j) = \frac{1}{T-1}\sum_t (r_{t,j}-\mu_j)(r_{t,i}-\mu_i) = \Sigma_{ji}
$$

by commutativity of scalar multiplication. Hence $\Sigma = \Sigma^\top$. Float asymmetry $||\Sigma - \Sigma^\top||_{\max} \le 10^{-10}$ is round-off only.

Symmetry enables Hermitian solver `eigh` — real eigenvalues, orthogonal eigenvectors.

---

## 4. Positive Semi-Definiteness $w^\top \Sigma w \ge 0$

For any $w \in \mathbb{R}^N$:

$$
w^\top \Sigma w = w^\top \left(\frac{1}{T-1}\tilde{X}^\top \tilde{X}\right) w = \frac{1}{T-1} (\tilde{X}w)^\top (\tilde{X}w) = \frac{1}{T-1} \|\tilde{X}w\|_2^2 \ge 0
$$

Proof in two lines:

1. $\Sigma = \frac{1}{T-1}\tilde{X}^\top \tilde{X}$ is Gram matrix.
2. $w^\top \tilde{X}^\top \tilde{X} w = \|\tilde{X}w\|^2 \ge 0$, equality only if $\tilde{X}w=0$.

Consequences:
- All eigenvalues $\lambda_i \ge 0$ (Task 3.3 checks via `eigvalsh`)
- Determinant $>0$ is necessary but not sufficient — eigenvalue test is required
- $\Sigma$ is invertible iff strictly PD ($\lambda_{\min}>0$); PSD with $\lambda_{\min}\approx 0$ indicates near-collinear tickers

Condition number $\kappa = \lambda_{\max}/\lambda_{\min}$: 10–200 moderate for 7 tickers over 5y; $>10^6$ warns multicollinearity.

---

## 5. Eigendecomposition $\Sigma = V \Lambda V^\top$

### 5.1 Spectral theorem for symmetric matrices

For symmetric $\Sigma$, there exists orthogonal $V$ ($V^\top V = I$, $V^{-1}=V^\top$) and real diagonal $\Lambda$:

$$
\Sigma = V \Lambda V^\top = \sum_{k=1}^N \lambda_k v_k v_k^\top
$$

where $v_k = V_{:,k}$ is eigenvector, $\lambda_k$ eigenvalue, $\Lambda = \text{diag}(\lambda_1,\dots,\lambda_N)$, sorted $\lambda_1 \ge \dots \ge \lambda_N$ (descending convention for variance-explained).

- $V$ orthogonal, $\|v_k\|=1$, $v_i^\top v_j = 0$ for $i \ne j$
- $\lambda_i \in \mathbb{R}$ guaranteed symmetric; $\lambda_i \ge 0$ guaranteed PSD
- Sign convention: $eigh$ returns arbitrary sign; we enforce largest-magnitude component positive for interpretability (does not change math but changes heatmap colors)

### 5.2 Variance explained

$$
\text{Tr}(\Sigma) = \sum_i \Sigma_{ii} = \sum_k \lambda_k = \text{total variance}
$$

$$
\text{explained}_k = \frac{\lambda_k}{\sum_j \lambda_j}, \quad \text{cumulative}_k = \sum_{j=1}^k \text{explained}_j
$$

First PC explains $\lambda_1 / \text{Tr}(\Sigma)$ of total variance. For 7-stock equity, first 3 PCs typically explain 70–85%, indicating effectively 3 independent risk factors. First eigenvector often has all same sign — market mode.

In eigenbasis $a = V^\top r$, components are uncorrelated:

$$
\text{Var}(a_k) = \lambda_k, \quad \text{Cov}(a_k,a_j)=0 \ (k \ne j)
$$

Reconstruction identity (must hold atol $10^{-10}$):

$$
V \Lambda V^\top \approx \Sigma, \quad \max_{ij} | (V\Lambda V^\top)_{ij} - \Sigma_{ij} | < 10^{-10}
$$

---

## 6. Correlation as Diagonal Scaling

### 6.1 Derivation

Let $\sigma_i = \sqrt{\Sigma_{ii}}$.

$$
\text{Corr}_{ij} = \frac{\Sigma_{ij}}{\sigma_i \sigma_j}
$$

Matrix form with $D = \text{diag}(\sigma_1,\dots,\sigma_N)$:

$$
\text{Corr} = D^{-1} \Sigma D^{-1}
$$

Steps:
1. $\sigma = \sqrt{\text{diag}(\Sigma)}$
2. $D^{-1} = \text{diag}(1/\sigma_i)$
3. $\text{Corr} = D^{-1} \Sigma D^{-1}$

### 6.2 Properties

- $\text{diag}(\text{Corr}) = 1$
- $\text{Corr}_{ij} \in [-1,1]$
- Symmetric, PSD
- Unit-free — measures linear co-movement independent of scale, so tech $300 and staple $40 comparable

### 6.3 Eigenvalues Corr vs Σ

$\text{Corr}$ is covariance of standardized returns $\tilde{X} D^{-1}$. Different eigenproblem from $\Sigma$:
- $\Sigma$ eigenvalues: variances in return units (scale matters for $w^\top \Sigma w$)
- $\text{Corr}$ eigenvalues: variances of standardized returns (unit-free, used for heatmap)

Phase 4 optimization uses $\Sigma$; heatmap uses Corr.

---

## 7. Annualization Conventions

Assume i.i.d. daily returns (approximation).

| Quantity | Daily | Annual | Reason |
| :--- | :--- | :--- | :--- |
| Mean $\mu$ | $\mu_{\text{daily}} = \frac{1}{T}\sum r_t$ | $\mu_{\text{annual}} = 252 \cdot \mu_{\text{daily}}$ | Linearity of expectation, $T$ days |
| Variance $\Sigma_{ii}$ | $\Sigma_{\text{daily}}$ | $\Sigma_{\text{annual}} = 252 \cdot \Sigma_{\text{daily}}$ | $\text{Var}(\sum_{t=1}^{252} r_t) = 252 \cdot \text{Var}(r_t)$ under i.i.d. |
| Volatility $\sigma$ | $\sigma_{\text{daily}} = \sqrt{\Sigma_{\text{daily},ii}}$ | $\sigma_{\text{annual}} = \sqrt{252}\cdot \sigma_{\text{daily}} = \sqrt{\text{diag}(\Sigma_{\text{annual}})}$ | $\sqrt{252}$ scaling, contract $\sqrt{\text{diag}(\Sigma_{\text{annual}})} = \sigma_{\text{annual}}$ |
| Covariance $\Sigma_{ij}$ | $\Sigma_{\text{daily},ij}$ | $252 \cdot \Sigma_{\text{daily},ij}$ | Same as variance |
| Correlation $\text{Corr}$ | $D^{-1}\Sigma_{\text{daily}} D^{-1}$ | Same — $252$ cancels $\frac{252\Sigma_{ij}}{\sqrt{252}\sigma_i \sqrt{252}\sigma_j} = \frac{\Sigma_{ij}}{\sigma_i\sigma_j}$ | Scale-invariant |

Check: $\sigma_{\text{annual}}$ from Phase 2 `compute_volatility` must equal $\sqrt{\text{diag}(\Sigma_{\text{annual}})}$ within $10^{-10}$ (Task 3.3 `verify_covariance_properties` hard requirement).

---

## 8. Why This Matters for Finance

Portfolio variance is exactly the quadratic form:

$$
\sigma_p^2 = w^\top \Sigma w = \sum_{i,j} w_i w_j \Sigma_{ij}
$$

Properties used:

- Symmetry ensures $w^\top \Sigma w$ real
- PSD ensures $\sigma_p^2 \ge 0$ for any $w$
- Eigen decomposition diagonalizes risk: in basis $a=V^\top w$, $\sigma_p^2 = \sum_k \lambda_k a_k^2$

Markowitz problem (foreshadow Phase 4):

$$
\begin{aligned}
\min_w \quad & w^\top \Sigma w \\
\text{s.t.} \quad & \mathbf{1}^\top w = 1, \quad \mu^\top w \ge \mu_{\text{target}}, \quad w \ge 0 \text{ (optional)}
\end{aligned}
$$

- Minimizing $w^\top \Sigma w$ is minimizing portfolio variance (quadratic form)
- Eigenvalues tell risk budget: steep drop-off = low-rank covariance = diversification limited to few factors
- Off-diagonal $\text{Corr}_{ij} \approx 1$ → near-collinear → small eigenvalue → near-zero variance direction → ill-conditioned $\Sigma$ → optimization unstable
- Condition number $\kappa(\Sigma)$ signals when to regularize (shrinkage, factor model)

Heatmap visualizes diversification potential: values near 0 good, near ±1 bad. Eigenvalue spectrum visualizes effective number of risk factors.

---

## 9. Implementation Map

| Formula | Module | Function | Prohibited Call |
| :--- | :--- | :--- | :--- |
| $\tilde{X}=X-\mathbf{1}\mu^\top$ | `linear_algebra/matrix_ops.py` | `center_columns` | — |
| $\Sigma = \tilde{X}^\top \tilde{X}/(T-1)$ | `linear_algebra/covariance.py` | `compute_covariance_matrix` | `DataFrame.cov()` |
| $\text{Corr}=D^{-1}\Sigma D^{-1}$ | `linear_algebra/covariance.py` | `compute_correlation_matrix` | `DataFrame.corr()` |
| $\Sigma = V\Lambda V^\top$ | `linear_algebra/eigen_analysis.py` | `eigen_decompose` | `DataFrame.eigen*` |
| $w^\top \Sigma w$ | `linear_algebra/matrix_ops.py` | `quadratic_form` | — |
| Symmetry/PSD/sigma check | `linear_algebra/covariance.py` | `verify_covariance_properties` | — |
| Variance-explained table | `linear_algebra/eigen_analysis.py` | `eigen_summary_table`, `first_principal_direction` | — |
| Heatmap + spectrum | `visualization/charts.py` | `plot_correlation_heatmap`, `plot_eigenvalue_spectrum` | `seaborn`, `matplotlib` |

Tests in `tests/test_covariance.py` lock identities against Pandas/NumPy oracles and internal consistency.

---

## 10. References for Grader

- Task 3.2: $\Sigma = X^T X/(T-1)$ shape $(N,T)(T,N)=N\times N$, Bessel's correction
- Task 3.3: Symmetry $\Sigma_{ij}=\Sigma_{ji}$ by scalar commutativity; PSD $w^\top\Sigma w = \frac{1}{T-1}\|\tilde{X}w\|^2 \ge 0$; $\sqrt{\text{diag}(\Sigma_{\text{annual}})} = \sigma_{\text{annual}}$ within $10^{-10}$
- Task 3.4: $\text{Corr}_{ij}=\Sigma_{ij}/(\sigma_i\sigma_j)$, $\text{Corr}=D^{-1}\Sigma D^{-1}$, diag 1, $\in[-1,1]$
- Task 3.5: $\Sigma=V\Lambda V^\top$, $eigh$ Hermitian solver, descending, $V^\top V=I$, sign convention largest component positive, reconstruction atol $10^{-10}$
- Task 3.6: Explained $\lambda_k/\text{Tr}(\Sigma)$, cumulative, first PC market mode all same sign for equity universe
- Task 3.7: Heatmap RdBu centered 0 range [-1,1] with 2-decimal annotations; eigenvalue bar + cumulative line secondary y-axis ends 1.0
- Task 3.8: Offline pytest 17 tests prove identities vs Pandas/NumPy oracles

This doc renders with LaTeX on GitHub (MathJax). No code blocks — math only.