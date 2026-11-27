Here is the updated specification document, adjusting the original specification to incorporate non-parametric pre-geometry feature selection, post-clustering feature importance profiling, and automated reporting of dropped/retained features and their rationales. [1]

Specification: Cluster Feature Analysis, Geometry Profiling & Reporting


1. Motivation: The Information Double-Counting Problem

In unsupervised clustering—especially on longitudinal or multivariate tabular data—there is no target variable y to weight features or correct distorted scales. Standard clustering pipelines treat every column as an independent, orthogonal axis of space. [1]
• The Double-Counting Issue: When variables are correlated across features or across time lags ($x_t, x_{t-1}, x_{t-2}$), standard Euclidean distance sums squared differences across redundant columns. Following the Principle of Inclusion-Exclusion applied to measurement ($H(X_1, X_2) = H(X_1) + H(X_2) - I(X_1; X_2)$), naive distance metrics fail to subtract overlapping mutual information. Additionally, sparse, near-constant, or redundant columns inflate covariance condition numbers and induce k-NN distance concentration in high dimensions. [1]
• The Consequence: Shared baseline variance and slow-moving ("sticky") variables get multiplied, stretching the geometric space into artificial ellipsoids. As a result, clustering algorithms produce exaggerated, artificial partitions driven by redundant variance or noise rather than discovering genuine behavioral states or structural subgroups. [1]

2. Core Factors Governing Clustering & Feature Selection

Any defensible clustering design must triangulate its feature space and distance metric across three sequential dimensions:
1. Pre-Geometry Feature Hygiene (Non-Parametric Pruning):
	• Removes degenerate, hyper-sparse, and monotonically redundant columns before geometric shape is evaluated, using strictly distribution-free (rank/order-statistic/entropy) rules so no Gaussian or linear cluster geometry is prematurely assumed.
2. Impact of Similarity & Domain Invariance (Measurement Scale):
	• Defines what differences actually matter in the problem domain and which mathematical transformations should equal zero distance.
	• Governed by Stevens' Levels of Measurement: preserving absolute levels (ratio scale), ignoring baseline offsets via translation invariance (interval scale / demeaned rows), ignoring overall magnitude via scale invariance (cosine), or ignoring both via affine invariance (row-wise z-scores). [1]
3. Cluster Geometry (Data Shape & Dependence Structure):
	• Defines how data points occupy high-dimensional space—specifically whether groups form isotropic spheres, tilted linear ellipsoids, varying-density clouds, or curved non-linear manifolds—and whether dependence is linear (correlation) or non-linear (mutual information). [1]

3. Non-Parametric Feature Selection & Reporting Requirements

Because feature pruning executes prior to geometry detection, all screening rules must rely exclusively on order statistics, ranks, and empirical entropy rather than parametric moments (mean, std, Pearson correlation):

3.1 Three Sequential Non-Parametric Screening Gates

1. Degeneracy & Sparsity Screen (Zero-Information Filter):
	• Metrics: Dominant mode frequency ($p_{\text{mode}}$) and Normalized Discrete Shannon Entropy ($H_{\text{norm}}(X_j) = -\sum p_k \log(p_k) / \log(N_{\text{unique}})$).
	• Rule: Drop feature j if $p_{\text{mode}} > 0.98$, $N_{\text{unique}} \le 1$, or $H_{\text{norm}}(X_j) < 0.05$.
	• Rationale: Near-constant or hyper-sparse columns carry negligible information while causing singular covariance matrices and unstable quantile transforms.
2. Non-Parametric Spread Screen (IQR & MAD Filter):
	• Metrics: Interquartile Range (IQR = Q₇₅ - Q₂₅) and Median Absolute Deviation (MAD).
	• Rule: Drop feature j if $\text{IQR}(X_j) = 0$ and $\text{MAD}(X_j) = 0$ (unless preserved as a valid binary flag with minority frequency ≥ 0.02).
	• Rationale: Prevents zero-division distortion under robust scaling and eliminates features where ≥ 50% of observations collapse onto a single value.
3. Monotonic Redundancy Screen (Spearman Rank Filter):
	• Metric: Pairwise Spearman rank correlation $|\rho_s(X_i, X_j)|$.
	• Rule: For any pair with $|\rho_s(X_i, X_j)| > 0.95$, retain the feature with higher normalized entropy $H_{\text{norm}}$ and drop the redundant twin, logging the retained proxy feature.
	• Rationale: Eliminates duplicate information (e.g., perfectly collinear lags or derived totals) before geometry estimation without assuming linear scaling.
The sample-to-feature ratio is reported as a reliability diagnostic. It is not a
feature-selection gate and must not be used to rank or arbitrarily remove
features.

3.2 Post-Clustering Feature Importance & Separation Profiling

Once clusters are fitted using the geometry-matched algorithm, the package must score each retained feature's contribution to cluster separation using geometry-consistent, non-parametric effect sizes:
• Kruskal-Wallis H-Statistic & Epsilon-Squared (ε²): Quantifies rank-order separation of each feature across discovered clusters without assuming normality.
• Mutual Information with Cluster Labels ($I(X_j; C)$): Captures non-linear feature relevance when clusters are curved manifolds or multi-modal regimes.
• Within-Cluster Dispersion Reduction: Compares global MAD/IQR against cluster-conditioned MAD/IQR to identify which features anchor cluster cohesion versus noise.

4. Methods to Detect Cluster Shape in High Dimensions

Because 2D projections (UMAP/t-SNE) distort global distances and hallucinate clusters, cluster shape is diagnosed directly in the pruned, invariance-adjusted high-dimensional space using four quantitative probes: [1]
• Linear Collinearity & Stability Audit: Compute covariance condition number ($\kappa = \lambda_{\max}/\lambda_{\min}$) and sample-to-feature ratio ($N/D$) to detect stretched ellipsoidal axes and multi-lag redundancy. [1]
• Marginal Tail & Skewness Check: Evaluate sample skewness and excess kurtosis to detect heavy-tailed distortions that break linear covariance estimates. [1]
• Linear vs. Non-Linear Information Gap:
	• Compare empirical Mutual Information ($I_{\text{emp}}$) against theoretical Gaussian MI ($I_{\text{Gaussian}} = -\frac{1}{2}\ln(1 - \rho^2)$).
	• Audit squared correlations ($\text{Corr}(z_i^2, z_j^2)$) on linearly whitened residuals to detect higher-order volatility coupling. [1]
• Manifold Curvature & Topology Probes:
	• Local vs. Global Dimensionality: Compare global Participation Ratio against Levina-Bickel k-NN Local Intrinsic Dimensionality ($d_{\text{local}} \ll d_{\text{global}}$ indicates a curved manifold).
	• Geodesic-to-Euclidean Ratio: Compare shortest-path k-NN graph distance against straight-line Euclidean distance to detect folded manifolds where straight lines shortcut across empty space.
	• Algorithm Disagreement (ARI): Compare partitions from a convex centroid model (K-Means) against a connectivity/density model (Average-Linkage / HDBSCAN) using Adjusted Rand Index. [1]

5. Decision Matrix: Feature Reduction, Shape Inference, Metrics & Algorithms


Detected Geometry & Diagnostics	Recommended Transformations & Dimensionality Reduction	Appropriate Distance Metric	Suggested Clustering Method
1. Isotropic Spherical Blobs
(Low condition number, small MI gap, high ARI agreement)	• Non-parametric feature pruning
• Domain invariance transform
• StandardScaler (column z-score)	Standardized Euclidean Distance	K-Means or Spherical GMM
2. Tilted / Stretched Ellipsoids
(High condition number / lag collinearity, low MI gap, convex shape)	• Non-parametric feature pruning
• Domain invariance transform
• Copula / Quantile Transform (if skewed)
• Truncated PCA / Cholesky Whitening (90–95% variance; Ledoit-Wolf shrinkage if N/D < 30)	Mahalanobis Distance
(Euclidean in whitened innovation space)	Full-Covariance GMM or Whitened K-Means
3. Non-Convex / Unequal Density
(Low convex-vs-connectivity ARI, flat local curvature, varying density)	• Non-parametric feature pruning
• Domain invariance transform
• Regularized PCA Whitening (95% variance cap to prevent k-NN distance concentration)	Mutual Reachability Distance (density-adjusted)	HDBSCAN
4. Curved Manifolds / Loops
(High Gaussian MI gap, $d_{\text{local}} \ll d_{\text{global}}$, high Geodesic/Euclidean ratio)	• Non-parametric feature pruning
• Domain invariance transform
• Quantile Transform + Diffusion Maps / Spectral Embedding (n_components = ceil(d_local) + 1)	Geodesic / Diffusion Distance (k-NN graph metric)	Spectral Clustering or HDBSCAN on Diffusion Coordinates


6. Deliverable Reporting Schema (ClusterFeatureAnalysisReport)

The package must export both a human-readable summary and a machine-readable JSON/DataFrame artifact containing four sections:
1. Executive Summary & Pipeline Specification:
	• Input vs. retained dimensions ($D_{\text{raw}} \rightarrow D_{\text{pruned}} \rightarrow D_{\text{reduced}}$), enforced domain invariance, diagnosed cluster geometry, recommended distance metric, and selected clustering family.
	• The original screened feature dimension and the retained/dropped dimension for
	  every sequential stage.
	• A `screening_strategy` narrative describing each test, its threshold, scope,
	  purpose, and whether it is a pre-clustering drop rule or post-clustering
	  ranking/profiling operation.
2. Dropped Features Audit Table (dropped_features_report):
	• feature_name: Name of excluded column.
	• gate: Screening rule triggered (1_sparsity, 2_zero_spread, 3_spearman_redundancy).
	• statistic_name & value vs. threshold: Exact quantitative evidence (e.g., spearman_rho = 0.974 > 0.95).
	• retained_proxy: Surviving feature that absorbed its information (for redundancy drops).
	• rationale: Human-readable explanation of why keeping the column would distort distance geometry.
3. High-Dimensional Geometry & Redundancy Diagnostics (geometry_diagnostics):
	• Condition number (κ), skewness/kurtosis flags, Gaussian MI gap (Δ I), whitened squared residual correlation, global vs. local intrinsic dimensionality, geodesic P90 ratio, and convex-vs-connectivity ARI.
4. Cluster-Based Feature Selection & Separation Profile (retained_features_profile):
	• Ranked table of retained features showing Normalized Entropy ($H_{\text{norm}}$), Kruskal-Wallis $\epsilon^2$, Cluster Mutual Information $I(X_j; C)$, and cluster-level median/IQR profiles identifying which features drive each discovered cluster.

The executive summary and human-readable narrative must explicitly account for feature
attrition. At minimum they must distinguish and report the original source column
count, model feature count after reorientation/engineering, pruned feature count,
reduced feature count, total dropped count, dropped percentage, and the
number of drops attributed to each of the three screening gates. Every dropped
feature must remain traceable to its gate and quantitative rationale.

When the analysis includes reorientation, identifier exclusion, or derived-feature
construction, the report must provide a sequential `feature_accounting` table. Each
row must state the step, input feature count, output feature count, removed feature
count, action, and explanation. Columns converted into observations or compressed
into derived measures must not be mislabeled as selector drops; they must be
identified as reorientation or feature engineering actions.

7. Geometry-Matched Visualization & Rendering Constraints

Visualization is a diagnostic deliverable, not a replacement for the numerical
audit. It must use the diagnosed geometry to select the projection:

• Isotropic spheres: PCA on PC1/PC2.
• Stretched ellipsoids: whitened PCA scatter with per-cluster 95% covariance ellipses.
• Non-convex / varying density: UMAP with local-neighbor scale; if the optional
  `umap-learn` dependency is unavailable, use Spectral Embedding and identify the
  fallback in the output.
• Curved manifolds / loops: Spectral Embedding or Diffusion Map coordinates.
• Avoid t-SNE as a default diagnostic projection because it does not preserve
  global distance, cluster size, or trajectory continuity.

Rendering limits are mandatory:

• PCA and whitened-PCA projections are fitted on all rows, with no more than 5,000
  points rendered.
• Graph/manifold projections are fitted and rendered on no more than 3,000 rows.
• Pairwise distance matrices are limited to 800 rows.
• Sampling must be cluster-stratified, never a row-order prefix. Preserve cluster
  proportions and include at least `min(50, cluster_size)` observations from every
  cluster. HDBSCAN noise (`label == -1`) is treated as its own stratum.
• The projection subtitle must state the sampled and total row counts and
  percentage.

The package must generate a four-panel `SummaryFigure` with:

1. Feature hygiene waterfall and a top-10 retained-feature ranking by a combined,
   non-parametric importance score (Kruskal-Wallis ε² / cluster mutual information).
2. Geometry diagnostics for condition number, Gaussian MI gap, geodesic/euclidean
   P90 ratio, and convex-vs-connectivity ARI, with threshold reference lines.
3. The geometry-matched 2D projection on the prescribed stratified sample.
4. A cluster signature heatmap showing cluster-level median values of the leading
   retained features after robust within-feature rank-quantile normalization.

The machine-readable report must include visualization metadata (method, fitted
sample count, rendered sample count, total rows, sampling fraction, and output
paths). To avoid illegible labels and overcrowded axes, the four panels must be
exported as four standalone, full-size PNG files rather than one composite image:
`feature_hygiene_summary.png`, `geometry_diagnostics.png`,
`geometry_matched_projection.png`, and `cluster_signature_heatmap.png`.

Feature-removal percentages must always state their denominator. A percentage for
the profiler's derived-feature pruning step is not a percentage of the original
source columns. Each `feature_accounting` row must provide
`stage_removed_percentage = dropped_features / input_features * 100`, while the
executive summary must label `profiler_pruning_percentage` as a percentage of the
profiler input feature count.

If you'd like, I can now generate the Python implementation of the reporting & feature-selection modules (feature_selector.py + post-clustering feature_profiler.py) that produce this exact 4-part JSON and DataFrame report.
