"""Pure high-dimensional cluster geometry diagnostics."""

from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
from scipy import stats
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import kneighbors_graph

_EPSILON = 1e-12


def compute_stability_metrics(X: np.ndarray) -> dict[str, float]:
    """Measure sample stability and the stretching of the correlation ellipsoid."""
    X_checked = _as_matrix(X)
    n_samples, n_features = X_checked.shape
    if n_features == 1:
        condition_number = 1.0
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            correlations = np.corrcoef(X_checked, rowvar=False)
        correlations = np.nan_to_num(correlations, nan=0.0, posinf=0.0, neginf=0.0)
        eigenvalues = np.linalg.eigvalsh(correlations)
        eigenvalues = np.maximum(eigenvalues, _EPSILON)
        condition_number = float(eigenvalues[-1] / eigenvalues[0])
    return {
        "n_to_d_ratio": float(n_samples / n_features),
        "condition_number": condition_number,
    }


def compute_tail_metrics(X: np.ndarray) -> dict[str, float]:
    """Return the strongest marginal skew and excess kurtosis."""
    X_checked = _as_matrix(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        skewness = stats.skew(X_checked, axis=0, bias=False, nan_policy="omit")
        kurtosis = stats.kurtosis(
            X_checked, axis=0, fisher=True, bias=False, nan_policy="omit"
        )
    return {
        "max_abs_skewness": _finite_max(np.abs(skewness)),
        "max_excess_kurtosis": _finite_max(kurtosis),
    }


def compute_nonlinear_coupling(
    X: np.ndarray,
    Z_whitened: np.ndarray,
    *,
    max_pairs: int = 50,
    random_state: int = 42,
) -> dict[str, float]:
    """Compare empirical and Gaussian information and detect volatility coupling."""
    X_checked = _as_matrix(X)
    Z_checked = _as_matrix(Z_whitened)
    if X_checked.shape[0] != Z_checked.shape[0]:
        raise ValueError("X and Z_whitened must contain the same number of samples.")
    if max_pairs < 1:
        raise ValueError("max_pairs must be at least 1.")

    rng = np.random.default_rng(random_state)
    row_indices = _subsample_indices(X_checked.shape[0], 1500, rng)
    X_sample = X_checked[row_indices]

    pairs = list(combinations(range(X_sample.shape[1]), 2))
    if len(pairs) > max_pairs:
        pairs = [pairs[index] for index in rng.choice(len(pairs), max_pairs, replace=False)]

    mi_gaps: list[float] = []
    n_neighbors = min(3, max(1, X_sample.shape[0] - 1))
    for first, second in pairs if X_sample.shape[0] > 1 else ():
        correlation = np.corrcoef(X_sample[:, first], X_sample[:, second])[0, 1]
        if not np.isfinite(correlation):
            correlation = 0.0
        gaussian_mi = -0.5 * np.log(max(_EPSILON, 1.0 - min(correlation**2, 1.0 - _EPSILON)))
        empirical_mi = mutual_info_regression(
            X_sample[:, [first]],
            X_sample[:, second],
            n_neighbors=n_neighbors,
            random_state=random_state,
        )[0]
        mi_gaps.append(float(empirical_mi - gaussian_mi))

    squared_coupling = 0.0
    if Z_checked.shape[1] > 1:
        with np.errstate(invalid="ignore", divide="ignore"):
            squared_correlations = np.corrcoef(
                np.square(Z_checked), rowvar=False
            )
        np.fill_diagonal(squared_correlations, 0.0)
        squared_coupling = _finite_max(np.abs(squared_correlations))

    return {
        "max_gaussian_mi_gap_nats": _finite_max(mi_gaps),
        "max_whitened_sq_corr": squared_coupling,
    }


def compute_manifold_curvature(
    Z_whitened: np.ndarray,
    n_neighbors: int = 12,
    *,
    random_state: int = 42,
) -> dict[str, float]:
    """Estimate global dimension, local dimension, and shortest-path curvature."""
    Z_checked = _as_matrix(Z_whitened)
    n_samples, n_features = Z_checked.shape
    if n_neighbors < 2:
        raise ValueError("n_neighbors must be at least 2.")
    if n_samples < 2:
        return {
            "global_participation_ratio": 1.0,
            "local_intrinsic_dimension": float(n_features),
            "geodesic_euclidean_p90_ratio": 1.0,
        }
    effective_neighbors = min(n_neighbors, n_samples - 1)

    centered = Z_checked - Z_checked.mean(axis=0)
    covariance = np.cov(centered, rowvar=False, ddof=1)
    eigenvalues = np.atleast_1d(np.linalg.eigvalsh(np.atleast_2d(covariance)))
    eigenvalues = np.maximum(eigenvalues, 0.0)
    eigenvalue_sum = eigenvalues.sum()
    if eigenvalue_sum <= _EPSILON:
        global_pr = 1.0
    else:
        global_pr = float(eigenvalue_sum**2 / np.square(eigenvalues).sum())

    if n_samples < 3:
        local_id = float(n_features)
        geodesic_ratio = 1.0
    else:
        distances = squareform(pdist(Z_checked, metric="euclidean"))
        np.fill_diagonal(distances, np.inf)
        nearest_indices = np.argpartition(
            distances, effective_neighbors - 1, axis=1
        )[:, :effective_neighbors]
        nearest_distances = np.take_along_axis(distances, nearest_indices, axis=1)
        nearest_distances.sort(axis=1)
        denominators = np.sum(
            np.log(
                nearest_distances[:, [-1]]
                / np.maximum(nearest_distances[:, :-1], _EPSILON)
            ),
            axis=1,
        )
        local_estimates = (effective_neighbors - 1) / np.maximum(
            denominators, _EPSILON
        )
        local_id = float(np.median(local_estimates[np.isfinite(local_estimates)]))

        rng = np.random.default_rng(random_state)
        sample_indices = _subsample_indices(n_samples, 800, rng)
        graph_space = Z_checked[sample_indices]
        graph_neighbors = min(effective_neighbors, graph_space.shape[0] - 1)
        graph = kneighbors_graph(
            graph_space,
            n_neighbors=graph_neighbors,
            mode="distance",
            include_self=False,
        )
        graph = graph.maximum(graph.T)
        path_distances = shortest_path(graph, directed=False)
        ambient_distances = squareform(pdist(graph_space, metric="euclidean"))
        valid = (
            np.isfinite(path_distances)
            & (ambient_distances > _EPSILON)
            & (path_distances > 0.0)
        )
        if np.any(valid):
            ratios = path_distances[valid] / ambient_distances[valid]
            geodesic_ratio = float(np.percentile(ratios, 90))
        else:
            geodesic_ratio = 1.0

    return {
        "global_participation_ratio": global_pr,
        "local_intrinsic_dimension": local_id,
        "geodesic_euclidean_p90_ratio": geodesic_ratio,
    }


def compute_algorithm_disagreement(
    Z_whitened: np.ndarray,
    k_test: int = 4,
    random_state: int = 42,
) -> dict[str, float]:
    """Compare convex centroid and connectivity-based cluster partitions."""
    Z_checked = _as_matrix(Z_whitened)
    if k_test < 1:
        raise ValueError("k_test must be at least 1.")
    cluster_count = min(k_test, Z_checked.shape[0])
    if cluster_count < 2:
        return {"convex_vs_connectivity_ari": 1.0}

    kmeans_labels = KMeans(
        n_clusters=cluster_count,
        n_init=10,
        random_state=random_state,
    ).fit_predict(Z_checked)
    connectivity_labels = AgglomerativeClustering(
        n_clusters=cluster_count,
        linkage="average",
    ).fit_predict(Z_checked)
    return {
        "convex_vs_connectivity_ari": float(
            adjusted_rand_score(kmeans_labels, connectivity_labels)
        )
    }


def _as_matrix(X: np.ndarray) -> np.ndarray:
    matrix = np.asarray(X, dtype=float)
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError("Input must be a non-empty two-dimensional array.")
    if not np.isfinite(matrix).all():
        raise ValueError("Input contains NaN or Inf values.")
    return matrix


def _finite_max(values: object) -> float:
    array = np.asarray(values, dtype=float).ravel()
    finite = array[np.isfinite(array)]
    return float(np.max(finite)) if finite.size else 0.0


def _subsample_indices(
    n_samples: int, limit: int, rng: np.random.Generator
) -> np.ndarray:
    if n_samples <= limit:
        return np.arange(n_samples)
    return np.sort(rng.choice(n_samples, limit, replace=False))
