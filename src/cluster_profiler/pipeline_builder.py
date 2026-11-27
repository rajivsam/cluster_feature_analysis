"""Build executable clustering pipelines from an audit."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import clone
from sklearn.cluster import HDBSCAN, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted

from .invariance import InvarianceTransformer
from .profiler import ClusterGeometryProfiler


class SpectralEmbeddingTransformer(TransformerMixin, BaseEstimator):
    """Adapt spectral embedding to sklearn's transformer contract."""

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 12,
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        X_checked = check_array(X, dtype=float)
        self.n_features_in_ = X_checked.shape[1]
        self.X_fit_ = X_checked.copy()
        self.embedding_ = self._embed(X_checked)
        self.neighbor_model_ = NearestNeighbors(
            n_neighbors=min(self.n_neighbors, X_checked.shape[0])
        ).fit(X_checked)
        return self

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None, **kwargs):
        self.fit(X, y)
        return self.embedding_.copy()

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "embedding_")
        X_checked = check_array(X, dtype=float)
        distances, indices = self.neighbor_model_.kneighbors(X_checked)
        weights = 1.0 / np.maximum(distances, 1e-12)
        weights /= weights.sum(axis=1, keepdims=True)
        return np.einsum("ij,ijk->ik", weights, self.embedding_[indices])

    def _embed(self, X: np.ndarray) -> np.ndarray:
        if X.shape[0] <= self.n_components:
            raise ValueError("n_components must be smaller than the sample count.")
        embedder = SpectralEmbedding(
            n_components=self.n_components,
            n_neighbors=min(self.n_neighbors, X.shape[0] - 1),
            random_state=self.random_state,
        )
        return embedder.fit_transform(X)


def build_pipeline(
    profiler: ClusterGeometryProfiler, n_clusters: int = 4
) -> Pipeline:
    """Construct the pipeline corresponding to a fitted audit recommendation."""
    check_is_fitted(profiler, "report_")
    if n_clusters < 1:
        raise ValueError("n_clusters must be at least 1.")
    report = profiler.report_
    recommendation = report["recommendation"]
    steps = [
        ("feature_selection", clone(profiler.feature_selector_)),
        (
            "invariance",
            InvarianceTransformer(invariance=report["invariance"]),
        )
    ]

    if recommendation["quantile_transform"]:
        steps.append(
            (
                "distribution",
                QuantileTransformer(
                    output_distribution="normal",
                    n_quantiles=min(1000, report["n_samples"]),
                    random_state=profiler.random_state,
                ),
            )
        )
    else:
        steps.append(("scale", StandardScaler()))

    geometry = recommendation["geometry"]
    if geometry == "NON-LINEAR MANIFOLD":
        steps.append(
            (
                "embedding",
                SpectralEmbeddingTransformer(
                    n_components=report["executive_summary"]["reduced_dimensions"],
                    n_neighbors=12,
                    random_state=profiler.random_state,
                ),
            )
        )
        estimator = SpectralClustering(
            n_clusters=n_clusters,
            affinity="nearest_neighbors",
            n_neighbors=12,
            assign_labels="kmeans",
            random_state=profiler.random_state,
        )
    elif geometry == "NON-CONVEX / UNEQUAL DENSITY":
        steps.append(("embedding", PCA(n_components=0.95, whiten=True)))
        estimator = HDBSCAN()
    elif geometry == "STRETCHED ELLIPSOIDS":
        steps.append(("embedding", PCA(n_components=0.95, whiten=True)))
        estimator = GaussianMixture(
            n_components=n_clusters,
            covariance_type="full",
            random_state=profiler.random_state,
        )
    elif geometry == "ISOTROPIC SPHERES":
        estimator = KMeans(
            n_clusters=n_clusters,
            n_init=10,
            random_state=profiler.random_state,
        )
    else:
        raise ValueError(f"Unsupported profiled geometry: {geometry!r}.")

    steps.append(("cluster", estimator))
    return Pipeline(steps)
