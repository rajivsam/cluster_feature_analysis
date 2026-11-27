"""Audit orchestration and recommendation synthesis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted

from .diagnostics import (
    compute_algorithm_disagreement,
    compute_manifold_curvature,
    compute_nonlinear_coupling,
    compute_stability_metrics,
    compute_tail_metrics,
)
from .feature_profiler import profile_cluster_separation
from .feature_selector import NonParametricFeatureSelector
from .invariance import InvarianceTransformer


class ClusterGeometryProfiler(BaseEstimator):
    """Diagnose invariance and high-dimensional geometry before clustering."""

    def __init__(
        self,
        random_state: int = 42,
    ) -> None:
        self.random_state = random_state

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        invariance: str = "none",
    ) -> "ClusterGeometryProfiler":
        """Scikit-learn-compatible alias for :meth:`fit_audit`."""
        return self.fit_audit(X, invariance=invariance)

    def fit_audit(
        self,
        X: np.ndarray,
        invariance: str = "none",
        feature_names: list[str] | None = None,
    ) -> "ClusterGeometryProfiler":
        """Fit the audit and populate ``report_``."""
        if hasattr(X, "columns") and feature_names is None:
            feature_names = [str(column) for column in X.columns]
        X_checked = check_array(
            X,
            dtype=float,
            ensure_2d=True,
            ensure_min_samples=3,
            ensure_min_features=1,
        )
        self.invariance_ = invariance
        self.feature_selector_ = NonParametricFeatureSelector(
            feature_names=feature_names,
        ).fit(X_checked)
        pruned_X = self.feature_selector_.transform(X_checked)
        invariance_transformer = InvarianceTransformer(invariance=invariance)
        invariant_X = invariance_transformer.fit_transform(pruned_X)

        tail_metrics = compute_tail_metrics(invariant_X)
        needs_quantile = (
            tail_metrics["max_abs_skewness"] > 2.0
            or tail_metrics["max_excess_kurtosis"] > 7.0
        )
        if needs_quantile:
            distribution_transformer = QuantileTransformer(
                output_distribution="normal",
                n_quantiles=min(1000, X_checked.shape[0]),
                subsample=None,
                random_state=self.random_state,
            )
            probe_X = distribution_transformer.fit_transform(invariant_X)
        else:
            probe_X = invariant_X

        scaled_X = StandardScaler().fit_transform(probe_X)
        whitening_transformer = PCA(
            n_components=0.95,
            whiten=True,
            svd_solver="full",
            random_state=self.random_state,
        )
        Z_whitened = whitening_transformer.fit_transform(scaled_X)

        metrics = {
            **compute_stability_metrics(probe_X),
            **tail_metrics,
            **compute_nonlinear_coupling(
                probe_X, Z_whitened, random_state=self.random_state
            ),
            **compute_manifold_curvature(
                Z_whitened, random_state=self.random_state
            ),
            **compute_algorithm_disagreement(
                Z_whitened, random_state=self.random_state
            ),
        }
        recommendation = self._synthesize_recommendation(metrics, needs_quantile)
        self.n_features_in_ = X_checked.shape[1]
        reduced_dimension = (
            Z_whitened.shape[1]
            if recommendation["embedding"] != "none"
            else pruned_X.shape[1]
        )
        if recommendation["geometry"] == "NON-LINEAR MANIFOLD":
            reduced_dimension = int(
                min(
                    pruned_X.shape[1],
                    max(2, np.ceil(metrics["local_intrinsic_dimension"]) + 1),
                    X_checked.shape[0] - 1,
                )
            )
        self.report_: dict[str, Any] = {
            "shape": recommendation["geometry"],
            "invariance": invariance,
            "n_samples": X_checked.shape[0],
            "feature_names": self.feature_selector_.feature_names_in_.tolist(),
            "metrics": metrics,
            "recommendation": recommendation,
            "executive_summary": {
                "input_dimensions": X_checked.shape[1],
                "pruned_dimensions": pruned_X.shape[1],
                "reduced_dimensions": reduced_dimension,
                "dropped_dimensions": (
                    X_checked.shape[1] - pruned_X.shape[1]
                ),
                "profiler_pruning_percentage": 100.0
                * (X_checked.shape[1] - pruned_X.shape[1])
                / X_checked.shape[1],
                "pruning_gate_counts": _gate_counts(
                    self.feature_selector_.dropped_features_
                ),
                "screening_stages": list(
                    self.feature_selector_.screening_stages_
                ),
                "screened_dimensions": (
                    self.feature_selector_.screening_stages_[-1][
                        "output_dimensions"
                    ]
                ),
                "domain_invariance": invariance,
                "diagnosed_geometry": recommendation["geometry"],
                "distance_metric": recommendation["metric"],
                "clustering_family": recommendation["clustering"],
                "screening_strategy": {
                    "scope": "pre-geometry; independent of cluster labels",
                    "gates": [
                        {
                            "gate": "1_sparsity",
                            "test": "dominant mode frequency, unique count, normalized discrete entropy",
                            "drop_rule": "unique_count <= 1 or mode_frequency > 0.98 or normalized_entropy < 0.05",
                        },
                        {
                            "gate": "2_zero_spread",
                            "test": "IQR and median absolute deviation",
                            "drop_rule": "IQR = 0 and MAD = 0, except valid binary flags",
                        },
                        {
                            "gate": "3_spearman_redundancy",
                            "test": "pairwise absolute Spearman rank correlation",
                            "drop_rule": "|rho_s| > 0.95; retain higher-entropy proxy",
                        },
                    ],
                    "post_clustering_profiling": (
                        "Kruskal-Wallis epsilon squared, cluster mutual information, "
                        "and dispersion reduction rank retained features only."
                    ),
                },
            },
            "dropped_features_report": list(
                self.feature_selector_.dropped_features_
            ),
            "geometry_diagnostics": dict(metrics),
            "retained_features_profile": [],
        }
        self.retained_feature_names_ = (
            self.feature_selector_.retained_feature_names_.copy()
        )
        return self

    def analyze_clusters(
        self,
        X: np.ndarray,
        n_clusters: int = 4,
        labels: np.ndarray | None = None,
        invariance: str = "none",
        feature_names: list[str] | None = None,
    ) -> "ClusterGeometryProfiler":
        """Fit the recommended pipeline and add post-clustering separation metrics."""
        if not hasattr(self, "report_"):
            self.fit_audit(X, invariance=invariance, feature_names=feature_names)
        from .pipeline_builder import build_pipeline

        X_checked = check_array(X, dtype=float, ensure_2d=True)
        if labels is None:
            fitted_pipeline = build_pipeline(self, n_clusters=n_clusters)
            labels = fitted_pipeline.fit_predict(X_checked)
            self.pipeline_ = fitted_pipeline
        self.labels_ = np.asarray(labels)
        pruned_X = self.feature_selector_.transform(X_checked)
        profile = profile_cluster_separation(
            pruned_X,
            self.labels_,
            self.feature_selector_.retained_feature_names_.tolist(),
            random_state=self.random_state,
        )
        self.report_["retained_features_profile"] = profile[
            "retained_features_profile"
        ]
        return self

    def set_feature_accounting(
        self,
        steps: list[dict[str, Any]],
    ) -> "ClusterGeometryProfiler":
        """Attach source-to-pruned provenance without changing audit semantics."""
        check_is_fitted(self, "report_")
        required = {
            "step",
            "input_features",
            "output_features",
            "dropped_features",
            "action",
        }
        normalized: list[dict[str, Any]] = []
        for row in steps:
            missing = required - set(row)
            if missing:
                raise ValueError(
                    f"Feature accounting row is missing: {', '.join(sorted(missing))}"
                )
            normalized_row = dict(row)
            input_features = int(normalized_row["input_features"])
            normalized_row["stage_removed_percentage"] = (
                100.0 * int(normalized_row["dropped_features"]) / input_features
                if input_features
                else 0.0
            )
            normalized.append(normalized_row)
        if not normalized:
            raise ValueError("Feature accounting requires at least one step.")
        self.report_["executive_summary"]["source_dimensions"] = normalized[0][
            "input_features"
        ]
        self.report_["executive_summary"]["feature_accounting"] = normalized
        self.report_["feature_accounting"] = normalized
        return self

    def _synthesize_recommendation(
        self, metrics: dict[str, float], quantile_transform: bool
    ) -> dict[str, Any]:
        local_id = max(metrics["local_intrinsic_dimension"], 1e-12)
        curved = (
            metrics["geodesic_euclidean_p90_ratio"] > 1.8
            or metrics["global_participation_ratio"] / local_id > 2.2
            or metrics["max_gaussian_mi_gap_nats"] > 0.15
            or metrics["max_whitened_sq_corr"] > 0.15
        )
        if curved:
            recommendation = {
                "geometry": "NON-LINEAR MANIFOLD",
                "transform": "Quantile + Spectral/Diffusion Embedding",
                "metric": "Geodesic / Diffusion Distance",
                "clustering": "SpectralClustering or HDBSCAN",
                "embedding": "spectral",
            }
        elif metrics["convex_vs_connectivity_ari"] < 0.45:
            recommendation = {
                "geometry": "NON-CONVEX / UNEQUAL DENSITY",
                "transform": "Whitening",
                "metric": "Mutual Reachability Distance",
                "clustering": "HDBSCAN",
                "embedding": "pca",
            }
        elif metrics["condition_number"] > 5.0:
            recommendation = {
                "geometry": "STRETCHED ELLIPSOIDS",
                "transform": "PCA Whitening (or Ledoit-Wolf if N/D < 30)",
                "metric": "Mahalanobis Distance",
                "clustering": (
                    "GaussianMixture(covariance_type='full') or Whitened KMeans"
                ),
                "embedding": "pca",
            }
        else:
            recommendation = {
                "geometry": "ISOTROPIC SPHERES",
                "transform": "StandardScaler",
                "metric": "Standardized Euclidean",
                "clustering": "KMeans or Spherical GMM",
                "embedding": "none",
            }
        recommendation["quantile_transform"] = quantile_transform
        return recommendation

    def summary(self) -> str:
        """Return a concise Markdown audit report."""
        check_is_fitted(self, "report_")
        report = self.report_
        recommendation = report["recommendation"]
        executive = report["executive_summary"]
        screening_strategy = report["executive_summary"]["screening_strategy"]
        strategy_lines = "\n".join(
            f"- **{gate['gate']}**: {gate['test']}. "
            f"Drop rule: {gate['drop_rule']}."
            for gate in screening_strategy["gates"]
        )
        metric_lines = "\n".join(
            f"| {name} | {value:.6g} |"
            for name, value in report["metrics"].items()
        )
        return (
            "# Cluster Geometry Audit\n\n"
            f"- **Geometry:** {report['shape']}\n"
            f"- **Invariance:** {report['invariance']}\n"
            f"- **Transform:** {recommendation['transform']}\n"
            f"- **Metric:** {recommendation['metric']}\n"
            f"- **Clustering:** {recommendation['clustering']}\n\n"
            "## Feature Accounting\n\n"
            f"- **Profiler input:** {executive['input_dimensions']} -> "
            f"{executive['pruned_dimensions']} pruned -> "
            f"{executive['reduced_dimensions']} reduced\n"
            f"- **Source dimensions:** "
            f"{executive.get('source_dimensions', executive['input_dimensions'])}\n"
            f"- **Profiler input dropped:** {executive['dropped_dimensions']} "
            f"of {executive['input_dimensions']} derived features "
            f"({executive['profiler_pruning_percentage']:.2f}%)\n"
            f"- **Gate counts:** {executive['pruning_gate_counts']}\n\n"
            + _format_feature_accounting(
                executive.get("screening_stages", [])
            )
            + _format_feature_accounting(executive.get("feature_accounting", []))
            +
            "## Screening and Testing Strategy\n\n"
            f"{strategy_lines}\n"
            f"- **Post-clustering:** "
            f"{screening_strategy['post_clustering_profiling']}\n\n"
            "| Diagnostic | Value |\n"
            "|---|---:|\n"
            f"{metric_lines}\n"
            "\n## Dropped Features\n\n"
            f"{pd.DataFrame(report['dropped_features_report']).to_string(index=False)}\n"
            "\n## Retained Feature Separation Profile\n\n"
            f"{pd.DataFrame(report['retained_features_profile']).to_string(index=False)}\n"
        )

    def to_report_dict(self) -> dict[str, Any]:
        """Return the required four-section machine-readable report."""
        check_is_fitted(self, "report_")
        return {
            "executive_summary": self.report_["executive_summary"],
            "dropped_features_report": self.report_["dropped_features_report"],
            "geometry_diagnostics": self.report_["geometry_diagnostics"],
            "retained_features_profile": self.report_["retained_features_profile"],
        }

    @property
    def retained_feature_names(self) -> list[str]:
        """Return the original feature names retained by pre-geometry screening."""
        check_is_fitted(self, "retained_feature_names_")
        return [str(name) for name in self.retained_feature_names_]

    @property
    def labels(self) -> np.ndarray | None:
        """Return fitted cluster labels without requiring estimator internals."""
        if not hasattr(self, "labels_"):
            return None
        return self.labels_.copy()

    def screening_stages(self) -> list[dict[str, Any]]:
        """Return a defensive copy of the sequential feature-screening stages."""
        check_is_fitted(self, "report_")
        return [
            dict(stage)
            for stage in self.report_["executive_summary"]["screening_stages"]
        ]

    def visualization_metadata(self) -> dict[str, Any] | None:
        """Return registered visualization metadata, if figures were created."""
        check_is_fitted(self, "report_")
        value = self.report_["executive_summary"].get("visualization")
        return dict(value) if value is not None else None

    def filtered_dataset(
        self,
        X: np.ndarray,
        *,
        include_cluster: bool = False,
    ) -> pd.DataFrame:
        """Return untransformed observations with only retained feature columns."""
        check_is_fitted(self, "feature_selector_")
        if isinstance(X, pd.DataFrame):
            index = X.index
            X_values = X.to_numpy(dtype=float)
        else:
            index = pd.RangeIndex(len(X))
            X_values = np.asarray(X, dtype=float)
        filtered = self.feature_selector_.transform(X_values)
        frame = pd.DataFrame(
            filtered,
            index=index,
            columns=self.feature_selector_.retained_feature_names_,
        )
        if include_cluster:
            if not hasattr(self, "labels_") or len(self.labels_) != len(frame):
                raise ValueError("Matching fitted cluster labels are required.")
            frame.insert(len(frame.columns), "cluster", self.labels_)
        return frame

    def export_filtered_dataset(
        self,
        X: np.ndarray,
        output_path: str | Path,
        *,
        include_cluster: bool = False,
        index: bool = True,
    ) -> Path:
        """Export a CSV containing retained columns and optional cluster labels."""
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame = self.filtered_dataset(X, include_cluster=include_cluster)
        frame.to_csv(destination, index=index)
        self.report_.setdefault("exports", {})["filtered_dataset"] = {
            "path": str(destination),
            "rows": int(frame.shape[0]),
            "columns": int(frame.shape[1]),
            "retained_feature_names": frame.columns.tolist(),
            "includes_cluster": include_cluster,
        }
        self.report_["executive_summary"].setdefault("exports", {})[
            "filtered_dataset"
        ] = dict(self.report_["exports"]["filtered_dataset"])
        return destination

    def export_drop_report(
        self, output_path: str | Path
    ) -> Path:
        """Export the dropped-feature audit table."""
        check_is_fitted(self, "report_")
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.report_["dropped_features_report"]).to_csv(
            destination, index=False
        )
        self.report_.setdefault("exports", {})["dropped_features"] = {
            "path": str(destination),
            "rows": len(self.report_["dropped_features_report"]),
        }
        self.report_["executive_summary"].setdefault("exports", {})[
            "dropped_features"
        ] = dict(self.report_["exports"]["dropped_features"])
        return destination

    def create_summary_figure(
        self,
        X: np.ndarray,
        output_path: str | Path = "output",
    ):
        """Create four standalone, geometry-aware diagnostic figures."""
        check_is_fitted(self, "report_")
        if not hasattr(self, "pipeline_"):
            raise ValueError("analyze_clusters must be run before visualization.")
        from .visualization import create_summary_figures

        figures, metadata = create_summary_figures(
            profiler=self,
            X=X,
            labels=self.labels_,
            output_dir=output_path,
        )
        self.report_["executive_summary"]["visualization"] = metadata
        self.summary_figures_ = figures
        return figures

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the report to JSON."""
        return json.dumps(self.to_report_dict(), indent=indent, default=_json_default)

    def report_frame(self, section: str) -> pd.DataFrame:
        """Return a report section as a pandas DataFrame."""
        report = self.to_report_dict()
        if section not in report:
            raise ValueError(f"Unknown report section {section!r}.")
        value = report[section]
        if isinstance(value, dict):
            return pd.DataFrame([value])
        return pd.DataFrame(value)


def _gate_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "1_sparsity": 0,
        "2_zero_spread": 0,
        "3_spearman_redundancy": 0,
    }
    for row in rows:
        counts[row["gate"]] = counts.get(row["gate"], 0) + 1
    return counts


def _format_feature_accounting(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    frame = pd.DataFrame(rows)
    return (
        "### Source-to-analysis accounting\n\n"
        f"{frame.to_string(index=False)}\n\n"
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")
