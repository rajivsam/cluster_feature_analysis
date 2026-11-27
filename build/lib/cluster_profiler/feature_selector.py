"""Distribution-free feature hygiene before geometry estimation."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class NonParametricFeatureSelector(TransformerMixin, BaseEstimator):
    """Prune degenerate, zero-spread, and redundant features."""

    def __init__(
        self,
        spearman_threshold: float = 0.95,
        mode_threshold: float = 0.98,
        entropy_threshold: float = 0.05,
        feature_names: list[str] | None = None,
    ) -> None:
        self.spearman_threshold = spearman_threshold
        self.mode_threshold = mode_threshold
        self.entropy_threshold = entropy_threshold
        self.feature_names = feature_names

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ):
        X_checked = check_array(
            X, dtype=float, ensure_2d=True, ensure_min_samples=2
        )
        if X_checked.shape[1] == 0:
            raise ValueError("X must contain at least one feature.")
        feature_names = feature_names if feature_names is not None else self.feature_names
        if feature_names is None:
            feature_names = [f"feature_{index}" for index in range(X_checked.shape[1])]
        if len(feature_names) != X_checked.shape[1]:
            raise ValueError("feature_names must match the number of X columns.")

        self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        self.n_features_in_ = X_checked.shape[1]
        self.feature_metrics_: dict[int, dict[str, float]] = {}
        self.dropped_features_: list[dict[str, Any]] = []
        self.screening_stages_: list[dict[str, Any]] = []
        dropped: set[int] = set()

        gate_input = X_checked.shape[1]
        for index in range(X_checked.shape[1]):
            metrics = _feature_metrics(X_checked[:, index])
            self.feature_metrics_[index] = metrics
            if metrics["unique_count"] <= 1:
                self._drop(index, "1_sparsity", "unique_count", 1.0, "> 1",
                           "A constant column carries no information and creates "
                           "singular covariance estimates.")
                dropped.add(index)
            elif metrics["mode_frequency"] > self.mode_threshold:
                self._drop(
                    index, "1_sparsity", "mode_frequency",
                    metrics["mode_frequency"], f"<= {self.mode_threshold}",
                    "The dominant value covers nearly all observations, so the "
                    "feature is effectively hyper-sparse.",
                )
                dropped.add(index)
            elif metrics["normalized_entropy"] < self.entropy_threshold:
                self._drop(
                    index, "1_sparsity", "normalized_entropy",
                    metrics["normalized_entropy"], f">= {self.entropy_threshold}",
                    "Empirical entropy is too low to support stable geometry.",
                )
                dropped.add(index)

        self._record_stage("1_sparsity", gate_input, dropped)
        gate_input = X_checked.shape[1] - len(dropped)
        for index in range(X_checked.shape[1]):
            if index in dropped:
                continue
            metrics = self.feature_metrics_[index]
            valid_binary = (
                metrics["unique_count"] == 2
                and metrics["minority_frequency"] >= 0.02
            )
            if metrics["iqr"] == 0.0 and metrics["mad"] == 0.0 and not valid_binary:
                self._drop(
                    index, "2_zero_spread", "iqr_and_mad", 0.0,
                    "IQR > 0 or MAD > 0",
                    "At least half the observations collapse to one value; "
                    "robust spread is zero.",
                )
                dropped.add(index)

        self._record_stage("2_zero_spread", gate_input, dropped)
        gate_input = X_checked.shape[1] - len(dropped)
        candidates = [i for i in range(X_checked.shape[1]) if i not in dropped]
        for position, first in enumerate(candidates):
            if first in dropped:
                continue
            for second in candidates[position + 1:]:
                if second in dropped:
                    continue
                rho = spearmanr(X_checked[:, first], X_checked[:, second]).statistic
                absolute_rho = float(abs(rho)) if np.isfinite(rho) else 0.0
                if absolute_rho <= self.spearman_threshold:
                    continue
                first_entropy = self.feature_metrics_[first]["normalized_entropy"]
                second_entropy = self.feature_metrics_[second]["normalized_entropy"]
                retained, redundant = (
                    (second, first)
                    if second_entropy > first_entropy
                    else (first, second)
                )
                self._drop(
                    redundant, "3_spearman_redundancy", "absolute_spearman_rho",
                    absolute_rho, f"<= {self.spearman_threshold}",
                    "Rank ordering is already represented by a higher-entropy "
                    "surviving proxy.",
                    retained_proxy=str(self.feature_names_in_[retained]),
                )
                dropped.add(redundant)
                if first == redundant:
                    break

        self._record_stage("3_spearman_redundancy", gate_input, dropped)
        for previous, current in zip(
            self.screening_stages_[:-1], self.screening_stages_[1:]
        ):
            if previous["output_dimensions"] != current["input_dimensions"]:
                raise RuntimeError("Screening stages are not sequential.")

        proxy_map = {
            row["feature_name"]: row["retained_proxy"]
            for row in self.dropped_features_
            if row["gate"] == "3_spearman_redundancy"
        }
        for row in self.dropped_features_:
            if row["gate"] != "3_spearman_redundancy":
                continue
            proxy = row["retained_proxy"]
            visited = {row["feature_name"]}
            while proxy in proxy_map and proxy not in visited:
                visited.add(proxy)
                proxy = proxy_map[proxy]
            row["retained_proxy"] = proxy

        self.retained_indices_ = np.asarray(
            [i for i in range(X_checked.shape[1]) if i not in dropped], dtype=int
        )
        if self.retained_indices_.size == 0:
            raise ValueError("Feature hygiene removed every feature.")
        self.retained_feature_names_ = self.feature_names_in_[self.retained_indices_]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "retained_indices_")
        X_checked = check_array(X, dtype=float, ensure_2d=True)
        if X_checked.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_checked.shape[1]} features, but "
                f"{self.__class__.__name__} is expecting "
                f"{self.n_features_in_} features as input."
            )
        return X_checked[:, self.retained_indices_]

    def get_feature_names_out(self, input_features: np.ndarray | None = None):
        check_is_fitted(self, "retained_feature_names_")
        return self.retained_feature_names_.copy()

    def _drop(
        self,
        index: int,
        gate: str,
        statistic_name: str,
        value: float,
        threshold: str,
        rationale: str,
        retained_proxy: str | None = None,
    ) -> None:
        self.dropped_features_.append(
            {
                "feature_name": str(self.feature_names_in_[index]),
                "gate": gate,
                "statistic_name": statistic_name,
                "value": float(value),
                "threshold": threshold,
                "retained_proxy": retained_proxy,
                "rationale": rationale,
            }
        )

    def _record_stage(
        self,
        gate: str,
        input_dimensions: int,
        cumulative_dropped: set[int],
    ) -> None:
        output_dimensions = self.n_features_in_ - len(cumulative_dropped)
        self.screening_stages_.append(
            {
                "stage": len(self.screening_stages_) + 1,
                "gate": gate,
                "input_dimensions": int(input_dimensions),
                "output_dimensions": int(output_dimensions),
                "dropped_dimensions": int(input_dimensions - output_dimensions),
                "dropped_percentage_of_input": (
                    100.0 * (input_dimensions - output_dimensions)
                    / input_dimensions
                    if input_dimensions
                    else 0.0
                ),
            }
        )


def _feature_metrics(column: np.ndarray) -> dict[str, float]:
    values, counts = np.unique(column, return_counts=True)
    probabilities = counts / counts.sum()
    unique_count = int(values.size)
    if unique_count <= 1:
        entropy = 0.0
    else:
        entropy = -np.sum(probabilities * np.log(probabilities)) / np.log(unique_count)
    sorted_counts = np.sort(counts)
    minority_frequency = (
        float(np.min(counts) / counts.sum()) if unique_count == 2 else 0.0
    )
    q25, median, q75 = np.percentile(column, [25, 50, 75])
    return {
        "unique_count": float(unique_count),
        "mode_frequency": float(np.max(counts) / counts.sum()),
        "normalized_entropy": float(entropy),
        "minority_frequency": minority_frequency,
        "iqr": float(q75 - q25),
        "mad": float(np.median(np.abs(column - median))),
        "sorted_min_count": float(sorted_counts[0]) if sorted_counts.size else 0.0,
    }
