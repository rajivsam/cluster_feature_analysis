"""Post-clustering non-parametric separation profiling."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import kruskal
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils.validation import check_array


def profile_cluster_separation(
    X: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str] | None = None,
    *,
    random_state: int = 42,
) -> dict[str, Any]:
    """Score retained features and summarize cluster-conditioned distributions."""
    X_checked = check_array(X, dtype=float, ensure_2d=True)
    labels_checked = np.asarray(labels)
    if labels_checked.ndim != 1 or labels_checked.shape[0] != X_checked.shape[0]:
        raise ValueError("labels must be one-dimensional and align with X.")
    if feature_names is None:
        feature_names = [f"feature_{index}" for index in range(X_checked.shape[1])]
    if len(feature_names) != X_checked.shape[1]:
        raise ValueError("feature_names must match the number of X columns.")
    clusters = np.unique(labels_checked)
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(feature_names):
        column = X_checked[:, index]
        groups = [column[labels_checked == cluster] for cluster in clusters]
        populated_groups = [group for group in groups if group.size]
        if len(populated_groups) >= 2:
            statistic = float(kruskal(*populated_groups, nan_policy="omit").statistic)
            degrees_of_freedom = len(populated_groups) - 1
            epsilon_squared = max(
                0.0,
                (statistic - degrees_of_freedom)
                / max(1.0, X_checked.shape[0] - len(populated_groups)),
            )
        else:
            statistic = 0.0
            epsilon_squared = 0.0
        mi = mutual_info_classif(
            column.reshape(-1, 1),
            labels_checked,
            discrete_features=False,
            random_state=random_state,
        )[0] if np.unique(labels_checked).size > 1 else 0.0
        global_median = float(np.median(column))
        global_iqr = float(np.percentile(column, 75) - np.percentile(column, 25))
        global_mad = float(np.median(np.abs(column - global_median)))
        conditioned_iqr = float(np.average(
            [np.percentile(group, 75) - np.percentile(group, 25)
             for group in populated_groups],
            weights=[group.size for group in populated_groups],
        ))
        conditioned_mad = float(np.average(
            [np.median(np.abs(group - np.median(group))) for group in populated_groups],
            weights=[group.size for group in populated_groups],
        ))
        dispersion_reduction = float(max(
            0.0,
            0.5 * (
                1.0 - conditioned_iqr / max(global_iqr, 1e-12)
                + 1.0 - conditioned_mad / max(global_mad, 1e-12)
            ),
        ))
        cluster_profiles = []
        for cluster, group in zip(clusters, groups):
            if not group.size:
                continue
            median = float(np.median(group))
            cluster_profiles.append(
                {
                    "cluster": _json_scalar(cluster),
                    "median": median,
                    "iqr": float(np.percentile(group, 75) - np.percentile(group, 25)),
                    "mad": float(np.median(np.abs(group - median))),
                }
            )
        values, counts = np.unique(column, return_counts=True)
        probabilities = counts / counts.sum()
        normalized_entropy = (
            0.0 if values.size <= 1
            else float(-np.sum(probabilities * np.log(probabilities)) / np.log(values.size))
        )
        rows.append(
            {
                "feature_name": str(name),
                "normalized_entropy": normalized_entropy,
                "kruskal_h": statistic,
                "epsilon_squared": epsilon_squared,
                "cluster_mutual_information": float(mi),
                "dispersion_reduction": dispersion_reduction,
                "global_median": global_median,
                "global_iqr": global_iqr,
                "global_mad": global_mad,
                "cluster_profiles": cluster_profiles,
            }
        )
    rows.sort(key=lambda row: row["epsilon_squared"], reverse=True)
    return {"retained_features_profile": rows}


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value
