"""Geometry-matched due-diligence visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding


def stratified_sample_indices(
    labels: np.ndarray,
    max_samples: int,
    *,
    minimum_per_cluster: int = 50,
    random_state: int = 42,
) -> np.ndarray:
    """Sample each cluster proportionally while protecting rare regimes."""
    labels_checked = np.asarray(labels)
    if labels_checked.ndim != 1 or labels_checked.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional array.")
    if max_samples < 1:
        raise ValueError("max_samples must be positive.")
    if labels_checked.size <= max_samples:
        return np.arange(labels_checked.size)

    rng = np.random.default_rng(random_state)
    groups = {
        value.item() if isinstance(value, np.generic) else value:
        np.flatnonzero(labels_checked == value)
        for value in np.unique(labels_checked)
    }
    quotas = {
        label: min(size, minimum_per_cluster)
        for label, group in groups.items()
        for size in [len(group)]
    }
    remaining = max_samples - sum(quotas.values())
    if remaining < 0:
        # Scale protected quotas proportionally when there are many tiny strata.
        scale = max_samples / sum(quotas.values())
        quotas = {label: int(np.floor(size * scale)) for label, size in quotas.items()}
        while sum(quotas.values()) < max_samples:
            label = max(groups, key=lambda key: len(groups[key]) - quotas[key])
            quotas[label] += 1
        remaining = 0
    weights = np.array(
        [max(0, len(groups[label]) - quotas[label]) for label in groups],
        dtype=float,
    )
    if weights.sum() and remaining:
        raw_additions = remaining * weights / weights.sum()
        additions = np.floor(raw_additions).astype(int)
        for offset in np.argsort(-(raw_additions - additions))[
            : remaining - int(additions.sum())
        ]:
            additions[offset] += 1
        for label, addition in zip(groups, additions):
            quotas[label] += int(
                min(addition, len(groups[label]) - quotas[label])
            )
    selected: list[int] = []
    for label, group in groups.items():
        selected.extend(rng.choice(group, quotas[label], replace=False).tolist())
    return np.sort(np.asarray(selected, dtype=int))


def create_summary_figures(
    profiler,
    X: np.ndarray,
    labels: np.ndarray,
    output_dir: str | Path = "output",
) -> tuple[list[Any], dict[str, Any]]:
    """Generate four standalone due-diligence figures."""
    X_checked = np.asarray(X, dtype=float)
    labels_checked = np.asarray(labels)
    if X_checked.shape[0] != labels_checked.shape[0]:
        raise ValueError("X and labels must contain the same number of rows.")
    geometry = profiler.report_["shape"]
    linear_limit = 5000
    manifold_limit = 3000
    max_projection_samples = (
        linear_limit
        if geometry in {"ISOTROPIC SPHERES", "STRETCHED ELLIPSOIDS"}
        else manifold_limit
    )
    projection_indices = stratified_sample_indices(
        labels_checked,
        max_projection_samples,
        random_state=profiler.random_state,
    )
    prepared = _transform_before_embedding(profiler, X_checked)
    coordinates, method = _project(
        prepared,
        geometry,
        projection_indices,
        profiler.random_state,
        profiler.report_["metrics"]["local_intrinsic_dimension"],
    )
    sampled_labels = labels_checked[projection_indices]

    destination_dir = Path(output_dir)
    if destination_dir.suffix:
        destination_dir = destination_dir.parent
    destination_dir.mkdir(parents=True, exist_ok=True)

    hygiene_figure, hygiene_axes = plt.subplots(
        1,
        2,
        figsize=(15, 8),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )
    _draw_hygiene_panel(hygiene_axes, profiler.report_)
    hygiene_figure.suptitle(
        "Feature Hygiene and Sequential Accounting",
        fontsize=15,
        y=1.02,
    )
    hygiene_figure.tight_layout()

    diagnostics_figure, diagnostics_axis = plt.subplots(figsize=(11, 6))
    _draw_diagnostics_panel(
        diagnostics_axis, profiler.report_["geometry_diagnostics"]
    )
    diagnostics_figure.tight_layout()

    projection_figure, projection_axis = plt.subplots(figsize=(11, 8))
    _draw_projection_panel(
        projection_axis,
        coordinates,
        sampled_labels,
        geometry,
        method,
        len(projection_indices),
        len(labels_checked),
    )
    projection_figure.tight_layout()

    signature_figure, signature_axis = plt.subplots(figsize=(12, 7))
    _draw_signature_panel(
        signature_axis,
        profiler.feature_selector_.transform(X_checked),
        labels_checked,
        profiler.feature_selector_.retained_feature_names_.tolist(),
        profiler.report_["retained_features_profile"],
    )
    signature_figure.tight_layout()

    figures = [
        hygiene_figure,
        diagnostics_figure,
        projection_figure,
        signature_figure,
    ]
    names = [
        "feature_hygiene_summary.png",
        "geometry_diagnostics.png",
        "geometry_matched_projection.png",
        "cluster_signature_heatmap.png",
    ]
    paths = []
    for figure, name in zip(figures, names):
        path = destination_dir / name
        figure.savefig(path, dpi=200, bbox_inches="tight")
        paths.append(str(path))
    for figure in figures:
        plt.close(figure)
    metadata = {
        "method": method,
        "fitted_sample_count": (
            len(labels_checked)
            if geometry in {"ISOTROPIC SPHERES", "STRETCHED ELLIPSOIDS"}
            else len(projection_indices)
        ),
        "rendered_sample_count": len(projection_indices),
        "total_rows": len(labels_checked),
        "sampling_fraction": len(projection_indices) / len(labels_checked),
        "output_paths": paths,
    }
    return figures, metadata


def create_summary_figure(
    profiler,
    X: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path = "output/cluster_due_diligence_summary.png",
) -> tuple[list[Any], dict[str, Any]]:
    """Backward-compatible alias for generating standalone figures."""
    return create_summary_figures(profiler, X, labels, output_path)


def _transform_before_embedding(profiler, X: np.ndarray) -> np.ndarray:
    value = np.asarray(X, dtype=float)
    for name, step in profiler.pipeline_.steps:
        if name in {"embedding", "cluster"}:
            break
        value = step.transform(value)
    return np.asarray(value)


def _project(
    X: np.ndarray,
    geometry: str,
    indices: np.ndarray,
    random_state: int,
    local_id: float,
) -> tuple[np.ndarray, str]:
    if geometry == "ISOTROPIC SPHERES":
        projector = PCA(n_components=2, svd_solver="full")
        coordinates = projector.fit_transform(X)[indices]
        return coordinates, "PCA (PC1 vs PC2)"
    if geometry == "STRETCHED ELLIPSOIDS":
        projector = PCA(n_components=2, whiten=True, svd_solver="full")
        coordinates = projector.fit_transform(X)[indices]
        return coordinates, "Whitened PCA with 95% ellipses"

    sampled = X[indices]
    n_neighbors = max(5, min(int(np.ceil(local_id)) + 8, len(indices) - 1))
    if geometry == "NON-CONVEX / UNEQUAL DENSITY":
        try:
            from umap import UMAP

            projector = UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                random_state=random_state,
            )
            return projector.fit_transform(sampled), "UMAP"
        except ImportError:
            pass
    projector = SpectralEmbedding(
        n_components=2,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )
    suffix = " fallback" if geometry == "NON-CONVEX / UNEQUAL DENSITY" else ""
    return projector.fit_transform(sampled), f"Spectral Embedding{suffix}"


def _draw_hygiene_panel(axes, report: dict[str, Any]) -> None:
    axis, importance_axis = axes
    summary = report["executive_summary"]
    axis.bar(
        ["profiler\ninput", "pruned", "reduced"],
        [
            summary["input_dimensions"],
            summary["pruned_dimensions"],
            summary["reduced_dimensions"],
        ],
        color=["#5277a8", "#69a879", "#a68ac3"],
    )
    for index, value in enumerate(
        [
            summary["input_dimensions"],
            summary["pruned_dimensions"],
            summary["reduced_dimensions"],
        ]
    ):
        axis.text(index, value, f" {value}", ha="center", va="bottom", fontsize=9)
    axis.set_title(
        f"Profiler feature states: {summary['input_dimensions']} input → "
        f"{summary['pruned_dimensions']} pruned → "
        f"{summary['reduced_dimensions']} reduced"
    )
    axis.set_ylabel("Feature count")
    gate_lines = [
        f"{gate}: {count}"
        for gate, count in summary["pruning_gate_counts"].items()
    ]
    accounting = summary.get("feature_accounting", [])
    if accounting:
        gate_lines.append("")
        for row in accounting:
            gate_lines.append(
                f"{row['step']}: {row['input_features']}→"
                f"{row['output_features']} "
                f"({row['dropped_features']} removed, "
                f"{row.get('stage_removed_percentage', 0):.1f}% of input)"
            )
    axis.text(
        0.02,
        0.98,
        "\n".join(gate_lines),
        transform=axis.transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#dddddd"},
    )

    profile = report["retained_features_profile"]
    ranked = sorted(
        profile,
        key=lambda row: row["epsilon_squared"]
        + row["cluster_mutual_information"],
        reverse=True,
    )[:10]
    if ranked:
        scores = [
            row["epsilon_squared"] + row["cluster_mutual_information"]
            for row in ranked
        ]
        positions = np.arange(len(ranked))
        importance_axis.barh(
            positions,
            scores,
            color="#e0a458",
            alpha=0.45,
            height=0.35,
        )
        importance_axis.set_yticks(
            positions, [row["feature_name"] for row in ranked]
        )
        importance_axis.invert_yaxis()
        importance_axis.set_xlabel("epsilon squared + cluster MI")
        importance_axis.set_title("Top retained features")
        importance_axis.tick_params(axis="y", labelsize=10)
        importance_axis.grid(axis="x", alpha=0.2)
    else:
        importance_axis.axis("off")
        importance_axis.text(
            0.5, 0.5, "No post-clustering profile", ha="center", va="center"
        )


def _draw_diagnostics_panel(axis, metrics: dict[str, float]) -> None:
    specifications = [
        ("Condition number", metrics["condition_number"], 5.0, ">"),
        ("Gaussian MI gap", metrics["max_gaussian_mi_gap_nats"], 0.15, ">"),
        (
            "Geodesic / Euclidean",
            metrics["geodesic_euclidean_p90_ratio"],
            1.8,
            ">",
        ),
        (
            "Convex-connectivity ARI",
            metrics["convex_vs_connectivity_ari"],
            0.45,
            "<",
        ),
    ]
    positions = np.arange(len(specifications))
    ratios = [
        min(max(value / threshold, 0.0), 3.0)
        for _, value, threshold, _ in specifications
    ]
    colors = [
        "#c75b39" if (value > threshold) == (direction == ">") else "#6a9f58"
        for _, value, threshold, direction in specifications
    ]
    axis.barh(positions, ratios, color=colors)
    axis.axvline(1.0, color="#333333", linestyle="--", linewidth=1)
    axis.set_yticks(positions, [item[0] for item in specifications])
    axis.invert_yaxis()
    axis.set_xlabel("Value / diagnostic cutoff (capped at 3x)")
    axis.set_title("B  Geometry & curvature audit")
    for position, (_, value, threshold, direction) in enumerate(specifications):
        axis.text(
            min(ratios[position] + 0.03, 2.95),
            position,
            f"{value:.3g} {direction} {threshold:g}",
            va="center",
            fontsize=8,
        )


def _draw_projection_panel(
    axis,
    coordinates: np.ndarray,
    labels: np.ndarray,
    geometry: str,
    method: str,
    sampled_count: int,
    total_count: int,
) -> None:
    for label in np.unique(labels):
        mask = labels == label
        axis.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=8,
            alpha=0.55,
            label=f"cluster {label}",
            linewidths=0,
        )
    if geometry == "STRETCHED ELLIPSOIDS":
        for label in np.unique(labels):
            _add_covariance_ellipse(axis, coordinates[labels == label], label)
    axis.set_title(
        f"C  {method}\nStratified sample: {sampled_count:,} / "
        f"{total_count:,} rows ({100 * sampled_count / total_count:.1f}%)"
    )
    axis.set_xlabel("Component 1")
    axis.set_ylabel("Component 2")
    axis.legend(markerscale=2, fontsize=7, ncols=2)


def _add_covariance_ellipse(axis, values: np.ndarray, label: object) -> None:
    if values.shape[0] < 2:
        return
    covariance = np.cov(values, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2.0 * np.sqrt(np.maximum(eigenvalues, 0.0)) * np.sqrt(5.991)
    axis.add_patch(
        Ellipse(
            values.mean(axis=0),
            width=width,
            height=height,
            angle=angle,
            fill=False,
            linewidth=1.2,
        )
    )


def _draw_signature_panel(
    axis,
    X: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    profile: list[dict[str, Any]],
) -> None:
    ranked = sorted(
        profile,
        key=lambda row: row["epsilon_squared"]
        + row["cluster_mutual_information"],
        reverse=True,
    )[:10]
    selected = [feature_names.index(row["feature_name"]) for row in ranked]
    clusters = np.unique(labels)
    if not selected:
        axis.text(0.5, 0.5, "No retained features", ha="center", va="center")
        return
    quantiles = np.column_stack(
        [rankdata(X[:, index], method="average") / X.shape[0] for index in selected]
    )
    heatmap = np.vstack(
        [np.median(quantiles[labels == cluster], axis=0) for cluster in clusters]
    )
    image = axis.imshow(heatmap, aspect="auto", cmap="coolwarm", vmin=0, vmax=1)
    axis.set_yticks(range(len(clusters)), [str(value) for value in clusters])
    axis.set_xticks(
        range(len(ranked)),
        [row["feature_name"] for row in ranked],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    axis.set_xlabel("Top discriminative retained features")
    axis.set_ylabel("Cluster")
    axis.set_title("D  Cluster signature medians (rank quantiles)")
    plt.colorbar(image, ax=axis, label="Within-feature rank quantile")
