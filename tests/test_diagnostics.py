import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from sklearn.datasets import make_circles

from cluster_profiler import (
    ClusterGeometryProfiler,
    InvarianceTransformer,
    build_pipeline,
    compute_algorithm_disagreement,
    compute_manifold_curvature,
    compute_stability_metrics,
    compute_tail_metrics,
    stratified_sample_indices,
)


def test_invariance_modes_are_row_wise():
    X = np.array([[3.0, 4.0], [0.0, 0.0]])

    assert np.allclose(InvarianceTransformer().fit_transform(X), X)
    assert np.allclose(
        InvarianceTransformer("translation").fit_transform(X),
        [[-0.5, 0.5], [0.0, 0.0]],
    )
    assert np.allclose(
        InvarianceTransformer("scale").fit_transform(X),
        [[0.6, 0.8], [0.0, 0.0]],
    )
    assert np.allclose(
        InvarianceTransformer("affine").fit_transform(X),
        [[-1.0, 1.0], [0.0, 0.0]],
    )


def test_invalid_invariance_and_nonfinite_input_raise():
    with pytest.raises(ValueError, match="Unknown invariance"):
        InvarianceTransformer("rotation").fit(np.ones((3, 2)))
    with pytest.raises(ValueError):
        ClusterGeometryProfiler().fit_audit([[1.0, 2.0], [np.nan, 3.0]])


def test_stability_and_tail_diagnostics():
    rng = np.random.default_rng(7)
    X = np.column_stack(
        [
            rng.normal(size=500),
            rng.lognormal(sigma=1.2, size=500),
        ]
    )
    correlated = np.column_stack([X[:, 0], X[:, 0] * 10.0 + rng.normal(scale=0.1, size=500)])

    stability = compute_stability_metrics(correlated)
    tails = compute_tail_metrics(X)

    assert stability["n_to_d_ratio"] == 250
    assert stability["condition_number"] > 100
    assert tails["max_abs_skewness"] > 2
    assert tails["max_excess_kurtosis"] > 7


def test_manifold_and_disagreement_metrics_detect_circles():
    X, _ = make_circles(
        n_samples=300, factor=0.35, noise=0.03, random_state=19
    )

    curvature = compute_manifold_curvature(X, random_state=19)
    disagreement = compute_algorithm_disagreement(
        X, k_test=2, random_state=19
    )

    assert curvature["geodesic_euclidean_p90_ratio"] > 1.1
    assert curvature["local_intrinsic_dimension"] < 2
    assert disagreement["convex_vs_connectivity_ari"] < 0.45


def test_translation_invariance_collapses_shifted_sine_waves():
    wave = np.sin(np.linspace(0.0, 2.0 * np.pi, 40))
    offsets = np.arange(12.0)[:, np.newaxis]
    X = np.tile(wave, (12, 1)) + offsets

    transformed = InvarianceTransformer("translation").fit_transform(X)

    assert np.linalg.norm(transformed - transformed[0], axis=1).max() < 1e-12


def test_isotropic_gaussian_blobs_predict_spheres():
    rng = np.random.default_rng(42)
    levels = np.repeat(np.arange(4.0), 100) * 5.0
    X = (levels + 0.2 * rng.normal(size=400))[:, np.newaxis]

    profiler = ClusterGeometryProfiler().fit_audit(X)

    assert profiler.report_["shape"] == "ISOTROPIC SPHERES"


def test_correlated_ar1_lag_matrix_is_pruned_before_geometry():
    rng = np.random.default_rng(42)
    n_samples, n_features = 480, 16
    indices = np.arange(n_features)
    covariance = 0.98 ** np.abs(np.subtract.outer(indices, indices))
    levels = np.repeat(np.arange(4.0), n_samples // 4) - 1.5
    innovations = rng.multivariate_normal(
        np.zeros(n_features), covariance, size=n_samples
    )
    X = levels[:, np.newaxis] * 10.0 + 0.1 * innovations

    profiler = ClusterGeometryProfiler().fit_audit(X)

    audit = profiler.report_["dropped_features_report"]
    assert any(row["gate"] == "3_spearman_redundancy" for row in audit)
    assert profiler.report_["executive_summary"]["pruned_dimensions"] < 16
    assert all(row["retained_proxy"] for row in audit
               if row["gate"] == "3_spearman_redundancy")


@pytest.mark.parametrize("dataset", ["circles", "roll"])
def test_curved_ground_truth_predicts_nonlinear_manifold(dataset):
    if dataset == "circles":
        from sklearn.datasets import make_circles

        X, _ = make_circles(
            n_samples=260, factor=0.35, noise=0.03, random_state=42
        )
    else:
        from sklearn.datasets import make_swiss_roll

        X, _ = make_swiss_roll(n_samples=320, noise=0.05, random_state=42)

    profiler = ClusterGeometryProfiler().fit_audit(X)

    assert profiler.report_["shape"] == "NON-LINEAR MANIFOLD"


def test_pipeline_is_executable_and_summary_is_readable():
    rng = np.random.default_rng(8)
    X = np.concatenate(
        [rng.normal((-2.0,), 0.2, size=(120, 1)),
         rng.normal((2.0,), 0.2, size=(120, 1))]
    )
    profiler = ClusterGeometryProfiler().fit_audit(X)
    pipeline = build_pipeline(profiler, n_clusters=2)

    labels = pipeline.fit_predict(X)

    assert labels.shape == (len(X),)
    assert set(pipeline.named_steps) >= {"invariance", "scale", "cluster"}
    assert "ISOTROPIC SPHERES" in profiler.summary()


def test_spectral_pipeline_is_executable():
    X, _ = make_circles(
        n_samples=180, factor=0.35, noise=0.04, random_state=7
    )
    profiler = ClusterGeometryProfiler(7).fit_audit(X)

    labels = build_pipeline(profiler, n_clusters=2).fit_predict(X)

    assert labels.shape == (180,)


def test_heavy_tails_enable_quantile_step():
    rng = np.random.default_rng(12)
    X = rng.lognormal(mean=0.0, sigma=2.0, size=(300, 3))

    profiler = ClusterGeometryProfiler().fit_audit(X)
    pipeline = build_pipeline(profiler, n_clusters=3)

    assert profiler.report_["recommendation"]["quantile_transform"] is True
    assert "distribution" in pipeline.named_steps


def test_nonparametric_feature_gates_and_budget():
    rng = np.random.default_rng(5)
    n = 120
    continuous = rng.normal(size=n)
    proxy = 10.0 * continuous + 1.0
    low_variance = np.ones(n)
    X = np.column_stack([
        continuous,
        proxy,
        low_variance,
        np.r_[np.zeros(119), 1.0],
        *[rng.normal(size=n) for _ in range(8)],
    ])
    names = [f"f{i}" for i in range(X.shape[1])]

    profiler = ClusterGeometryProfiler().fit_audit(
        X, feature_names=names
    )
    audit = profiler.report_["dropped_features_report"]

    assert {row["gate"] for row in audit} >= {
        "1_sparsity",
        "3_spearman_redundancy",
    }
    assert any(
        row["feature_name"] == "f1"
        and row["retained_proxy"] == "f0"
        for row in audit
    )
    stages = profiler.report_["executive_summary"]["screening_stages"]
    assert [stage["stage"] for stage in stages] == [1, 2, 3]
    assert all(
        previous["output_dimensions"] == current["input_dimensions"]
        for previous, current in zip(stages[:-1], stages[1:])
    )
    assert stages[0]["input_dimensions"] == 12
    assert stages[-1]["output_dimensions"] == (
        profiler.report_["executive_summary"]["pruned_dimensions"]
    )


def test_zero_spread_gate_preserves_valid_binary_flag():
    rng = np.random.default_rng(3)
    n = 200
    X = np.column_stack([
        np.r_[np.ones(198), 0.0, 1.0] * 2,
        np.r_[np.zeros(194), np.ones(6)],
        rng.normal(size=n),
    ])

    profiler = ClusterGeometryProfiler().fit_audit(X)
    audit = {
        row["feature_name"]: row
        for row in profiler.report_["dropped_features_report"]
    }

    assert audit["feature_0"]["gate"] == "1_sparsity"
    assert "feature_1" not in audit


def test_cluster_analysis_exports_four_part_report():
    rng = np.random.default_rng(4)
    X = np.concatenate(
        [rng.normal(-3.0, 0.2, size=(100, 3)),
         rng.normal(3.0, 0.2, size=(100, 3))]
    )
    profiler = ClusterGeometryProfiler().analyze_clusters(
        X, n_clusters=2, feature_names=["a", "b", "c"]
    )
    report = profiler.to_report_dict()

    assert set(report) == {
        "executive_summary",
        "dropped_features_report",
        "geometry_diagnostics",
        "retained_features_profile",
    }
    assert report["retained_features_profile"]
    assert all(
        row["feature_name"] in {"a", "b", "c"}
        and "epsilon_squared" in row
        and "cluster_mutual_information" in row
        and row["cluster_profiles"]
        for row in report["retained_features_profile"]
    )
    assert "retained_features_profile" in profiler.to_json()
    assert not profiler.report_frame("geometry_diagnostics").empty


def test_stratified_sampling_protects_rare_clusters():
    labels = np.concatenate([
        np.zeros(1_000, dtype=int),
        np.ones(30, dtype=int),
        np.full(10, -1, dtype=int),
    ])

    selected = stratified_sample_indices(
        labels, 100, minimum_per_cluster=50, random_state=4
    )
    sampled = labels[selected]

    assert len(selected) == 100
    assert np.sum(sampled == 1) == 30
    assert np.sum(sampled == -1) == 10


def test_summary_figure_reports_sampling_and_pruning(tmp_path):
    rng = np.random.default_rng(21)
    X = np.concatenate(
        [
            rng.normal(-3.0, 0.25, size=(180, 3)),
            rng.normal(3.0, 0.25, size=(180, 3)),
        ]
    )
    profiler = ClusterGeometryProfiler().analyze_clusters(X, n_clusters=2)
    destination = tmp_path / "summary.png"

    figures = profiler.create_summary_figure(X, output_path=tmp_path)
    metadata = profiler.report_["executive_summary"]["visualization"]

    assert len(figures) == 4
    assert all(figure.axes for figure in figures)
    assert all(Path(path).is_file() for path in metadata["output_paths"])
    assert metadata["total_rows"] == 360
    assert metadata["rendered_sample_count"] <= 3_000
    assert profiler.to_report_dict()["executive_summary"]["dropped_dimensions"] >= 0


def test_feature_accounting_preserves_source_dimensions():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(80, 4))
    profiler = ClusterGeometryProfiler().fit_audit(X)
    steps = [
        {
            "step": "source_schema",
            "input_features": 100,
            "output_features": 99,
            "dropped_features": 1,
            "action": "excluded_metadata",
        },
        {
            "step": "feature_engineering",
            "input_features": 99,
            "output_features": 4,
            "dropped_features": 0,
            "action": "derived",
        },
    ]

    profiler.set_feature_accounting(steps)

    summary = profiler.to_report_dict()["executive_summary"]
    assert summary["source_dimensions"] == 100
    assert summary["input_dimensions"] == 4
    assert summary["feature_accounting"][0]["stage_removed_percentage"] == 1.0
    assert summary["feature_accounting"][1]["stage_removed_percentage"] == 0.0


def test_filtered_dataset_export_contains_only_retained_features(tmp_path):
    rng = np.random.default_rng(6)
    X = pd.DataFrame(
        {
            "kept": rng.normal(size=120),
            "duplicate": None,
            "constant": 1.0,
        }
    )
    X["duplicate"] = X["kept"] + 0.01 * rng.normal(size=120)
    profiler = ClusterGeometryProfiler().analyze_clusters(X, n_clusters=2)
    output = tmp_path / "filtered.csv"

    destination = profiler.export_filtered_dataset(
        X, output, include_cluster=True
    )
    filtered = pd.read_csv(destination, index_col=0)
    drops = pd.DataFrame(profiler.report_["dropped_features_report"])

    assert destination == output
    assert list(filtered.columns) == ["kept", "cluster"]
    assert set(drops["feature_name"]) == {"duplicate", "constant"}
    assert profiler.report_["executive_summary"]["exports"]["filtered_dataset"][
        "columns"
    ] == 2


def test_public_client_notebook_api():
    rng = np.random.default_rng(17)
    X = pd.DataFrame({
        "first": rng.normal(size=100),
        "second": rng.normal(size=100),
    })
    profiler = ClusterGeometryProfiler(random_state=17).fit_audit(X)

    assert set(profiler.retained_feature_names) <= set(X.columns)
    assert len(profiler.screening_stages()) == 3
    assert profiler.labels is None
    assert profiler.visualization_metadata() is None

    profiler.analyze_clusters(X, n_clusters=2)
    assert profiler.labels.shape == (100,)
