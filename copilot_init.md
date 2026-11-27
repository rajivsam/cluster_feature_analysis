# Copilot Initialization Guide

Use this file when starting work in this repository.

## Project purpose

`cluster-geometry-profiler` performs non-parametric feature hygiene, cluster
geometry diagnosis, geometry-matched clustering, post-clustering feature
profiling, and auditable reporting.

## Important architecture

- `src/cluster_profiler/feature_selector.py`: three pre-geometry screening gates.
- `src/cluster_profiler/diagnostics.py`: geometry and manifold diagnostics.
- `src/cluster_profiler/profiler.py`: audit orchestration and report exports.
- `src/cluster_profiler/pipeline_builder.py`: recommended sklearn pipeline.
- `src/cluster_profiler/feature_profiler.py`: post-clustering importance.
- `src/cluster_profiler/visualization.py`: geometry-aware figures and sampling.
- `tests/test_diagnostics.py`: synthetic-regime and reporting regression tests.

## Requirements and data conventions

Read:

- `documents/cluster_feature_analysis_req.md`
- `documents/workspace_organization_requirements.md`

Do not describe Olist source columns as model features without accounting for
orientation and feature engineering. Reports must distinguish source columns,
weekly candidate features, derived profiler inputs, pruned features, and
embedding coordinates.

## Validation

Install and validate with:

```bash
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest
```

For notebook changes, execute the entire notebook and verify persisted report,
CSV, and image artifacts. Never report stale pre-execution notebook outputs.

## Reporting invariants

- Keep pre-clustering drops separate from post-clustering feature rankings.
- Every removed feature needs a gate, statistic, threshold, rationale, and proxy
  where applicable.
- Every percentage must state or encode its denominator.
- Prefer four standalone diagnostic figures over an overcrowded composite.
- Sampling must be cluster-stratified and respect the documented caps.
- Filtered dataset exports must use retained original feature values, not
  transformed or embedded coordinates, unless explicitly labeled otherwise.
