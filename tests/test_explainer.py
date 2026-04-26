"""Tests for the SHAP explainer."""

import numpy as np
import pandas as pd
import pytest

from data.generator import ALL_FEATURES, TARGET_COLUMN, generate_loan_applications
from explainer.shap_explainer import SHAPExplainer, FEATURE_LABELS_NP
from models.ensemble import build_model


@pytest.fixture(scope="module")
def fitted_setup():
    df = generate_loan_applications(n_samples=300, random_state=42)
    X = df[ALL_FEATURES]
    y = df[TARGET_COLUMN]
    pipeline = build_model(rf_n_estimators=20, lgbm_n_estimators=20, random_state=42)
    pipeline.fit(X, y)
    background = df.iloc[:50][ALL_FEATURES]
    explainer = SHAPExplainer(pipeline, background_data=background)
    return pipeline, explainer, X


def test_explainer_initialises(fitted_setup):
    _, explainer, _ = fitted_setup
    assert explainer._explainer is not None


def test_explain_returns_dict(fitted_setup):
    _, explainer, X = fitted_setup
    result = explainer.explain(X.iloc[:5])
    assert isinstance(result, dict)
    for key in ("shap_values", "base_value", "feature_names", "X_transformed"):
        assert key in result, f"Missing key: {key}"


def test_shap_values_shape(fitted_setup):
    _, explainer, X = fitted_setup
    n = 10
    result = explainer.explain(X.iloc[:n])
    assert result["shap_values"].shape[0] == n


def test_global_importance_dataframe(fitted_setup):
    _, explainer, X = fitted_setup
    imp = explainer.global_importance(X.iloc[:50], top_n=10)
    assert isinstance(imp, pd.DataFrame)
    assert len(imp) <= 10
    assert "feature" in imp.columns
    assert "mean_abs_shap" in imp.columns


def test_global_importance_sorted_descending(fitted_setup):
    _, explainer, X = fitted_setup
    imp = explainer.global_importance(X.iloc[:50], top_n=15)
    vals = imp["mean_abs_shap"].tolist()
    assert vals == sorted(vals, reverse=True)


def test_local_explanation_dataframe(fitted_setup):
    _, explainer, X = fitted_setup
    local = explainer.local_explanation(X.iloc[[0]], top_n=5)
    assert isinstance(local, pd.DataFrame)
    assert len(local) <= 5
    for col in ("feature", "feature_label_np", "shap_value", "direction"):
        assert col in local.columns


def test_local_explanation_direction_values(fitted_setup):
    _, explainer, X = fitted_setup
    local = explainer.local_explanation(X.iloc[[0]], top_n=5)
    valid_directions = {"increases_risk", "decreases_risk"}
    assert set(local["direction"].unique()).issubset(valid_directions)


def test_nepali_labels_not_empty():
    assert len(FEATURE_LABELS_NP) > 0
    for key, val in FEATURE_LABELS_NP.items():
        assert isinstance(key, str) and len(key) > 0
        assert isinstance(val, str) and len(val) > 0


def test_explainer_raises_before_fit():
    df = generate_loan_applications(n_samples=50, random_state=1)
    X = df[ALL_FEATURES]
    pipeline = build_model(rf_n_estimators=10, lgbm_n_estimators=10, random_state=1)
    pipeline.fit(X, df[TARGET_COLUMN])
    expl = SHAPExplainer(pipeline)  # no background_data → not yet fit
    with pytest.raises(RuntimeError, match="fit"):
        expl.explain(X.iloc[:2])
