"""Tests for the stacked ensemble model."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from data.generator import ALL_FEATURES, TARGET_COLUMN, generate_loan_applications
from models.ensemble import build_model, get_feature_importances


@pytest.fixture(scope="module")
def small_dataset():
    df = generate_loan_applications(n_samples=300, random_state=42)
    X = df[ALL_FEATURES]
    y = df[TARGET_COLUMN]
    return X, y


@pytest.fixture(scope="module")
def fitted_pipeline(small_dataset):
    X, y = small_dataset
    pipeline = build_model(
        rf_n_estimators=20,
        lgbm_n_estimators=20,
        random_state=42,
    )
    pipeline.fit(X, y)
    return pipeline


def test_build_model_returns_pipeline():
    pipeline = build_model()
    assert isinstance(pipeline, Pipeline)


def test_pipeline_has_expected_steps():
    pipeline = build_model()
    step_names = [name for name, _ in pipeline.steps]
    assert "preprocessor" in step_names
    assert "stacker" in step_names


def test_fit_predict(fitted_pipeline, small_dataset):
    X, y = small_dataset
    preds = fitted_pipeline.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})


def test_predict_proba_shape(fitted_pipeline, small_dataset):
    X, _ = small_dataset
    proba = fitted_pipeline.predict_proba(X)
    assert proba.shape == (len(X), 2)


def test_predict_proba_sums_to_one(fitted_pipeline, small_dataset):
    X, _ = small_dataset
    proba = fitted_pipeline.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_roc_auc_above_chance(fitted_pipeline, small_dataset):
    from sklearn.metrics import roc_auc_score
    X, y = small_dataset
    proba = fitted_pipeline.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, proba)
    assert auc > 0.55, f"ROC-AUC {auc:.3f} not meaningfully above 0.5"


def test_feature_importances(fitted_pipeline):
    imp = get_feature_importances(fitted_pipeline)
    assert isinstance(imp, pd.DataFrame)
    assert "feature" in imp.columns
    assert "importance" in imp.columns
    assert len(imp) > 0
    assert (imp["importance"] >= 0).all()


def test_predict_single_row(fitted_pipeline, small_dataset):
    X, _ = small_dataset
    single = X.iloc[[0]]
    pred = fitted_pipeline.predict(single)
    assert pred.shape == (1,)
    proba = fitted_pipeline.predict_proba(single)
    assert proba.shape == (1, 2)
