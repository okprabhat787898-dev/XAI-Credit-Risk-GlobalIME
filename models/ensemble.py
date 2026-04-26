"""Stacked Ensemble model (Random Forest + LightGBM) for credit-risk prediction.

Architecture
------------
* Base learners  : RandomForestClassifier and LGBMClassifier
* Meta learner   : LogisticRegression (stacking)
* Preprocessing  : StandardScaler for numerics, OneHotEncoder for categoricals

The pipeline is sklearn-compatible and exposes ``fit``, ``predict``, and
``predict_proba`` methods.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from lightgbm import LGBMClassifier
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "lightgbm is required.  Install it with: pip install lightgbm"
    ) from exc

from data.generator import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def _build_preprocessor() -> ColumnTransformer:
    """Return a ColumnTransformer that handles numeric and categorical columns."""
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            )
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_model(
    rf_n_estimators: int = 200,
    rf_max_depth: int = 8,
    lgbm_n_estimators: int = 200,
    lgbm_learning_rate: float = 0.05,
    lgbm_num_leaves: int = 31,
    random_state: int = 42,
) -> Pipeline:
    """Build and return the full preprocessing + stacking pipeline.

    Parameters
    ----------
    rf_n_estimators:
        Number of trees in the Random Forest.
    rf_max_depth:
        Maximum depth of each tree.
    lgbm_n_estimators:
        Number of boosting rounds for LightGBM.
    lgbm_learning_rate:
        Learning rate for LightGBM.
    lgbm_num_leaves:
        Maximum number of leaves per tree in LightGBM.
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Untrained pipeline ready for ``fit``.
    """
    preprocessor = _build_preprocessor()

    rf = RandomForestClassifier(
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    lgbm = LGBMClassifier(
        n_estimators=lgbm_n_estimators,
        learning_rate=lgbm_learning_rate,
        num_leaves=lgbm_num_leaves,
        class_weight="balanced",
        random_state=random_state,
        verbose=-1,
    )

    meta_learner = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        random_state=random_state,
    )

    stacker = StackingClassifier(
        estimators=[("rf", rf), ("lgbm", lgbm)],
        final_estimator=meta_learner,
        cv=5,
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("stacker", stacker),
        ]
    )
    return pipeline


def get_feature_importances(
    pipeline: Pipeline,
    feature_names_out: list[str] | None = None,
) -> pd.DataFrame:
    """Extract feature importances from the Random Forest base learner.

    Parameters
    ----------
    pipeline:
        Fitted pipeline returned by :func:`build_model`.
    feature_names_out:
        Optional list of feature names after preprocessing.  If ``None``,
        generic names are used.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``feature`` and ``importance``, sorted
        descending by importance.
    """
    stacker: StackingClassifier = pipeline.named_steps["stacker"]
    rf_model: RandomForestClassifier = stacker.named_estimators_["rf"]
    importances = rf_model.feature_importances_

    if feature_names_out is None:
        feature_names_out = [f"feature_{i}" for i in range(len(importances))]

    df = pd.DataFrame(
        {"feature": feature_names_out, "importance": importances}
    ).sort_values("importance", ascending=False)
    return df
