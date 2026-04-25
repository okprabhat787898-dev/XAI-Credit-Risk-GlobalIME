"""Training script for XAI-RAS.

Run this module directly to train the model and save it to disk:

    python -m models.train

The trained pipeline is saved to ``models/artifacts/xai_ras_pipeline.joblib``.
"""

from __future__ import annotations

import logging
import pathlib

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from data.generator import ALL_FEATURES, TARGET_COLUMN, generate_loan_applications
from models.ensemble import build_model

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

ARTIFACTS_DIR = pathlib.Path(__file__).parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "xai_ras_pipeline.joblib"


def train(
    n_samples: int = 5000,
    test_size: float = 0.20,
    random_state: int = 42,
    save: bool = True,
) -> tuple:
    """Train the stacked ensemble and optionally persist it.

    Parameters
    ----------
    n_samples:
        Number of synthetic loan applications to generate.
    test_size:
        Fraction of data reserved for evaluation.
    random_state:
        Random seed.
    save:
        Whether to save the fitted pipeline to disk.

    Returns
    -------
    tuple
        ``(pipeline, metrics_dict)`` where ``metrics_dict`` contains
        ``roc_auc``, ``classification_report``, ``X_test``, and ``y_test``.
    """
    log.info("Generating %d synthetic loan applications …", n_samples)
    df = generate_loan_applications(n_samples=n_samples, random_state=random_state)

    X = df[ALL_FEATURES]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    log.info(
        "Train: %d rows | Test: %d rows | Default rate: %.1f%%",
        len(X_train),
        len(X_test),
        100 * y.mean(),
    )

    pipeline = build_model(random_state=random_state)
    log.info("Fitting stacked ensemble (RF + LightGBM) …")
    pipeline.fit(X_train, y_train)

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)
    roc_auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)

    log.info("ROC-AUC: %.4f", roc_auc)
    log.info("\n%s", classification_report(y_test, y_pred))

    if save:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)
        log.info("Pipeline saved to %s", MODEL_PATH)

    metrics = {
        "roc_auc": roc_auc,
        "classification_report": report,
        "X_test": X_test,
        "y_test": y_test,
    }
    return pipeline, metrics


def load_or_train(n_samples: int = 5000, random_state: int = 42) -> object:
    """Load the saved pipeline if available; otherwise train and save it.

    Parameters
    ----------
    n_samples:
        Number of samples to use when training from scratch.
    random_state:
        Random seed.

    Returns
    -------
    Fitted sklearn Pipeline.
    """
    if MODEL_PATH.exists():
        log.info("Loading saved pipeline from %s", MODEL_PATH)
        return joblib.load(MODEL_PATH)
    log.info("No saved model found – training from scratch.")
    pipeline, _ = train(n_samples=n_samples, random_state=random_state, save=True)
    return pipeline


if __name__ == "__main__":
    train()
