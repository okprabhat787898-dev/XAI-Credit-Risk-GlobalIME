"""SHAP-based explainability layer for XAI-RAS.

Wraps the fitted stacked ensemble in a SHAP TreeExplainer (applied to the
LightGBM base learner) to provide:

* Global feature importances (mean |SHAP|)
* Local per-application explanations
* Compliance summaries aligned with NRB AI Guidelines 2025
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import shap
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "shap is required.  Install it with: pip install shap"
    ) from exc
from sklearn.pipeline import Pipeline

from data.generator import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)

# ---------------------------------------------------------------------------
# Nepali translations for top features (customer-facing explanations)
# ---------------------------------------------------------------------------
FEATURE_LABELS_NP: dict[str, str] = {
    "credit_score": "क्रेडिट स्कोर",
    "loan_repayment_history": "ऋण भुक्तान इतिहास",
    "loan_to_income_ratio": "ऋण-आय अनुपात",
    "total_monthly_income": "कुल मासिक आय",
    "collateral_value": "धितोको मूल्य",
    "existing_loans": "विद्यमान ऋणहरू",
    "loan_amount": "ऋण रकम",
    "remittance_monthly_income": "रेमिट्यान्स मासिक आय",
    "cooperative_member": "सहकारी सदस्यता",
    "agricultural_annual_income": "कृषि वार्षिक आय",
    "seasonal_factor": "मौसमी कारक",
    "gold_holdings_tola": "सुन सञ्चय (तोला)",
    "land_area_ropani": "जग्गा क्षेत्रफल (रोपनी)",
    "cooperative_savings": "सहकारी बचत",
    "age": "उमेर",
    "dependents": "आश्रितहरू",
    "has_agricultural_income": "कृषि आय छ?",
    "receives_remittance": "रेमिट्यान्स प्राप्त?",
    "base_monthly_income": "आधार मासिक आय",
    "loan_tenure_months": "ऋण अवधि (महिना)",
    "application_month": "आवेदन महिना",
}


class SHAPExplainer:
    """Compute SHAP values for the LightGBM base learner inside the pipeline.

    Parameters
    ----------
    pipeline:
        Fitted sklearn Pipeline produced by :func:`models.ensemble.build_model`.
    background_data:
        Optional DataFrame used to build the SHAP background distribution.
        If ``None``, a small internal background is built during :meth:`fit`.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        background_data: pd.DataFrame | None = None,
    ) -> None:
        self.pipeline = pipeline
        self._explainer: shap.TreeExplainer | None = None
        self._feature_names: list[str] | None = None
        if background_data is not None:
            self.fit(background_data)

    def fit(self, X: pd.DataFrame) -> "SHAPExplainer":
        """Initialise the SHAP TreeExplainer using the preprocessed ``X``.

        Parameters
        ----------
        X:
            Raw feature DataFrame (same columns as training data).

        Returns
        -------
        self
        """
        preprocessor = self.pipeline.named_steps["preprocessor"]
        X_transformed = preprocessor.transform(X)
        self._feature_names = _get_feature_names(preprocessor)

        stacker = self.pipeline.named_steps["stacker"]
        lgbm_model = stacker.named_estimators_["lgbm"]

        self._explainer = shap.TreeExplainer(
            lgbm_model,
            data=X_transformed,
            model_output="probability",
            feature_perturbation="interventional",
        )
        return self

    def explain(self, X: pd.DataFrame) -> dict:
        """Return SHAP-based explanations for ``X``.

        Parameters
        ----------
        X:
            Raw feature DataFrame for one or more applications.

        Returns
        -------
        dict with keys:
            * ``shap_values``  : ndarray of shape (n_samples, n_features)
            * ``base_value``   : float – average model output
            * ``feature_names``: list[str]
            * ``X_transformed``: ndarray of transformed features
        """
        if self._explainer is None:
            raise RuntimeError(
                "Call SHAPExplainer.fit(background_data) before explain()."
            )
        preprocessor = self.pipeline.named_steps["preprocessor"]
        X_transformed = preprocessor.transform(X)

        shap_explanation = self._explainer(X_transformed)
        shap_values = shap_explanation.values
        base_value = float(np.mean(shap_explanation.base_values))

        # For binary classification SHAP may return (n, f, 2) – take class-1
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        return {
            "shap_values": shap_values,
            "base_value": base_value,
            "feature_names": self._feature_names,
            "X_transformed": X_transformed,
        }

    def global_importance(self, X: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
        """Compute global feature importance as mean |SHAP| across ``X``.

        Parameters
        ----------
        X:
            Raw feature DataFrame.
        top_n:
            Number of top features to return.

        Returns
        -------
        pd.DataFrame with columns ``feature``, ``mean_abs_shap``.
        """
        result = self.explain(X)
        mean_abs = np.abs(result["shap_values"]).mean(axis=0)
        df = pd.DataFrame(
            {"feature": result["feature_names"], "mean_abs_shap": mean_abs}
        ).sort_values("mean_abs_shap", ascending=False)
        return df.head(top_n).reset_index(drop=True)

    def local_explanation(
        self, X_row: pd.DataFrame, top_n: int = 5
    ) -> pd.DataFrame:
        """Return a ranked explanation for a single loan application.

        Parameters
        ----------
        X_row:
            Single-row DataFrame for the application.
        top_n:
            Number of features to surface in the explanation.

        Returns
        -------
        pd.DataFrame with columns ``feature``, ``shap_value``,
        ``direction`` (``'increases_risk'`` / ``'decreases_risk'``),
        ``feature_label_np`` (Nepali label).
        """
        result = self.explain(X_row)
        shap_vals = result["shap_values"][0]
        features = result["feature_names"]

        df = pd.DataFrame({"feature": features, "shap_value": shap_vals})
        df["abs_shap"] = df["shap_value"].abs()
        df = df.sort_values("abs_shap", ascending=False).head(top_n)

        df["direction"] = df["shap_value"].apply(
            lambda v: "increases_risk" if v > 0 else "decreases_risk"
        )
        df["feature_label_np"] = df["feature"].map(
            lambda f: FEATURE_LABELS_NP.get(_base_feature(f), f)
        )
        return df[
            ["feature", "feature_label_np", "shap_value", "direction"]
        ].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_feature_names(preprocessor) -> list[str]:
    """Extract feature names from a fitted ColumnTransformer."""
    names: list[str] = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(transformer, "get_feature_names_out"):
            out_names = transformer.get_feature_names_out()
        else:
            # Pipeline wrapping a transformer
            last_step = transformer.steps[-1][1]
            if hasattr(last_step, "get_feature_names_out"):
                out_names = last_step.get_feature_names_out(cols)
            else:
                out_names = cols
        names.extend(out_names)
    return names


def _base_feature(encoded_name: str) -> str:
    """Strip OHE prefix (e.g. ``'cat__district_Kathmandu'`` → ``'district'``)."""
    # ColumnTransformer adds "num__" / "cat__" prefixes after get_feature_names_out
    for prefix in ("num__", "cat__"):
        if encoded_name.startswith(prefix):
            encoded_name = encoded_name[len(prefix):]
            break
    # OHE appends "_<category>" – return the original column name
    for col in CATEGORICAL_FEATURES:
        if encoded_name.startswith(col + "_") or encoded_name == col:
            return col
    for col in NUMERIC_FEATURES:
        if encoded_name == col:
            return col
    return encoded_name
