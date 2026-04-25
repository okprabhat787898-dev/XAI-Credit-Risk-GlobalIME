"""XAI-RAS: Explainable AI Risk Assessment System – Streamlit Dashboard.

Run with:
    streamlit run app/main.py
"""

from __future__ import annotations

import sys
import pathlib

# Ensure the repo root is on sys.path so relative imports work when running
# with `streamlit run app/main.py` from any directory.
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from data.generator import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    LOAN_PURPOSES,
    NEPAL_DISTRICTS,
    NUMERIC_FEATURES,
    OCCUPATION_TYPES,
    REMITTANCE_COUNTRIES,
    generate_loan_applications,
)
from explainer.shap_explainer import SHAPExplainer
from models.train import load_or_train

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="XAI-RAS | Global IME Bank",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Training model (first run only) …")
def get_pipeline():
    return load_or_train(n_samples=3000)


@st.cache_resource(show_spinner="Fitting SHAP explainer …")
def get_explainer(_pipeline):
    background = generate_loan_applications(n_samples=200, random_state=0)
    expl = SHAPExplainer(_pipeline)
    expl.fit(background[ALL_FEATURES])
    return expl


@st.cache_data(show_spinner="Loading portfolio …")
def get_portfolio():
    return generate_loan_applications(n_samples=500, random_state=99)


# ---------------------------------------------------------------------------
# Sidebar – application input form
# ---------------------------------------------------------------------------

def sidebar_form() -> pd.DataFrame:
    """Render input widgets and return a single-row application DataFrame."""
    st.sidebar.header("📋 Loan Application")

    age = st.sidebar.slider("Age", 18, 65, 35)
    district = st.sidebar.selectbox("District", NEPAL_DISTRICTS)
    occupation = st.sidebar.selectbox("Occupation", OCCUPATION_TYPES)
    dependents = st.sidebar.slider("Dependents", 0, 8, 2)

    st.sidebar.markdown("---")
    base_monthly_income = st.sidebar.number_input(
        "Base Monthly Income (NPR)", 5_000, 500_000, 40_000, step=1_000
    )
    receives_remittance = st.sidebar.checkbox("Receives Remittance?", False)
    remittance_country = "None"
    remittance_monthly_income = 0.0
    if receives_remittance:
        remittance_country = st.sidebar.selectbox(
            "Remittance Country", REMITTANCE_COUNTRIES
        )
        remittance_monthly_income = st.sidebar.number_input(
            "Remittance Monthly Income (NPR)", 0, 200_000, 30_000, step=1_000
        )

    has_agricultural_income = st.sidebar.checkbox("Has Agricultural Income?", False)
    agricultural_annual_income = 0.0
    land_area_ropani = 0.0
    seasonal_factor = 0.3
    if has_agricultural_income:
        agricultural_annual_income = st.sidebar.number_input(
            "Agricultural Annual Income (NPR)", 0, 1_000_000, 100_000, step=5_000
        )
        land_area_ropani = st.sidebar.number_input(
            "Land Area (Ropani)", 0.0, 200.0, 5.0, step=0.5
        )
        application_month = st.sidebar.slider("Application Month", 1, 12, 6)
        high_months = {3, 4, 5, 6, 10, 11}
        seasonal_factor = 0.85 if application_month in high_months else 0.35
    else:
        application_month = st.sidebar.slider("Application Month", 1, 12, 6)

    st.sidebar.markdown("---")
    gold_holdings_tola = st.sidebar.number_input(
        "Gold Holdings (Tola)", 0.0, 100.0, 3.0, step=0.5
    )
    credit_score = st.sidebar.slider("Credit Score", 300, 900, 620)
    existing_loans = st.sidebar.slider("Existing Loans", 0, 5, 1)
    loan_repayment_history = st.sidebar.slider(
        "Repayment History (0=poor, 1=perfect)", 0.0, 1.0, 0.80, step=0.01
    )
    cooperative_member = st.sidebar.checkbox("Cooperative Member?", False)
    cooperative_savings = 0.0
    if cooperative_member:
        cooperative_savings = st.sidebar.number_input(
            "Cooperative Savings (NPR)", 0, 500_000, 25_000, step=1_000
        )

    st.sidebar.markdown("---")
    loan_amount = st.sidebar.number_input(
        "Loan Amount (NPR)", 10_000, 10_000_000, 500_000, step=10_000
    )
    loan_purpose = st.sidebar.selectbox("Loan Purpose", LOAN_PURPOSES)
    loan_tenure_months = st.sidebar.select_slider(
        "Loan Tenure (months)", options=[12, 24, 36, 48, 60, 84, 120], value=36
    )
    collateral_value = st.sidebar.number_input(
        "Collateral Value (NPR)", 0, 20_000_000, 800_000, step=10_000
    )

    total_monthly_income = (
        base_monthly_income
        + remittance_monthly_income
        + agricultural_annual_income / 12.0
    )
    loan_to_income_ratio = loan_amount / max(
        total_monthly_income * loan_tenure_months, 1
    )

    data = {
        "age": age,
        "district": district,
        "occupation": occupation,
        "dependents": dependents,
        "base_monthly_income": float(base_monthly_income),
        "receives_remittance": int(receives_remittance),
        "remittance_country": remittance_country,
        "remittance_monthly_income": float(remittance_monthly_income),
        "application_month": application_month,
        "seasonal_factor": seasonal_factor,
        "has_agricultural_income": int(has_agricultural_income),
        "agricultural_annual_income": float(agricultural_annual_income),
        "total_monthly_income": float(total_monthly_income),
        "land_area_ropani": float(land_area_ropani),
        "gold_holdings_tola": float(gold_holdings_tola),
        "credit_score": credit_score,
        "existing_loans": existing_loans,
        "loan_repayment_history": float(loan_repayment_history),
        "cooperative_member": int(cooperative_member),
        "cooperative_savings": float(cooperative_savings),
        "loan_amount": float(loan_amount),
        "loan_purpose": loan_purpose,
        "loan_tenure_months": loan_tenure_months,
        "collateral_value": float(collateral_value),
        "loan_to_income_ratio": loan_to_income_ratio,
    }
    return pd.DataFrame([data])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    pipeline = get_pipeline()
    explainer = get_explainer(pipeline)
    portfolio = get_portfolio()

    # Header
    st.title("🏦 XAI-RAS: Explainable AI Risk Assessment System")
    st.caption("Global IME Bank | AI/ML Hackathon 2026 | NRB AI Guidelines 2025 Compliant")

    application = sidebar_form()

    # Tab layout: officer view | customer view | portfolio | about
    tabs = st.tabs(
        ["🏦 Officer View", "🙋 Customer View (नेपाली)", "📊 Portfolio", "ℹ️ About"]
    )

    with tabs[0]:
        from app.components.officer_view import render_officer_view
        render_officer_view(
            application,
            pipeline,
            explainer,
            portfolio_df=portfolio[ALL_FEATURES],
        )

    with tabs[1]:
        from app.components.customer_view import render_customer_view
        render_customer_view(application, pipeline, explainer)

    with tabs[2]:
        st.subheader("📊 Portfolio Risk Analytics")
        from app.components.officer_view import render_portfolio_metrics
        render_portfolio_metrics(portfolio[ALL_FEATURES], pipeline)

        with st.expander("Sample Applications (first 20 rows)"):
            st.dataframe(portfolio.head(20))

    with tabs[3]:
        _render_about()


def _render_about() -> None:
    st.markdown(
        """
## ℹ️ About XAI-RAS

**XAI-RAS** (eXplainable AI Risk Assessment System) is an open-source machine-learning
framework for autonomous credit-risk assessment in the Nepalese banking sector.

### Key Features

| Feature | Details |
|---------|---------|
| **Model** | Stacked Ensemble: Random Forest + LightGBM meta-learner |
| **Explainability** | SHAP (SHapley Additive exPlanations) values |
| **Hyper-local features** | Remittance income, agricultural seasonality, cooperative membership, land (Ropani), gold (Tola) |
| **Compliance** | Nepal Rastra Bank AI Guidelines 2025 |
| **Languages** | English (technical) + Nepali (customer) |
| **Deployment** | Docker & Streamlit Cloud |

### Architecture

```
Input Features (25+)
       │
  ┌────▼────────────────────┐
  │  ColumnTransformer       │  ← StandardScaler + OHE
  └────┬────────────────────┘
       │
  ┌────▼────────────────────┐
  │  StackingClassifier     │
  │  ├─ RandomForest        │  ← Base learner 1
  │  └─ LightGBM            │  ← Base learner 2
  │  Meta: LogisticRegress. │
  └────┬────────────────────┘
       │
  ┌────▼────────────────────┐
  │  SHAPExplainer          │  ← SHAP TreeExplainer on LGBM
  └────┬────────────────────┘
       │
  ┌────▼────────────────────────────────────┐
  │  Dual Dashboard                         │
  │  ├─ Officer View (technical + SHAP)     │
  │  └─ Customer View (Nepali plain text)   │
  └─────────────────────────────────────────┘
```

### Tech Stack
- **Python 3.9** | scikit-learn | LightGBM | SHAP | Streamlit | Pandas | Plotly

### License
MIT License – © 2026 Global IME Bank Hackathon Contributors
        """
    )


if __name__ == "__main__":
    main()
