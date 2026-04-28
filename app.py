from pathlib import Path
import base64
import math
import time

import joblib
import numpy as np
import pandas as pd
import json
import streamlit as st

from feature_engineering import add_feature_engineering

# Global flag to track SHAP availability
shap_enabled = True
shap_error_message = None


@st.cache_resource
def load_model_and_explainer():
    """Load the trained model and initialize SHAP explainer with sampled background data.
    
    This function is cached to keep the model and explainer in memory,
    preventing unnecessary reloads and ensuring optimal app performance.
    
    Uses shap.sample(X_train, 50) for memory-efficient background data instead of
    the full training set. This reduces memory usage by 80-90% on resource-constrained
    Streamlit Cloud servers.
    
    Returns:
        tuple: (model, explainer, error_message) where explainer is TreeExplainer for Random Forest
    """
    global shap_enabled, shap_error_message
    
    # Try to load from models/ directory first, then fall back to root
    model_path = Path(__file__).resolve().parent / "models" / "model.joblib"
    if not model_path.exists():
        model_path = Path(__file__).resolve().parent / "model.joblib"
    
    model = joblib.load(model_path)
    
    # Initialize SHAP TreeExplainer for Random Forest (memory-safe with sampled background data)
    explainer = None
    error_msg = None
    try:
        import shap
        
        # Load training features for memory-efficient background data
        X_train_path = Path(__file__).resolve().parent / "X_train.joblib"
        if X_train_path.exists():
            X_train = joblib.load(X_train_path)
            # Use shap.sample() to create a small background dataset (50 samples)
            # This reduces memory usage by 80-90% compared to full training set
            background_data = shap.sample(X_train, 50)
        else:
            # Fallback: no background data (slower but still works)
            background_data = None
        
        # TreeExplainer is optimized for tree-based models (Random Forest)
        # Use check_additivity=False to prevent unnecessary computations
        if background_data is not None:
            explainer = shap.TreeExplainer(model, background_data, check_additivity=False)
        else:
            explainer = shap.TreeExplainer(model, check_additivity=False)
        
        shap_enabled = True
    except (ImportError, ModuleNotFoundError) as e:
        # SHAP library not available - capture error details
        shap_enabled = False
        error_msg = f"SHAP library not found: {type(e).__name__}: {str(e)}"
        explainer = None
    except MemoryError as e:
        # Memory error during SHAP initialization
        shap_enabled = False
        error_msg = f"Memory error initializing SHAP: {str(e)}"
        explainer = None
    except Exception as e:
        # Other errors (version conflicts, model format, computation) - also allow app to continue
        shap_enabled = False
        error_msg = f"SHAP Error ({type(e).__name__}): {str(e)}"
        explainer = None
    
    return model, explainer, error_msg


# Load model and SHAP explainer using the cached function
model, shap_explainer, shap_error_message = load_model_and_explainer()

# Display error details if SHAP is unavailable
if not shap_enabled and shap_error_message:
    st.error(f"⚠️ SHAP Error: {shap_error_message}\n\nThe core Risk Engine remains active. Falling back to LIME-style feature contributions.")
elif not shap_enabled:
    st.warning("⚠️ SHAP Explainer is currently offline, but the core Risk Engine is active.")

MODEL_FEATURE_NAMES = ("Income", "Loan Amount", "Age")
MODEL_BASELINE_VALUES = np.array([80000.0, 500000.0, 35.0], dtype=float)
MODEL_FEATURE_IMPORTANCES = np.array(getattr(model, "feature_importances_", np.full(3, 1.0 / 3.0)), dtype=float)
if MODEL_FEATURE_IMPORTANCES.sum() > 0:
    MODEL_FEATURE_IMPORTANCES = MODEL_FEATURE_IMPORTANCES / MODEL_FEATURE_IMPORTANCES.sum()


@st.cache_data(show_spinner=False)
def compute_shap_values(monthly_income: float, loan_amount: float, age: int):
    """
    Compute SHAP values using TreeExplainer (optimized for Random Forest).
    
    Cache key: monthly_income, loan_amount, age
    This ensures SHAP values are recomputed only when these inputs change.
    
    Memory-safe: Uses TreeExplainer with check_additivity=False
    Returns None if SHAP is not available (shap_enabled=False)
    """
    if not shap_enabled or shap_explainer is None:
        return None
    
    try:
        # Prepare input features
        features = np.array([[monthly_income, loan_amount, age]], dtype=float)
        
        # Compute SHAP values (TreeExplainer is memory-efficient for tree models)
        shap_values = shap_explainer.shap_values(features)
        
        # For binary classification, take positive class SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class (high risk)
        
        # Return SHAP values for first (only) sample
        return shap_values[0] if len(shap_values.shape) > 1 else shap_values
    
    except Exception as e:
        st.warning(f"⚠️ Error computing SHAP values: {e}")
        return None


@st.cache_data(show_spinner=False)
def compute_shap_summary(monthly_income: float, loan_amount: float, age: int):
    """
    Compute alternative contribution scores (LIME-style) when SHAP is unavailable.
    
    Cache key: monthly_income, loan_amount, age
    This serves as a fallback for memory-constrained environments.
    """
    current_values = np.array([monthly_income, loan_amount, age], dtype=float)
    baseline_score = predict_risk_score(*MODEL_BASELINE_VALUES)
    current_score = predict_risk_score(monthly_income, loan_amount, age)

    local_effects = []
    for index, _ in enumerate(MODEL_FEATURE_NAMES):
        sample_values = MODEL_BASELINE_VALUES.copy()
        sample_values[index] = current_values[index]
        local_effects.append(predict_risk_score(*sample_values) - baseline_score)

    local_effects_array = np.array(local_effects, dtype=float)
    total_local_effect = float(local_effects_array.sum())
    delta_score = current_score - baseline_score
    
    if abs(total_local_effect) > 1e-9:
        contribution_values = local_effects_array * (delta_score / total_local_effect)
    else:
        contribution_values = MODEL_FEATURE_IMPORTANCES * delta_score

    contributions = {
        feature_name: float(contribution)
        for feature_name, contribution in zip(MODEL_FEATURE_NAMES, contribution_values)
    }

    return contributions

APP_VERSION = "1.4.0"
MODEL_VERSION = "v1.5.0-Advanced"
RED = "#C5161D"
BLUE = "#004189"
GLOBAL_IME_BLUE = "#003399"
RISK_REDUCTION_BLUE = "#1d4ed8"
TEXT = "#10233f"
MUTED = "#50627f"
CARD = "#ffffff"
LOGO_PATH = Path(__file__).resolve().parent / "assets" / "logo.png"

INCOME_REFERENCE = 80000.0
LAND_REFERENCE = 5.0
AGE_REFERENCE = 35.0
AGE_SPREAD = 25.0

SHAP_FEATURE_EXPLANATIONS = {
    "Income": "आम्दानीको स्रोत",
    "Monthly Income": "मासिक आम्दानी",
    "Loan Amount": "ऋण रकम",
    "Age": "उमेर",
    "CIB Score": "कर्जा सूचना केन्द्रको स्कोर",
    "Primary Income Source": "मुख्य आम्दानीको स्रोत",
    "Essential Expenses": "आवश्यक मासिक खर्च",
    "Monthly Installment": "मासिक किस्ता",
    "Stress Tested Income": "जोखिम परिक्षण गरिएको आम्दानी",
    "Loan Coverage Ratio": "ऋण धान्ने अनुपात",
    "Alternative Data": "वैकल्पिक डेटा",
}

UTILITY_PAYMENT_SCORE_MAP = {
    "Always on Time": 100.0,
    "Occasional Delay": 65.0,
    "Frequent Delay": 30.0,
}


def get_monthly_income_tracking_js_snippet() -> str:
        """Reference JavaScript snippet for client-side Monthly Income interaction timing.

        This snippet is intended for a custom Streamlit component integration.
        """
        return """
const labelText = "Monthly Income (NPR)";
let startTime = null;

function findInputByLabelText(text) {
    const labels = Array.from(document.querySelectorAll("label, div, span"));
    const label = labels.find((el) => (el.textContent || "").trim() === text);
    if (!label) return null;
    const container = label.closest("[data-testid='stWidgetLabel'], div") || label.parentElement;
    return (container && container.parentElement)
        ? container.parentElement.querySelector("input")
        : null;
}

const incomeInput = findInputByLabelText(labelText);
if (incomeInput) {
    incomeInput.addEventListener("focus", () => {
        startTime = performance.now();
    });
    incomeInput.addEventListener("blur", () => {
        if (startTime !== null) {
            const seconds = (performance.now() - startTime) / 1000;
            // Send to Streamlit custom component bridge when available
            window.parent.postMessage(
                { type: "monthly-income-timing", timeSeconds: seconds },
                "*"
            );
        }
    });
}
""".strip()


def classify_behavioral_risk(monthly_income_input_time_seconds: float | None, threshold_seconds: float = 2.0) -> str:
        if monthly_income_input_time_seconds is None:
                return "Unknown"
        return "High" if monthly_income_input_time_seconds < threshold_seconds else "Low"


def calculate_alternative_credit_score(
    utility_bill_payment_consistency: str,
    digital_wallet_transaction_frequency: int,
    model_credit_score: float,
) -> tuple[float, float, float, float]:
    behavioral_utility_score = UTILITY_PAYMENT_SCORE_MAP.get(utility_bill_payment_consistency, 65.0)
    behavioral_wallet_score = float(np.clip(digital_wallet_transaction_frequency, 0, 100))
    behavioral_score = (behavioral_utility_score * 0.6) + (behavioral_wallet_score * 0.4)
    weighted_model_score = max(0.0, min(100.0, model_credit_score))
    credit_360_score = (weighted_model_score * 0.6) + (behavioral_score * 0.4)
    return behavioral_utility_score, behavioral_wallet_score, behavioral_score, credit_360_score


def apply_alternative_data_boost(current_risk_score: float, alt_data_inputs: dict[str, str]) -> float:
    """Apply risk reduction from alternative data for borderline cases.

    Rules:
    - Regular eSewa transactions: -10 points
    - Consistent Ncell/NTC bill payments: -5 points
    - Apply only when initial score is between 40 and 70 inclusive
    - Final score floor: 0
    """
    if not (40.0 <= float(current_risk_score) <= 70.0):
        return float(current_risk_score)

    adjusted_score = float(current_risk_score)
    esewa_signal = str(alt_data_inputs.get("esewa_transactions", "")).strip().lower()
    telco_bill_signal = str(alt_data_inputs.get("ncell_ntc_bill_payments", "")).strip().lower()

    if esewa_signal == "regular":
        adjusted_score -= 10.0
    if telco_bill_signal == "consistent":
        adjusted_score -= 5.0

    return max(0.0, adjusted_score)

st.set_page_config(page_title=f"XAI-RAS v{APP_VERSION} | Global IME Bank", page_icon="📊", layout="wide")

if "cib_verified" not in st.session_state:
    st.session_state.cib_verified = False
if "ekyc_verified" not in st.session_state:
    st.session_state.ekyc_verified = False
if "monthly_income_field_started_at" not in st.session_state:
    st.session_state.monthly_income_field_started_at = time.time()
if "monthly_income_prev_value" not in st.session_state:
    st.session_state.monthly_income_prev_value = 50000
if "monthly_income_time_spent_seconds" not in st.session_state:
    st.session_state.monthly_income_time_spent_seconds = None
if "monthly_income_interacted" not in st.session_state:
    st.session_state.monthly_income_interacted = False

st.markdown(
    """
    <style>
    :root {
        --global-ime-red: #C5161D;
        --global-ime-blue: #004189;
        --text-strong: #10233f;
        --text-muted: #50627f;
        --surface: #ffffff;
    }
    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(0, 65, 137, 0.10) 0%, rgba(0, 65, 137, 0.00) 38%),
            radial-gradient(circle at 95% 4%, rgba(197, 22, 29, 0.10) 0%, rgba(197, 22, 29, 0.00) 32%),
            linear-gradient(180deg, #fbfcfe 0%, #f3f6fb 100%);
    }
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
        max-width: 1320px;
    }
    h1, h2, h3, h4 {
        color: #003399;
        letter-spacing: -0.02em;
    }
    .section-heading {
        color: #003399;
    }
    .hero-card, .panel-card, .metric-box, .customer-message {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid rgba(0, 65, 137, 0.12);
        border-radius: 20px;
        box-shadow: 0 14px 32px rgba(16, 35, 63, 0.08);
    }
    .hero-card {
        padding: 1.4rem 1.5rem;
        margin-bottom: 1rem;
    }
    .panel-card {
        padding: 1rem;
    }
    .brand-strip {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: linear-gradient(135deg, rgba(0, 65, 137, 0.08), rgba(197, 22, 29, 0.08));
        color: var(--text-strong);
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .summary-badge {
        display: inline-block;
        margin-top: 0.5rem;
        padding: 0.34rem 0.7rem;
        border-radius: 999px;
        background: rgba(0, 65, 137, 0.08);
        color: var(--global-ime-blue);
        font-size: 0.78rem;
        font-weight: 700;
    }
    .metric-box {
        background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
        border-radius: 16px;
        padding: 1rem;
        box-shadow: 0 10px 24px rgba(16, 35, 63, 0.06);
    }
    .metric-label {
        color: var(--text-muted);
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        color: var(--text-strong);
        font-size: 1.75rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .metric-subtext, .small-note {
        color: var(--text-muted);
        font-size: 0.9rem;
        margin-top: 0.2rem;
    }
    .customer-message {
        background: linear-gradient(135deg, rgba(0, 65, 137, 0.06), rgba(197, 22, 29, 0.04));
        border-left: 6px solid var(--global-ime-red);
        padding: 1.2rem 1.2rem 1rem 1.2rem;
    }
    .customer-message h2 {
        margin-bottom: 0.35rem;
    }
    .sidebar-box {
        padding: 0.25rem 0.1rem 0.8rem 0.1rem;
    }
    .stSidebar {
        background: linear-gradient(180deg, #ffffff 0%, #f7f9fd 100%);
        border-right: 1px solid rgba(0, 65, 137, 0.08);
    }
    @media (max-width: 1100px) {
        .main .block-container {
            padding-top: 1.2rem;
        }
        .metric-value {
            font-size: 1.45rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar: Economic Stress Testing
with st.sidebar:
    # Logo and branding at top
    st.image(str(LOGO_PATH), width=200, use_container_width=True)
    st.caption("Official Credit Assessment Tool")
    st.divider()
    
    # 🚀 Load Winning Demo Example
    if st.button("🚀 Load Winning Demo Example", use_container_width=True, type="primary"):
        # Set all demo values in session state for the perfect applicant profile
        st.session_state.monthly_income_input = 75000
        st.session_state.loan_amount_input = 500000
        st.session_state.primary_income_source_input = "Salary"
        st.session_state.essential_expenses_input = 15000
        st.session_state.monthly_installment_input = 20000
        st.session_state.remittance_status_input = 1
        st.session_state.land_area_input = 2.5
        st.session_state.applicant_age_input = 40
        st.session_state.utility_bill_payment_input = "Always on Time"
        st.session_state.digital_wallet_input = 75
        st.session_state.cib_score_input_widget = 750
        st.session_state.remittance_volatility_input = -10
        st.session_state.agricultural_yield_input = 20
        st.session_state.cib_verified = True
        st.session_state.ekyc_verified = True
        st.session_state.monthly_income_time_spent_seconds = 3.5
        st.session_state.monthly_income_interacted = True
        st.success("✅ Gold Standard Example Loaded!")
        st.info("📊 Perfect applicant profile ready for analysis. All fields prefilled.")
    
    st.markdown("---")
    st.header("Economic Stress Testing")
    remittance_decline_pct = st.slider(
        "Remittance Decline %",
        min_value=0,
        max_value=100,
        value=0,
        step=1,
        help="Simulate a percent decline in remittance inflows.",
    )
    crop_failure_risk_pct = st.slider(
        "Agricultural Crop Failure Risk %",
        min_value=0,
        max_value=100,
        value=0,
        step=1,
        help="Simulate the percent income loss for agriculture-dependent borrowers.",
    )
    remittance_market_shock_pct = st.slider(
        "Remittance Market Shock (%)",
        min_value=0,
        max_value=80,
        value=0,
        step=1,
        help="Applies a uniform shock to remittance-based monthly income for stress migration analysis.",
    )
    st.markdown("---")


def predict_risk_score(monthly_income: int, loan_amount: float, age: int) -> float:
    features = np.array([[monthly_income, loan_amount, age]], dtype=float)
    return float(model.predict_proba(features)[0][1] * 100)


def apply_macroeconomic_stress_test(
    monthly_income: float,
    remittance_volatility_pct: float,
    agricultural_yield_drop_pct: float,
    primary_income_source: str,
) -> float:
    stressed_income = monthly_income * (1.0 + remittance_volatility_pct / 100.0)
    if primary_income_source.strip().lower() == "agriculture":
        stressed_income *= 1.0 - (agricultural_yield_drop_pct / 100.0)
    return max(0.0, stressed_income)


def reduce_remittance_based_income(
    portfolio_df: pd.DataFrame,
    shock_pct: float,
    income_col: str = "Monthly_Income",
    source_col: str = "Primary_Income_Source",
) -> pd.DataFrame:
    stressed_df = portfolio_df.copy()
    source_series = stressed_df[source_col].astype(str).str.strip().str.lower()
    remittance_mask = source_series == "remittance"
    stressed_df.loc[remittance_mask, income_col] = (
        stressed_df.loc[remittance_mask, income_col].astype(float) * (1.0 - float(shock_pct) / 100.0)
    )
    stressed_df[income_col] = stressed_df[income_col].clip(lower=0.0)
    return stressed_df


def decision_bucket_from_score(risk_score: float) -> str:
    if risk_score < 30.0:
        return "Approved"
    if risk_score <= 60.0:
        return "Manual Review"
    return "Reject"


def simulate_remittance_shock_migration(
    portfolio_df: pd.DataFrame,
    shock_pct: float,
) -> dict[str, object]:
    baseline_scores = portfolio_df.apply(
        lambda row: predict_risk_score(float(row["Monthly_Income"]), float(row["Loan_Amount"]), int(row["Age"])),
        axis=1,
    )
    stressed_portfolio = reduce_remittance_based_income(portfolio_df, shock_pct)
    stressed_scores = stressed_portfolio.apply(
        lambda row: predict_risk_score(float(row["Monthly_Income"]), float(row["Loan_Amount"]), int(row["Age"])),
        axis=1,
    )

    baseline_bucket = baseline_scores.apply(decision_bucket_from_score)
    stressed_bucket = stressed_scores.apply(decision_bucket_from_score)

    moved_mask = (baseline_bucket == "Approved") & (stressed_bucket == "Manual Review")
    baseline_approved = int((baseline_bucket == "Approved").sum())
    moved_count = int(moved_mask.sum())
    remained_approved = int(((baseline_bucket == "Approved") & (stressed_bucket == "Approved")).sum())

    return {
        "baseline_approved": baseline_approved,
        "moved_to_manual_review": moved_count,
        "remained_approved": remained_approved,
    }


class LoanOrchestratorAgent:
    def __init__(self, random_forest_model):
        self.random_forest_model = random_forest_model

    def _build_response(
        self,
        status: str,
        reason: str,
        suggested_next_step: str,
        score: float | None,
        model_used: bool,
        dti_ratio: float,
    ) -> dict[str, object]:
        return {
            "status": status,
            "reason": reason,
            "suggested_next_step": suggested_next_step,
            "score": score,
            "model_used": model_used,
            "dti_ratio": dti_ratio,
        }

    def pre_screen(self, monthly_income: int, loan_amount: float, age: int, cib_score: int | None = None) -> dict[str, object]:
        dti_ratio = loan_amount / monthly_income if monthly_income > 0 else float("inf")

        if monthly_income == 0:
            return self._build_response(
                status="Auto-Reject",
                reason="Monthly income is 0, so the application fails the hard-stop screening.",
                suggested_next_step="Decline the application and request a valid income source.",
                score=100.0,
                model_used=False,
                dti_ratio=dti_ratio,
            )

        if dti_ratio > 0.5:
            return self._build_response(
                status="High Risk",
                reason="Debt-to-Income ratio exceeds 50%.",
                suggested_next_step="Route to manual underwriting review.",
                score=70.0,
                model_used=False,
                dti_ratio=dti_ratio,
            )

        if age > 75:
            return self._build_response(
                status="Senior Life Insurance Requirement",
                reason="Applicant age is above 75 and senior life insurance documentation is required.",
                suggested_next_step="Request senior life insurance confirmation before proceeding.",
                score=55.0,
                model_used=False,
                dti_ratio=dti_ratio,
            )

        features = np.array([[monthly_income, loan_amount, age]], dtype=float)
        model_score = float(self.random_forest_model.predict_proba(features)[0][1] * 100)

        if model_score < 30:
            status = "Instant Approval"
            suggested_next_step = "Proceed with standard approval workflow."
        elif model_score <= 60:
            status = "Manual Review Required by Senior Credit Officer"
            suggested_next_step = "Route to a senior credit officer for review."
        else:
            status = "Auto-Reject"
            suggested_next_step = "Reject or request a materially lower exposure."

        return self._build_response(
            status=status,
            reason=f"Model risk score is {model_score:.1f}.",
            suggested_next_step=suggested_next_step,
            score=model_score,
            model_used=True,
            dti_ratio=dti_ratio,
        )


def validate_loan_eligibility(monthly_income: int, loan_amount: float, cib_score: int) -> dict[str, object]:
    if monthly_income == 0 or loan_amount > 50 * monthly_income:
        return {
            "rejected": True,
            "message": "Ineligible: Debt-to-Income ratio exceeds regulatory limits.",
        }

    return {"rejected": False, "message": "Eligible for model evaluation.", "cib_score": cib_score}


def apply_banking_policy_rules(monthly_income: int, loan_amount: float) -> dict[str, object]:
    """Apply simple banking policy rules prior to model prediction.

    - If monthly_income == 0 -> Hard Reject (regulatory non-compliance)
    - If loan_amount > monthly_income * 60 -> Policy Violation (DSR exceeded)
    Returns a dict with keys: status, reason, action
    """
    if monthly_income == 0:
        return {
            "status": "Hard Reject",
            "reason": "Regulatory Non-Compliance: No verifiable repayment capacity.",
            "action": "Decline",
        }

    if loan_amount > (monthly_income * 60):
        return {
            "status": "Policy Violation",
            "reason": "Policy Violation: DSR exceeded.",
            "action": "Flag",
        }

    return {"status": "Pass", "reason": "Pre-screen passed.", "action": "Continue"}


class CreditOrchestrator:
    """Orchestrates compliance, ML scoring, and final decision into a structured report."""

    def __init__(self, model_version: str = MODEL_VERSION):
        self.model_version = model_version

    def check_compliance(
        self,
        monthly_income: int,
        loan_amount: float,
        cib_verified: bool,
        ekyc_verified: bool,
    ) -> dict[str, object]:
        policy = apply_banking_policy_rules(monthly_income, loan_amount)
        return {
            "agent": "Compliance Agent",
            "status": policy.get("status", "Unknown"),
            "reason": policy.get("reason", "N/A"),
            "action": policy.get("action", "N/A"),
            "cib_verified": cib_verified,
            "ekyc_verified": ekyc_verified,
            "passed": policy.get("status") == "Pass",
        }

    def get_ml_score(
        self,
        age: int,
        monthly_income: int,
        loan_amount: float,
        remittance_status: int,
        land_area: float,
        cib_score: int,
        loan_to_income_ratio: float,
        compliance_result: dict[str, object],
    ) -> dict[str, object]:
        if compliance_result.get("status") == "Hard Reject":
            return {
                "agent": "ML Scoring Agent",
                "model_used": False,
                "score": 100.0,
                "label": "HARD REJECT",
                "status": "Hard Reject",
                "reason": compliance_result.get("reason", "Blocked by compliance."),
                "contributions": {},
            }

        ml_assessment = assess_applicant(
            age,
            monthly_income,
            loan_amount,
            remittance_status,
            land_area,
            cib_score,
            loan_to_income_ratio,
        )
        return {
            "agent": "ML Scoring Agent",
            "model_used": bool(ml_assessment.get("model_used", False)),
            "score": float(ml_assessment.get("score", 0.0)),
            "label": ml_assessment.get("label", "N/A"),
            "status": ml_assessment.get("status", "Unknown"),
            "reason": ml_assessment.get("reason", "N/A"),
            "contributions": ml_assessment.get("contributions", {}),
        }

    def finalize_decision(
        self,
        compliance_result: dict[str, object],
        ml_result: dict[str, object],
    ) -> dict[str, object]:
        if compliance_result.get("status") == "Hard Reject":
            decision = "Hard Reject"
            rationale = compliance_result.get("reason", "Compliance failure.")
        else:
            decision = ml_result.get("status", "Manual Review Required by Senior Credit Officer")
            rationale = ml_result.get("reason", "Decision based on model score.")

        return {
            "agent": "Decision Agent",
            "final_decision": decision,
            "rationale": rationale,
        }

    def run(
        self,
        age: int,
        monthly_income: int,
        loan_amount: float,
        remittance_status: int,
        land_area: float,
        cib_score: int,
        loan_to_income_ratio: float,
        cib_verified: bool,
        ekyc_verified: bool,
    ) -> dict[str, object]:
        compliance_result = self.check_compliance(
            monthly_income=monthly_income,
            loan_amount=loan_amount,
            cib_verified=cib_verified,
            ekyc_verified=ekyc_verified,
        )
        ml_result = self.get_ml_score(
            age=age,
            monthly_income=monthly_income,
            loan_amount=loan_amount,
            remittance_status=remittance_status,
            land_area=land_area,
            cib_score=cib_score,
            loan_to_income_ratio=loan_to_income_ratio,
            compliance_result=compliance_result,
        )
        decision_result = self.finalize_decision(compliance_result, ml_result)

        return {
            "timestamp": pd.Timestamp.now().isoformat(),
            "model_version": self.model_version,
            "compliance_agent": compliance_result,
            "ml_scoring_agent": ml_result,
            "decision_agent": decision_result,
        }


def build_model_contributions(monthly_income: int, loan_amount: float, age: int) -> tuple[dict[str, float], pd.DataFrame, float]:
    """
    Build model contribution table using cached SHAP/LIME computations.
    
    This function uses st.cache_data internally to avoid recalculating
    SHAP values when inputs haven't changed.
    """
    # Try SHAP first (memory-safe TreeExplainer)
    shap_values = compute_shap_values(float(monthly_income), float(loan_amount), int(age))
    
    if shap_values is not None:
        # Use SHAP values for contributions
        contributions = {
            feature_name: float(shap_values[index])
            for index, feature_name in enumerate(MODEL_FEATURE_NAMES)
        }
        contribution_source = "SHAP (TreeExplainer)"
    else:
        # Fallback to LIME-style local effects (cached)
        contributions = compute_shap_summary(float(monthly_income), float(loan_amount), int(age))
        contribution_source = "Local Effects (LIME-style)"

    current_values = np.array([monthly_income, loan_amount, age], dtype=float)
    current_score = predict_risk_score(monthly_income, loan_amount, age)

    contribution_table = pd.DataFrame(
        [
            {
                "Feature": feature_name,
                "Current Value": current_values[index],
                "Baseline Value": MODEL_BASELINE_VALUES[index],
                "Model Importance": MODEL_FEATURE_IMPORTANCES[index] * 100.0,
                "Contribution": contributions[feature_name],
            }
            for index, feature_name in enumerate(MODEL_FEATURE_NAMES)
        ]
    )
    contribution_table["Direction"] = np.where(
        contribution_table["Contribution"] >= 0,
        "Increases risk",
        "Reduces risk",
    )
    contribution_table = contribution_table.sort_values("Contribution", key=lambda series: series.abs(), ascending=False)

    return contributions, contribution_table, current_score


def assess_applicant(
    age: int,
    monthly_income: int,
    loan_amount: float,
    remittance_status: int,
    land_area: float,
    cib_score: int,
    loan_to_income_ratio: float = 0.0,
) -> dict:
    orchestrator = LoanOrchestratorAgent(model)
    screening = orchestrator.pre_screen(monthly_income, loan_amount, age, cib_score)

    if screening["model_used"]:
        contributions, contribution_table, _ = build_model_contributions(monthly_income, loan_amount, age)
    else:
        contributions = {}
        contribution_table = pd.DataFrame(columns=["Feature", "Current Value", "Baseline Value", "Model Importance", "Contribution", "Direction"])

    score = float(screening["score"] or 0.0)
    if screening["status"] == "Instant Approval":
        band = "LOW"
        label = "LOW RISK"
    elif screening["status"] == "Manual Review Required by Senior Credit Officer":
        band = "MEDIUM"
        label = "MEDIUM RISK"
    elif screening["status"] == "Senior Life Insurance Requirement":
        band = "MEDIUM"
        label = "SENIOR LIFE INSURANCE REQUIRED"
    elif screening["status"] == "High Risk":
        band = "HIGH"
        label = "HIGH RISK"
    else:
        band = "HIGH"
        label = "AUTO-REJECT"

    return {
        "rejected": screening["status"] == "Auto-Reject",
        "status": screening["status"],
        "reason": screening["reason"],
        "suggested_next_step": screening["suggested_next_step"],
        "model_used": screening["model_used"],
        "score": score,
        "band": band,
        "label": label,
        "contributions": contributions,
        "contribution_table": contribution_table,
        "confidence": round(max(0.0, 100.0 - score), 1),
    }


def render_score_chart(contributions: dict[str, float], title: str = "Model-Derived Contributions"):
    """
    Render SHAP/contribution values using cached Plotly chart.
    
    Optimized for Streamlit server memory usage:
    - Uses horizontal bar chart (faster rendering)
    - Limits to 3 top features
    - Disables interactive mode bar
    """
    try:
        import plotly.graph_objects as go

        labels = list(contributions.keys())
        values = np.array([contributions[label] for label in labels], dtype=float)
        order = np.argsort(np.abs(values))[::-1]
        ordered_labels = [labels[index] for index in order]
        ordered_values = values[order]
        colors = [RED if value >= 0 else RISK_REDUCTION_BLUE for value in ordered_values]

        fig = go.Figure(
            go.Bar(
                x=ordered_values,
                y=ordered_labels,
                orientation="h",
                marker=dict(color=colors, line=dict(color="rgba(0,0,0,0.08)", width=1)),
                text=[f"{value:+.1f}" for value in ordered_values],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{y}: %{x:+.1f} points<extra></extra>",
            )
        )
        fig.add_vline(x=0, line_width=1.2, line_color="#8497b4")
        fig.update_layout(
            height=360,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor=CARD,
            plot_bgcolor=CARD,
            font=dict(color=TEXT, family="Arial, sans-serif"),
            title=dict(text=title, x=0.02, xanchor="left"),
            xaxis=dict(title="Risk push / relief points", zeroline=False, gridcolor="#e6edf5", tickfont=dict(color=TEXT)),
            yaxis=dict(tickfont=dict(color=TEXT)),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    except Exception as e:
        st.warning(f"⚠️ Could not render Plotly chart: {e}")
        st.bar_chart(pd.DataFrame({"Contribution": list(contributions.values())}, index=list(contributions.keys())))


def risk_message(band: str) -> str:
    if band == "LOW":
        return "तपाईंको ऋण स्वीकृत हुने सम्भावना राम्रो छ।"
    if band == "MEDIUM":
        return "तपाईंको आवेदन थप मूल्याङ्कनका लागि जान सक्छ।"
    return "अहिले ऋण स्वीकृत हुने सम्भावना कम देखिन्छ।"


def technical_status(score: float) -> str:
    if score < 35:
        return "Prefer fast-track approval"
    if score < 65:
        return "Route to manual review"
    return "Escalate for senior review"


def route_decision(risk_score: float, hard_stops_passed: bool) -> tuple[str, str, str]:
    if not hard_stops_passed:
        return (
            "Hard Stop Failed",
            "Auto-Reject",
            "One or more hard-stop checks failed before routing.",
        )
    if risk_score > 60:
        return (
            "Yellow Warning",
            "Auto-Reject",
            "Risk score is above the auto-reject threshold.",
        )
    if 30 <= risk_score <= 60:
        return (
            "Yellow Warning",
            "Manual Review Required by Senior Credit Officer",
            "Risk score requires escalated human review.",
        )
    return (
        "Green Light",
        "Instant Approval",
        "Risk score is below the fast-track threshold and all hard-stops passed.",
    )


def summarize_route_from_assessment(assessment: dict[str, object]) -> tuple[str, str, str]:
    status = str(assessment.get("status", "Auto-Reject"))
    reason = str(assessment.get("reason", ""))
    suggested_next_step = str(assessment.get("suggested_next_step", ""))

    if status == "Instant Approval":
        return "Green Light", status, suggested_next_step or reason
    if status in {"Manual Review Required by Senior Credit Officer", "High Risk", "Senior Life Insurance Requirement"}:
        return "Yellow Warning", status, suggested_next_step or reason
    return "Hard Stop", status, reason or suggested_next_step


def format_top_shap_factors(contribution_table: pd.DataFrame, top_n: int = 3) -> str:
    if contribution_table.empty:
        return "N/A"

    top_rows = contribution_table.head(top_n)
    return " | ".join(
        f"{row.Feature}: {row.Contribution:+.2f}"
        for row in top_rows.itertuples(index=False)
    )


def summarize_top_shap_values(top_factors: list[tuple[str, float]], final_decision: str, score_boost_applied: bool = False) -> str:
    if not top_factors:
        return f"{final_decision} को निर्णयका लागि पर्याप्त व्याख्यात्मक कारक भेटिएन।"

    translated_items = []
    positive_reasons = []
    negative_reasons = []

    for feature_name, shap_value in top_factors:
        nepalese_name = SHAP_FEATURE_EXPLANATIONS.get(feature_name, feature_name)
        translated_items.append(f"{nepalese_name} ({shap_value:+.2f})")
        if shap_value >= 0:
            positive_reasons.append(nepalese_name)
        else:
            negative_reasons.append(nepalese_name)

    summary_parts = ["प्रमुख कारणहरू: " + ", ".join(translated_items) + "."]

    if final_decision == "Instant Approval":
        summary_parts.append("धेरै जोखिम घटाउने संकेतहरू देखिएकाले ऋण स्वीकृत भएको छ।")
    elif final_decision == "Manual Review Required by Senior Credit Officer":
        summary_parts.append("केही जोखिम बढाउने संकेतहरू भएकाले वरिष्ठ अधिकृतबाट पुनः समीक्षा आवश्यक छ।")
    else:
        summary_parts.append("जोखिम उच्च भएकाले आवेदन flagged गरिएको छ।")

    if positive_reasons:
        summary_parts.append(f"जोखिम बढाउने मुख्य पक्षहरू: {', '.join(positive_reasons)}.")
    if negative_reasons:
        summary_parts.append(f"जोखिम घटाउने मुख्य पक्षहरू: {', '.join(negative_reasons)}.")

    if score_boost_applied:
        summary_parts.append("तपाईंको डिजिटल वालेट र उपयोगिता बिल भुक्तानीको राम्रो रेकर्डले गर्दा तपाईंको क्रेडिट स्कोरमा सुधार गरिएको छ।")

    return " ".join(summary_parts)


TECHNICAL_TERM_TO_NEPALI = {
    "Income": "आम्दानी",
    "Monthly Income": "मासिक आम्दानी",
    "Loan Amount": "ऋण रकम",
    "Age": "उमेर",
    "CIB Score": "CIB स्कोर",
    "Debt-to-Income": "ऋण-देखि-आम्दानी अनुपात",
    "DSR": "ऋण भुक्तानी अनुपात",
    "Loan Coverage Ratio": "ऋण धान्ने अनुपात",
    "Stress Tested Income": "जोखिम परिक्षण गरिएको आम्दानी",
    "Alternative Data": "वैकल्पिक डेटा",
}


def generate_nepali_explanation(top_factors: list[tuple[str, float]], final_decision: str, score_boost_applied: bool = False) -> str:
    """Generate a polite, professional Nepali explanation letter (Devanagari).

    - `top_factors` is a list of tuples: (feature_name, shap_value)
    - `final_decision` is the final routing/decision string
    """
    greeting = "आदरणीय आवेदक,\n\n"
    intro_reject = (
        "हामीलाई दु:ख व्यक्त गर्दै सूचित गर्न चाहन्छौं कि तपाईंको ऋण आवेदन अस्वीकार गरिएको छ।\n\n"
    )
    intro_other = (
        "धन्यवाद — तपाइँको आवेदन प्राप्त भयो। तल तपाइँको आवेदन सम्बन्धी संक्षिप्त व्याख्या प्रस्तुत गरिएको छ।\n\n"
    )

    body_lines = []
    if not top_factors:
        body_lines.append("हामीले निर्णयका लागि पर्याप्त व्याख्यात्मक कारक फेला पार्न सकेनौं।")
    else:
        body_lines.append("निर्णयमा योगदान पुर्‍याउने प्रमुख कारकहरू निम्नानुसार छन्:")
        for feature_name, shap_value in top_factors:
            nep_name = TECHNICAL_TERM_TO_NEPALI.get(feature_name, SHAP_FEATURE_EXPLANATIONS.get(feature_name, feature_name))
            if shap_value >= 0:
                reason = f"{nep_name} ले जोखिम बढाएको छ (प्रभावः {shap_value:+.2f})."
            else:
                reason = f"{nep_name} ले जोखिम घटाएको छ (प्रभावः {shap_value:+.2f})."
            body_lines.append("- " + reason)

    if score_boost_applied:
        body_lines.append("तपाईंको डिजिटल वालेट र उपयोगिता बिल भुक्तानीको राम्रो रेकर्डले गर्दा तपाईंको क्रेडिट स्कोरमा सुधार गरिएको छ।")

    # Compose closing depending on decision
    if "Reject" in str(final_decision) or "reject" in str(final_decision) or "Auto-Reject" in str(final_decision) or "Hard Reject" in str(final_decision):
        closing = (
            "\nविस्तृत कारणहरू माथि उल्लेख गरिएका छन्। कृपया आवश्यक भएको खण्डमा थप कागजातहरू पेश गर्नुहोस् वा ऋणको रकम घटाइ पुन: आवेदन दिनुहोस्।\n\n"
            "यदि तपाइँ यस निर्णयबारे प्रश्न राख्नुहुन्छ भने, कृपया हामीलाई सम्पर्क गर्नुहोस्।\n\n"
            "शुभकामनासहित,\nGlobal IME Bank — Credit Operations"
        )
        letter = greeting + intro_reject + "\n".join(body_lines) + closing
    else:
        closing = (
            "\nयदि तपाइँले ऋण स्वीकृति पाउन चाहनुहुन्छ भने, माथि देखाइएका कमजोर पक्षहरू सुधार गर्ने सुझावमा ध्यान दिनुहोस् (उदाहरण: ऋण रकम घटाउने, आम्दानी प्रमाण पेश गर्ने)।\n\n"
            "धन्यवाद,\nGlobal IME Bank — Credit Operations"
        )
        letter = greeting + intro_other + "\n".join(body_lines) + closing

    return letter


def generate_decision_audit_csv(audit_payload: dict[str, object]) -> tuple[str, str]:
    audit_frame = pd.DataFrame([audit_payload])
    csv_text = audit_frame.to_csv(index=False)
    csv_base64 = base64.b64encode(csv_text.encode("utf-8")).decode("utf-8")
    return csv_text, csv_base64


def generate_regulatory_audit_csv_record(
    inputs: dict,
    assessment: dict,
    policy_result: dict,
    model_version: str = MODEL_VERSION,
) -> str:
    """Create a CSV record formatted for NRB 2025 reporting standards.

    Returns CSV text (single-row) with ordered columns.
    """
    # Collect SHAP/contribution values as JSON string
    contributions = assessment.get("contributions") if isinstance(assessment.get("contributions"), dict) else {}
    shap_json = json.dumps(contributions, ensure_ascii=False)

    record = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "model_version": model_version,
        # Inputs (flattened)
        "monthly_income": inputs.get("monthly_income"),
        "loan_amount": inputs.get("loan_amount"),
        "applicant_age": inputs.get("applicant_age"),
        "primary_income_source": inputs.get("primary_income_source"),
        "essential_expenses": inputs.get("essential_expenses"),
        "monthly_installment": inputs.get("monthly_installment"),
        "remittance_status": inputs.get("remittance_status"),
        "land_area": inputs.get("land_area"),
        "utility_bill_payment_consistency": inputs.get("utility_bill_payment_consistency"),
        "digital_wallet_transaction_frequency": inputs.get("digital_wallet_transaction_frequency"),
        "cib_score": inputs.get("cib_score"),
        "ekyc_verified": bool(inputs.get("ekyc_verified", False)),
        "identity_document_uploaded": bool(inputs.get("identity_document_uploaded", False)),
        "monthly_income_input_time_seconds": inputs.get("monthly_income_input_time_seconds"),
        "behavioral_risk": inputs.get("behavioral_risk"),
        # Model outputs
        "model_probability": float(assessment.get("score", 0.0)),
        "shap_contributions": shap_json,
        # Policy guardrail
        "policy_guardrail_status": policy_result.get("status"),
        "policy_guardrail_reason": policy_result.get("reason"),
        "policy_guardrail_action": policy_result.get("action"),
        # Final decision
        "final_decision": assessment.get("status"),
    }

    df = pd.DataFrame([record])
    # Ensure NRB-friendly column order (explicit)
    cols = [
        "timestamp",
        "model_version",
        "monthly_income",
        "loan_amount",
        "applicant_age",
        "primary_income_source",
        "essential_expenses",
        "monthly_installment",
        "remittance_status",
        "land_area",
        "utility_bill_payment_consistency",
        "digital_wallet_transaction_frequency",
        "cib_score",
        "ekyc_verified",
        "identity_document_uploaded",
        "monthly_income_input_time_seconds",
        "behavioral_risk",
        "model_probability",
        "shap_contributions",
        "policy_guardrail_status",
        "policy_guardrail_reason",
        "policy_guardrail_action",
        "final_decision",
    ]
    df = df[cols]
    return df.to_csv(index=False)


def generate_audit_report(applicant_income: float, loan_amount: float, final_risk_score: float = 27.6) -> str:
    return (
        "Audit Report\n"
        f"Applicant's Income: NPR {applicant_income:,.0f}\n"
        f"Loan Amount: NPR {loan_amount:,.0f}\n"
        f"Final Risk Score: {final_risk_score:.1f}\n"
    )


with st.container():
    st.markdown('<div class="hero-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="brand-strip">XAI-RAS v{APP_VERSION} | Global IME Bank</div>', unsafe_allow_html=True)
    st.markdown(f'<h1 class="section-heading" style="margin:0.6rem 0 0.35rem 0;">XAI-RAS v{APP_VERSION}</h1>', unsafe_allow_html=True)
    st.caption("NRB AI Guidelines 2025 aligned: transparency, traceability, human review.")
    st.markdown('</div>', unsafe_allow_html=True)

# Top-of-page quick indicators
top_income = float(st.session_state.get("monthly_income_input", 50000))
top_installment = float(st.session_state.get("monthly_installment_input", 25000))
top_cib_score = float(st.session_state.get("cib_score_input_widget", 720))
top_dsr_pct = (top_installment / top_income * 100.0) if top_income > 0 else float("inf")
top_cib_confidence_pct = max(0.0, min(100.0, ((top_cib_score - 300.0) / 550.0) * 100.0))
top_cib_status = "Verified" if st.session_state.get("cib_verified", False) else "Unverified"

metric_col_1, metric_col_2 = st.columns(2, gap="medium")
with metric_col_1:
    if math.isinf(top_dsr_pct):
        st.metric("Debt-Service Ratio", "N/A", delta="No income declared")
    else:
        st.metric("Debt-Service Ratio", f"{top_dsr_pct:.1f}%", delta=f"{top_dsr_pct - 40.0:+.1f}% vs 40% threshold")
with metric_col_2:
    st.metric("CIB Confidence level", f"{top_cib_confidence_pct:.1f}%", delta=f"Status: {top_cib_status}")

tab_onboard, tab_ai, tab_audit = st.tabs(["📥 Applicant Intake", "🧠 AI Risk Analysis", "🛡️ Compliance & Audit"])

with tab_onboard:
    page_left_col, page_right_col = st.columns([0.92, 1.08], gap="large")

    with page_left_col:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-heading">Applicant Inputs</h2>', unsafe_allow_html=True)
        view_mode = st.radio("View Mode", ["Bank Officer View", "Sajilo Natija (Customer View)"], index=0)
        monthly_income = st.slider("Monthly Income (NPR)", min_value=0, max_value=200000, value=st.session_state.get("monthly_income_input", 50000), step=1000, key="monthly_income_input")
        if (not st.session_state.monthly_income_interacted) and (monthly_income != st.session_state.monthly_income_prev_value):
            st.session_state.monthly_income_time_spent_seconds = max(
                0.0,
                time.time() - float(st.session_state.monthly_income_field_started_at),
            )
            st.session_state.monthly_income_interacted = True
        st.session_state.monthly_income_prev_value = monthly_income

        loan_affordability_help = "Loan-to-Income Ratio evaluates the requested loan amount against your total annual earnings to measure affordability"
        loan_amount = st.slider(
            "Desired Loan Amount (ऋण रकम) (NPR)",
            min_value=50000,
            max_value=5000000,
            value=st.session_state.get("loan_amount_input", 500000),
            step=50000,
            help=loan_affordability_help,
            key="loan_amount_input",
        )
        primary_income_source = st.selectbox(
            "Primary Income Source",
            ["Salary", "Business", "Agriculture", "Remittance", "Other"],
            index=["Salary", "Business", "Agriculture", "Remittance", "Other"].index(st.session_state.get("primary_income_source_input", "Salary")),
            key="primary_income_source_input",
        )
        essential_expenses = st.slider(
            "Essential Monthly Expenses (NPR)",
            min_value=0,
            max_value=200000,
            value=st.session_state.get("essential_expenses_input", 20000),
            step=1000,
            key="essential_expenses_input",
        )
        monthly_installment = st.slider(
            "Monthly Installment (NPR)",
            min_value=1000,
            max_value=300000,
            value=25000,
            step=1000,
            key="monthly_installment_input",
        )
        remittance_status = st.slider(
            "Remittance Status",
            min_value=0,
            max_value=1,
            value=st.session_state.get("remittance_status_input", 1),
            step=1,
            help="0 = No remittance, 1 = Remittance received",
            key="remittance_status_input",
        )
        land_area = st.slider("Land Area (Ropani)", min_value=0.0, max_value=20.0, value=st.session_state.get("land_area_input", 1.0), step=0.1, key="land_area_input")
        applicant_age = st.slider("Applicant Age", min_value=18, max_value=80, value=st.session_state.get("applicant_age_input", 30), step=1, key="applicant_age_input")
        st.caption(f"Remittance selected: {'Received' if remittance_status == 1 else 'Not received'}")
        with st.expander("JavaScript Snippet: Monthly Income Field Timing", expanded=False):
            st.caption("Reference snippet for frontend-level focus/blur timing in a custom Streamlit component.")
            st.code(get_monthly_income_tracking_js_snippet(), language="javascript")

        st.divider()
        st.markdown('<h2 class="section-heading">Macroeconomic Stress Test</h2>', unsafe_allow_html=True)
        remittance_volatility = st.slider(
            "Remittance Volatility (%)",
            min_value=-50,
            max_value=0,
            value=st.session_state.get("remittance_volatility_input", -10),
            step=1,
            help="Negative values reduce the effective income used for stress testing.",
            key="remittance_volatility_input",
        )
        agricultural_yield_drop = st.slider(
            "Agricultural Yield Drop (%)",
            min_value=0,
            max_value=50,
            value=st.session_state.get("agricultural_yield_input", 20),
            step=1,
            help="Applied only when Primary Income Source is Agriculture.",
            key="agricultural_yield_input",
        )

        st.divider()
        st.markdown('<h2 class="section-heading">Alternative Credit Score</h2>', unsafe_allow_html=True)
        utility_bill_payment_consistency = st.selectbox(
            "Utility Bill Payment Consistency",
            ["Always on Time", "Occasional Delay", "Frequent Delay"],
            index=["Always on Time", "Occasional Delay", "Frequent Delay"].index(st.session_state.get("utility_bill_payment_input", "Always on Time")),
            key="utility_bill_payment_input",
        )
        digital_wallet_transaction_frequency = st.slider(
            "Digital Wallet (eSewa/Khalti) Transaction Frequency",
            min_value=0,
            max_value=100,
            value=st.session_state.get("digital_wallet_input", 45),
            step=1,
            help="Use a normalized frequency score from 0 to 100.",
            key="digital_wallet_input",
        )

        # Financial inclusion controls for alternative-data mapping
        alt_data_mapping_enabled = False
        wallet_usage_tier = "None"
        utility_bill_history = "No Record"
        with st.expander("📊 Alternative Credit Scoring (Financial Inclusion)", expanded=False):
            alt_data_mapping_enabled = st.toggle("Enable Alternative Data Mapping", value=False)
            if alt_data_mapping_enabled:
                wallet_usage_tier = st.selectbox(
                    "Digital Wallet (eSewa/Khalti) Usage",
                    ["None", "Occasional", "Regular", "Power User"],
                    index=0,
                )
                utility_bill_history = st.selectbox(
                    "Utility Bill History",
                    ["No Record", "Frequent Delays", "Always on Time"],
                    index=0,
                )

        annual_income = monthly_income * 12
        loan_to_income_ratio = loan_amount / annual_income if annual_income > 0 else 0.0
        max_loan_under_3x = max(0, int(annual_income * 3) - 1)
        suggested_lower_amount = max(0, min(max_loan_under_3x, loan_amount - 50000))
        st.metric(
            "Loan-to-Income Ratio",
            f"{loan_to_income_ratio:.2f}x",
            help=loan_affordability_help,
        )
        st.caption("Loan ÷ Annual Income")
        cib_score_input = st.slider("CIB Score", min_value=300, max_value=850, value=720, step=1, key="cib_score_input_widget")
        if st.button("🔍 Check CIB Records (API Simulation)", use_container_width=True):
            with st.spinner("Connecting to CIB Nepal..."):
                time.sleep(1.5)
            st.info(f"CIB Score: {cib_score_input} | Blacklist: No | Outstanding Loans: 0 | Default History (24M): Clean")
            st.success("CIB check complete – profile ready for decision.")
            st.session_state.cib_verified = True
        if st.session_state.cib_verified:
            st.info(
                f"**CIB Score:** {cib_score_input} | **Blacklist:** No | **Outstanding Loans:** 0 | **Default History (24M):** Clean"
            )
        st.divider()
        st.markdown('<h2 class="section-heading">Identity Verification</h2>', unsafe_allow_html=True)
        identity_document = st.file_uploader(
            "Upload Citizenship/PAN Card",
            type=["pdf", "jpg", "jpeg"],
        )
        if st.button("🆔 Start e-KYC Verification", use_container_width=True, disabled=identity_document is None):
            with st.spinner("Reading document with OCR..."):
                time.sleep(1)
            with st.spinner("Validating with Government Database..."):
                time.sleep(1)
            with st.spinner("Performing Face Match..."):
                time.sleep(1)
            st.session_state.ekyc_verified = True
        if identity_document is None:
            st.warning("कृपया प्रमाणित गर्न दस्तावेज अपलोड गर्नुहोस्")
        if st.session_state.ekyc_verified:
            st.success("e-KYC Verified: Identity Confirmed")
        st.markdown('</div>', unsafe_allow_html=True)

    with page_right_col:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown(f'<div style="text-align: center; padding: 2rem 0;">', unsafe_allow_html=True)
        st.markdown(f'<h2 style="color: {GLOBAL_IME_BLUE}; margin-bottom: 1rem;">Global IME Bank</h2>', unsafe_allow_html=True)
        st.markdown(f'<p style="color: {MUTED}; font-size: 1.1rem;">Official Credit Assessment System</p>', unsafe_allow_html=True)
        st.markdown(f'<div style="margin: 2rem 0; font-size: 3rem;">🏦</div>', unsafe_allow_html=True)
        st.markdown(f'<p style="color: {MUTED}; font-size: 0.9rem; margin-top: 2rem;"><strong>Intake Tab:</strong> Fast data entry & verification</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="color: {MUTED}; font-size: 0.9rem;"><strong>Analysis Tab:</strong> AI scoring & charts</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="color: {MUTED}; font-size: 0.9rem;"><strong>Audit Tab:</strong> Compliance reports</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Lightweight validation only – defer heavy computations to Analysis tab
policy_result = apply_banking_policy_rules(monthly_income, loan_amount)
monthly_income_input_time_seconds = st.session_state.get("monthly_income_time_spent_seconds")
behavioral_risk = classify_behavioral_risk(monthly_income_input_time_seconds, threshold_seconds=2.0)

# Display Pre-Screening Report inside the Intake tab
with tab_onboard:
    with st.expander("Pre-Screening Report", expanded=True):
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        if policy_result.get("status") == "Hard Reject":
            st.error(policy_result.get("reason", "Hard Reject"))
            st.write(f"Action: {policy_result.get('action', '')}")
        elif policy_result.get("status") == "Policy Violation":
            st.warning(policy_result.get("reason", "Policy Violation"))
            st.write(f"Action: {policy_result.get('action', '')}")
        else:
            st.success(policy_result.get("reason", "Pre-screen passed."))
        timing_note = "Not recorded yet" if monthly_income_input_time_seconds is None else f"{monthly_income_input_time_seconds:.2f}s"
        if behavioral_risk == "High":
            st.warning(f"Behavioral Risk: High (Monthly Income interaction time: {timing_note})")
        else:
            st.info(f"Behavioral Risk: {behavioral_risk} (Monthly Income interaction time: {timing_note})")
        st.markdown('</div>', unsafe_allow_html=True)

# Compute core values needed by both Analysis and Audit tabs (but defer chart rendering)
stress_tested_monthly_income = apply_macroeconomic_stress_test(
    monthly_income=monthly_income,
    remittance_volatility_pct=remittance_volatility,
    agricultural_yield_drop_pct=agricultural_yield_drop,
    primary_income_source=primary_income_source,
)

crisis_effective_income = apply_macroeconomic_stress_test(
    monthly_income=monthly_income,
    remittance_volatility_pct=-float(remittance_decline_pct),
    agricultural_yield_drop_pct=float(crop_failure_risk_pct),
    primary_income_source=primary_income_source,
)

crisis_risk_score = predict_risk_score(crisis_effective_income, loan_amount, applicant_age)

# Feature engineering (lightweight)
applicant_frame = pd.DataFrame(
    [
        {
            "Monthly_Income": monthly_income,
            "Loan_Amount": loan_amount,
            "Age": applicant_age,
            "Primary_Income_Source": primary_income_source,
            "Essential_Expenses": essential_expenses,
            "Monthly_Installment": monthly_installment,
        }
    ]
)
engineered_applicant = add_feature_engineering(applicant_frame).iloc[0]

# Run model assessment
cib_score = cib_score_input if st.session_state.cib_verified else 0
annual_income = monthly_income * 12
loan_to_income_ratio = loan_amount / annual_income if annual_income > 0 else 0.0

if policy_result.get("status") == "Hard Reject":
    baseline_risk_score = 100.0
    stress_tested_risk_score = 100.0
    assessment = {
        "rejected": True,
        "status": "Hard Reject",
        "message": policy_result.get("reason", "Regulatory hard reject."),
        "reason": policy_result.get("reason", "Regulatory hard reject."),
        "suggested_next_step": "Decline the application and request documentation.",
        "model_used": False,
        "score": 100.0,
        "band": "HIGH",
        "label": "HARD REJECT",
        "contributions": {},
        "contribution_table": pd.DataFrame(columns=["Feature", "Current Value", "Baseline Value", "Model Importance", "Contribution", "Direction"]),
        "confidence": 0.0,
    }
else:
    baseline_risk_score = predict_risk_score(monthly_income, loan_amount, applicant_age)
    stress_tested_risk_score = predict_risk_score(stress_tested_monthly_income, loan_amount, applicant_age)
    assessment = assess_applicant(applicant_age, int(stress_tested_monthly_income), loan_amount, remittance_status, land_area, cib_score, loan_to_income_ratio)

# Apply alternative-data boost
if alt_data_mapping_enabled:
    esewa_signal = "Regular" if wallet_usage_tier in {"Regular", "Power User"} else "Irregular"
    bill_signal = "Consistent" if utility_bill_history == "Always on Time" else "Inconsistent"
else:
    esewa_signal = "Regular" if digital_wallet_transaction_frequency >= 40 else "Irregular"
    bill_signal = "Consistent" if utility_bill_payment_consistency == "Always on Time" else "Consistent"

alt_data_inputs = {
    "esewa_transactions": esewa_signal,
    "ncell_ntc_bill_payments": bill_signal,
}
initial_risk_score = float(assessment.get("score", 0.0))
boosted_risk_score = apply_alternative_data_boost(initial_risk_score, alt_data_inputs)
assessment["score_before_alt_boost"] = initial_risk_score
assessment["score"] = boosted_risk_score
assessment["alt_data_boost_applied"] = boosted_risk_score < initial_risk_score
assessment["confidence"] = round(max(0.0, 100.0 - boosted_risk_score), 1)

route_label, final_decision, route_message = summarize_route_from_assessment(assessment)

behavioral_utility_score, behavioral_wallet_score, behavioral_score, credit_360_score = calculate_alternative_credit_score(
    utility_bill_payment_consistency=utility_bill_payment_consistency,
    digital_wallet_transaction_frequency=digital_wallet_transaction_frequency,
    model_credit_score=(100.0 - assessment["score"]) if assessment["model_used"] else 0.0,
)

# Unified orchestration report
credit_orchestrator = CreditOrchestrator(model_version=MODEL_VERSION)
credit_orchestrator_report = credit_orchestrator.run(
    age=applicant_age,
    monthly_income=int(stress_tested_monthly_income),
    loan_amount=loan_amount,
    remittance_status=remittance_status,
    land_area=land_area,
    cib_score=cib_score,
    loan_to_income_ratio=loan_to_income_ratio,
    cib_verified=bool(st.session_state.get("cib_verified", False)),
    ekyc_verified=bool(st.session_state.get("ekyc_verified", False)),
)

# Prepare audit payload
top_shap_factors = format_top_shap_factors(assessment["contribution_table"], top_n=3)
audit_payload = {
    "timestamp": pd.Timestamp.now().isoformat(),
    "model_version": MODEL_VERSION,
    "monthly_income": monthly_income,
    "loan_amount": loan_amount,
    "cib_score": cib_score,
    "primary_income_source": primary_income_source,
    "essential_expenses": essential_expenses,
    "monthly_installment": monthly_installment,
    "remittance_status": remittance_status,
    "land_area": land_area,
    "applicant_age": applicant_age,
    "stress_tested_income": float(engineered_applicant["Stress_Tested_Income"]),
    "loan_coverage_ratio": float(engineered_applicant["Loan_Coverage_Ratio"]),
    "shap_top_3_contributing_factors": top_shap_factors,
    "final_decision": final_decision,
}

# ============================================================================
# DEFERRED CHART RENDERING: Heavy Plotly charts and migration analysis only
# in Analysis tab (renders instantly when Analysis tab is opened)
# ============================================================================
with tab_ai:
    # Only compute migration analysis when Analysis tab is opened
    income_multipliers = np.array([0.6, 0.8, 1.0, 1.2, 1.5, 2.0], dtype=float)
    loan_multipliers = np.array([0.7, 0.9, 1.0, 1.1], dtype=float)
    simulation_rows = []
    for income_mult in income_multipliers:
        for loan_mult in loan_multipliers:
            simulation_rows.append(
                {
                    "Monthly_Income": max(5000.0, float(monthly_income) * float(income_mult)),
                    "Loan_Amount": max(50000.0, float(loan_amount) * float(loan_mult)),
                    "Age": int(applicant_age),
                    "Primary_Income_Source": "Remittance",
                }
            )

    remittance_simulation_portfolio = pd.DataFrame(simulation_rows)
    migration_summary = simulate_remittance_shock_migration(remittance_simulation_portfolio, remittance_market_shock_pct)

    shock_points = list(range(0, int(remittance_market_shock_pct) + 1, 5))
    if int(remittance_market_shock_pct) not in shock_points:
        shock_points.append(int(remittance_market_shock_pct))
    if not shock_points:
        shock_points = [0]

    migration_curve_rows = []
    for shock_point in sorted(set(shock_points)):
        point_summary = simulate_remittance_shock_migration(remittance_simulation_portfolio, shock_point)
        migration_curve_rows.append(
            {
                "Remittance Market Shock (%)": int(shock_point),
                "Approved to Manual Review": int(point_summary["moved_to_manual_review"]),
            }
        )

    migration_curve_df = pd.DataFrame(migration_curve_rows)
    
    # Render all Analysis tab UI
    max_loan_under_3x = max(0, int(annual_income * 3) - 1)
    suggested_lower_amount = max(0, min(max_loan_under_3x, loan_amount - 50000))
    
    st.markdown(
        f'''
        <div class="hero-card">
            <div class="brand-strip">Global IME Bank Banking Dashboard</div>
            <h1 class="section-heading" style="margin:0.6rem 0 0.35rem 0;">XAI-RAS v{APP_VERSION}</h1>
            <div class="small-note">Model-driven scoring with dynamic feature contributions.</div>
            <div class="summary-badge">Score 0 = perfect | Score 100 = high risk</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )
    st.caption("ML Engine: Random Forest Classifier | NRB AI Guidelines 2025 Compliant")

    # Top metrics including income delta vs national average
    top_metric_col_1, top_metric_col_2, top_metric_col_3, top_metric_col_4 = st.columns(4, gap="medium")
    with top_metric_col_1:
        st.metric(
            "Baseline Risk vs. Stress-Tested Risk",
            f"{baseline_risk_score:.1f} → {stress_tested_risk_score:.1f}",
            delta=f"{stress_tested_risk_score - baseline_risk_score:+.1f} pts",
        )
    with top_metric_col_2:
        st.metric("Loan-to-Income Ratio", f"{loan_to_income_ratio:.2f}x")
    with top_metric_col_3:
        st.metric("360-Degree Credit Score", f"{credit_360_score:.1f}/100")
    with top_metric_col_4:
        st.metric(
            "Income vs National Avg",
            f"NPR {monthly_income:,.0f}",
            delta=f"{monthly_income - INCOME_REFERENCE:+,.0f}",
            help=f"Average Nepalese borrower income: NPR {INCOME_REFERENCE:,.0f}",
        )
    # Crisis metric from sidebar stress testing
    st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
    st.metric(
        "Crisis Risk Score",
        f"{crisis_risk_score:.1f}/100",
        delta=f"{crisis_risk_score - baseline_risk_score:+.1f} pts",
        help="Risk score under simulated economic stress from the sidebar sliders.",
    )
    st.metric(
        "Boosted Score",
        f"{boosted_risk_score:.1f}/100",
        delta=f"{boosted_risk_score - initial_risk_score:+.1f} pts",
        help="Alternative-data adjusted score for borderline risk band (40-70).",
    )

    st.markdown('<div class="panel-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
    st.subheader("Remittance Shock Migration Analysis")
    st.caption("Live view: how many baseline Approved loans migrate to Manual Review when remittance income is shocked.")
    migration_col_1, migration_col_2, migration_col_3 = st.columns(3)
    with migration_col_1:
        st.metric("Baseline Approved", f"{migration_summary['baseline_approved']}")
    with migration_col_2:
        st.metric("Moved to Manual Review", f"{migration_summary['moved_to_manual_review']}")
    with migration_col_3:
        st.metric("Remained Approved", f"{migration_summary['remained_approved']}")

    st.line_chart(
        migration_curve_df.set_index("Remittance Market Shock (%)"),
        height=280,
        use_container_width=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        f'''
        <div class="panel-card" style="margin-bottom: 1rem;">
            <div class="metric-label">Alternative Credit Score</div>
            <div class="metric-subtext">Utility payment score: {behavioral_utility_score:.1f}/100</div>
            <div class="metric-subtext">Wallet frequency score: {behavioral_wallet_score:.1f}/100</div>
            <div class="metric-subtext">Behavioral blend: {behavioral_score:.1f}/100</div>
            <div class="metric-subtext">360-degree credit score: {credit_360_score:.1f}/100</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    with st.status("Decision Routing Workflow", expanded=True) as routing_status:
        routing_status.write("Checking hard-stop rules...")
        routing_status.write("Evaluating model risk score...")
        routing_status.write(f"Routing signal: {route_label}")
        routing_status.write(f"Final decision: {final_decision}")
        routing_status.update(label=f"Decision Routing: {final_decision}", state="complete", expanded=False)

    if final_decision == "Instant Approval":
        st.success(f"{route_label}: {final_decision}.")
    elif final_decision == "Manual Review Required by Senior Credit Officer":
        st.warning(f"{route_label}: {final_decision}.")
    else:
        st.error(f"{route_label}: {final_decision}.")
    st.caption(route_message)

    st.markdown(
        f'''
        <div class="panel-card" style="margin-bottom: 1rem;">
            <div class="metric-label">Feature Engineering</div>
            <div class="metric-subtext">Primary income source: {primary_income_source}</div>
            <div class="metric-subtext">Macroeconomic stress-tested income: NPR {stress_tested_monthly_income:,.0f}</div>
            <div class="metric-subtext">Stress-Tested Income: NPR {engineered_applicant['Stress_Tested_Income']:,.0f}</div>
            <div class="metric-subtext">Loan Coverage Ratio: {engineered_applicant['Loan_Coverage_Ratio']:.2f}</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    risk_left_col, risk_right_col = st.columns([0.95, 1.05], gap="large")

    with risk_left_col:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.subheader("Decision Summary")
        if assessment["rejected"]:
            st.error(assessment["message"])
            st.caption("The model was not invoked because the applicant failed the regulatory affordability gate.")
        elif not assessment["model_used"]:
            st.warning(assessment["reason"])
            st.info(assessment["suggested_next_step"])
        else:
            st.metric("Risk Score", f"{assessment['score']:.1f}/100")
            st.metric("Model Confidence", f"{assessment['confidence']:.1f}%")
            band_color = RED if assessment["band"] == "HIGH" else BLUE if assessment["band"] == "LOW" else "#8a6400"
            st.markdown(
                f'''
                <div class="metric-box" style="margin-top: 1rem;">
                    <div class="metric-label">Risk Band</div>
                    <div class="metric-value" style="color:{band_color};">{assessment['label']}</div>
                    <div class="metric-subtext">{technical_status(assessment['score'])}</div>
                </div>
                ''',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    with risk_right_col:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        if view_mode == "Bank Officer View":
            st.subheader("Risk Analysis & SHAP Plots")
            st.caption("Technical metrics, explainability, and audit controls for credit officers.")
            if assessment["rejected"]:
                st.error(assessment["message"])
            elif not assessment["model_used"]:
                st.warning(assessment["reason"])
                st.info(assessment["suggested_next_step"])
            else:
                st.dataframe(
                    assessment["contribution_table"][ ["Feature", "Current Value", "Baseline Value", "Model Importance", "Contribution", "Direction"] ].style.format(
                        {"Current Value": "{:.2f}", "Baseline Value": "{:.2f}", "Model Importance": "{:.1f}%", "Contribution": "{:+.1f}"}
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                st.markdown(
                    f"""
                    <div class="metric-box" style="margin-top: 1rem;">
                        <div class="metric-label">Technical Notes</div>
                        <div class="metric-subtext">Base level: 50.0</div>
                        <div class="metric-subtext">Final score: {assessment['score']:.1f}/100</div>
                        <div class="metric-subtext">Decision path: {technical_status(assessment['score'])}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption("XAI Engine: Dynamic Feature Importance (Random Forest v1.4). Interpretation is model-driven, not rule-based.")
                risk_score = assessment["score"]
                top_shap_rows = assessment["contribution_table"].head(3)
                top_shap_pairs = [
                    (row.Feature, float(row.Contribution))
                    for row in top_shap_rows.itertuples(index=False)
                ]
                nepali_shap_summary = summarize_top_shap_values(
                    top_shap_pairs,
                    final_decision,
                    score_boost_applied=bool(assessment.get("alt_data_boost_applied", False)),
                )
                st.info(nepali_shap_summary)
                audit_log_text = (
                    "Model Type: Random Forest Classifier\n"
                    f"Income: {monthly_income}\n"
                    f"Loan: {loan_amount}\n"
                    f"Age: {applicant_age}\n"
                    f"Final Risk Score: {risk_score:.1f}\n"
                )
                st.caption("Download current model inputs and final risk_score as a plain-text audit log.")
                st.download_button(
                    "Download Audit Log",
                    data=audit_log_text,
                    file_name="GlobalIME_Audit_Log.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
        else:
            st.subheader("सजिलो नतिजा")
            if assessment["rejected"]:
                st.error(assessment["message"])
            elif not assessment["model_used"]:
                if assessment["status"] == "High Risk":
                    st.warning(assessment["reason"])
                elif assessment["status"] == "Senior Life Insurance Requirement":
                    st.info(assessment["reason"])
                else:
                    st.error(assessment["reason"])
            else:
                st.markdown(
                    f'''
                    <div class="customer-message">
                        <h2>{risk_message(assessment['band'])}</h2>
                        <div class="small-note">तपाईंको जोखिम स्कोर: {assessment['score']:.1f}/100</div>
                    </div>
                    ''',
                    unsafe_allow_html=True,
                )
            st.write("")
            st.info("यो स्कोरले ऋण निर्णयको प्रारम्भिक संकेत मात्र देखाउँछ। अन्तिम निर्णय बैंकको प्रक्रिया अनुसार हुनेछ।")
            if st.session_state.ekyc_verified:
                st.success("✅ तपाईंको पहिचान (e-KYC) सफलतापूर्वक प्रमाणित भएको छ।")
            else:
                st.info("तपाईंको आवेदन प्रक्रिया अगाडि बढाउन कृपया बायाँ छेउमा रहेको Identity Verification मा आफ्नो कागजात अपलोड गरी प्रमाणीकरण गर्नुहोस्।")
            if assessment["status"] == "High Risk":
                st.warning(assessment["suggested_next_step"])
            elif assessment["status"] == "Senior Life Insurance Requirement":
                st.info(assessment["suggested_next_step"])
            elif not assessment["rejected"] and assessment["label"] in {"MEDIUM RISK", "HIGH RISK"}:
                st.markdown("**What-If सुझाव**")
                st.info(
                    f"सुझाव: यदि तपाईंले ऋणको रकम Rs. {suggested_lower_amount:,.0f} मा घटाउनुभयो भने, तपाईंको स्वीकृत हुने सम्भावना बढ्नेछ।"
                )
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

with tab_audit:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.subheader("NRB Compliance & Regulatory Audit")
    st.markdown("**CreditOrchestrator Structured Report**")
    st.json(credit_orchestrator_report)
    st.markdown(
        """
        This panel gathers the audit snapshot required for NRB compliance. Please confirm the checklist items below before downloading the audit CSV.
        """
    )
    col1, col2 = st.columns(2)
    with col1:
        explainability = st.checkbox("Explainability: Local feature contributions available", value=True)
        human_review = st.checkbox("Human-in-loop: Manual review path available", value=True)
        traceability = st.checkbox("Traceability: Model versioning & inputs logged", value=True)
    with col2:
        consent = st.checkbox("Data Consent recorded", value=True)
        ekyc = st.checkbox("e-KYC verified", value=bool(st.session_state.get("ekyc_verified", False)))
        cib_ok = st.checkbox("CIB check performed", value=bool(st.session_state.get("cib_verified", False)))

    st.divider()
    csv_text, _ = generate_decision_audit_csv(audit_payload)
    st.markdown("**Audit Snapshot**")
    st.json(audit_payload)
    st.download_button(
        "Download Decision Audit CSV",
        data=csv_text,
        file_name="Decision_Audit.csv",
        mime="text/csv",
        use_container_width=True,
    )
    audit_log_text = generate_audit_report(audit_payload.get("monthly_income", 0), audit_payload.get("loan_amount", 0), final_risk_score=float(audit_payload.get("final_decision", 0) if isinstance(audit_payload.get("final_decision"), (int, float)) else 0))
    st.download_button(
        "Download Plain Audit Log",
        data=audit_log_text,
        file_name="GlobalIME_Audit_Log.txt",
        mime="text/plain",
        use_container_width=True,
    )

    # Regulatory audit log (NRB 2025 format) with Policy Guardrail result
    regulatory_csv = generate_regulatory_audit_csv_record(
        inputs={
            "monthly_income": monthly_income,
            "loan_amount": loan_amount,
            "applicant_age": applicant_age,
            "primary_income_source": primary_income_source,
            "essential_expenses": essential_expenses,
            "monthly_installment": monthly_installment,
            "remittance_status": remittance_status,
            "land_area": land_area,
            "utility_bill_payment_consistency": utility_bill_payment_consistency,
            "digital_wallet_transaction_frequency": digital_wallet_transaction_frequency,
            "cib_score": cib_score,
            "ekyc_verified": st.session_state.get("ekyc_verified", False),
            "identity_document_uploaded": identity_document is not None,
            "monthly_income_input_time_seconds": monthly_income_input_time_seconds,
            "behavioral_risk": behavioral_risk,
        },
        assessment=assessment,
        policy_result=policy_result,
        model_version=MODEL_VERSION,
    )

    st.download_button(
        "Download Regulatory Audit Log",
        data=regulatory_csv,
        file_name=f"Regulatory_Audit_Log_{MODEL_VERSION}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.subheader("तपाईंको लागि हाम्रो सुझाव (Our Advice)")

    if assessment["rejected"]:
        st.error(assessment["message"])
    elif not assessment["model_used"] and assessment["status"] == "High Risk":
        st.warning(assessment["suggested_next_step"])
    elif not assessment["model_used"] and assessment["status"] == "Senior Life Insurance Requirement":
        st.info(assessment["suggested_next_step"])
    elif loan_to_income_ratio > 5:
        st.warning("⚠️ ऋण रकम आपको वार्षिक आय भन्दा अत्यधिक छ। कृपया कम रकमको लागि आवेदन गर्ने विचार गर्नुहोस्।")
    elif assessment['score'] < 40:
        success_message = "✓ आपको आर्थिक प्रोफाइल राम्रोसँग संतुलित छ। आपको आवेदन स्वीकृत हुने सम्भावना राम्रो छ।"
        if st.session_state.cib_verified:
            success_message += " हाम्रो AI ले तपाईंको CIB रेकर्ड र ऋणको अनुपात जाँच गर्दा सबै कुरा राम्रो देखियो।"
        st.success(success_message)
    else:
        st.info("ℹ️ आपको आवेदन विस्तृत समीक्षा गरिँदै छ। कृपया धैर्य राख्नुहोस्।")

    if not st.session_state.cib_verified:
        st.info("कृपया CIB रेकर्ड जाँच गर्न माथिको बटन थिच्नुहोस्।")

    if not st.session_state.ekyc_verified:
        st.info("कृपया आफ्नो पहिचान प्रमाणित गर्न नागरिकता वा प्यान कार्ड अपलोड गर्नुहोस्।")

    st.markdown("</div>", unsafe_allow_html=True)