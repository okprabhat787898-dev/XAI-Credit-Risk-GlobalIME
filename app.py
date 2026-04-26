from pathlib import Path
import math
import time

import numpy as np
import pandas as pd
import streamlit as st

APP_VERSION = "1.4.0"
RED = "#C5161D"
BLUE = "#004189"
RISK_REDUCTION_BLUE = "#1d4ed8"
TEXT = "#10233f"
MUTED = "#50627f"
CARD = "#ffffff"
LOGO_PATH = Path(__file__).resolve().parent / "assets" / "logo.png"

RISK_WEIGHTS = {
    "Income": 0.45,
    "Remittance": 0.20,
    "Land Area": 0.20,
    "Age": 0.15,
}

INCOME_REFERENCE = 80000.0
LAND_REFERENCE = 5.0
AGE_REFERENCE = 35.0
AGE_SPREAD = 25.0

st.set_page_config(page_title=f"XAI-RAS v{APP_VERSION} | Global IME Bank", page_icon="📊", layout="wide")

if "cib_verified" not in st.session_state:
    st.session_state.cib_verified = False
if "ekyc_verified" not in st.session_state:
    st.session_state.ekyc_verified = False

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
        color: var(--text-strong);
        letter-spacing: -0.02em;
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


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(float(value), maximum))


def income_risk(income: float) -> float:
    return clamp((INCOME_REFERENCE - float(income)) / INCOME_REFERENCE)


def remittance_risk(remittance_status: int) -> float:
    return 0.0 if int(remittance_status) == 1 else 1.0


def land_risk(land_area: float) -> float:
    return clamp((LAND_REFERENCE - float(land_area)) / LAND_REFERENCE)


def age_risk(age: int) -> float:
    return clamp(abs(float(age) - AGE_REFERENCE) / AGE_SPREAD)


def assess_applicant(age: int, income: int, remittance_status: int, land_area: float, loan_to_income_ratio: float = 0.0) -> dict:
    factor_risks = {
        "Income": income_risk(income),
        "Remittance": remittance_risk(remittance_status),
        "Land Area": land_risk(land_area),
        "Age": age_risk(age),
    }
    income_factor = 1.0 - factor_risks["Income"]
    remittance_factor = 1.0 - factor_risks["Remittance"]

    ml_logit = -0.5
    ml_logit += income_factor * -0.45
    ml_logit += remittance_factor * -0.6
    ml_logit += loan_to_income_ratio * 0.5

    score = (1 / (1 + math.exp(-ml_logit))) * 100
    
    contributions = {
        name: (factor_risks[name] - 0.5) * RISK_WEIGHTS[name] * 100.0 for name in RISK_WEIGHTS
    }

    factor_table = pd.DataFrame(
        [
            {
                "Factor": name,
                "Risk Signal": factor_risks[name],
                "Weight": weight,
                "Contribution": contributions[name],
            }
            for name, weight in RISK_WEIGHTS.items()
        ]
    )
    factor_table["Direction"] = np.where(factor_table["Contribution"] >= 0, "Increases risk", "Reduces risk")
    factor_table = factor_table.sort_values("Contribution", key=lambda series: series.abs(), ascending=False)

    if score < 35:
        band = "LOW"
        label = "LOW RISK"
    elif score < 65:
        band = "MEDIUM"
        label = "MEDIUM RISK"
    else:
        band = "HIGH"
        label = "HIGH RISK"

    return {
        "score": round(score, 1),
        "band": band,
        "label": label,
        "factor_risks": factor_risks,
        "contributions": contributions,
        "factor_table": factor_table,
        "confidence": round(100.0 - score, 1),
    }


def render_score_chart(contributions: dict[str, float]):
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
            title=dict(text="XAI Risk Drivers", x=0.02, xanchor="left"),
            xaxis=dict(title="Risk push / relief points", zeroline=False, gridcolor="#e6edf5", tickfont=dict(color=TEXT)),
            yaxis=dict(tickfont=dict(color=TEXT)),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    except Exception:
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


def generate_audit_report(applicant_income: float, loan_amount: float, final_risk_score: float = 27.6) -> str:
    return (
        "Audit Report\n"
        f"Applicant's Income: NPR {applicant_income:,.0f}\n"
        f"Loan Amount: NPR {loan_amount:,.0f}\n"
        f"Final Risk Score: {final_risk_score:.1f}\n"
    )


with st.sidebar:
    st.markdown('<div class="sidebar-box">', unsafe_allow_html=True)
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=250)
    else:
        st.warning("Logo not found at assets/logo.png")
    st.markdown(f'<div class="brand-strip">XAI-RAS v{APP_VERSION} | Global IME Bank</div>', unsafe_allow_html=True)
    st.caption("NRB AI Guidelines 2025 aligned: transparency, traceability, human review.")
    view_mode = st.radio("View Mode", ["Bank Officer View", "Sajilo Natija (Customer View)"], index=0)
    st.divider()
    st.subheader("Applicant Inputs")
    monthly_income = st.slider("Monthly Income (NPR)", min_value=0, max_value=200000, value=50000, step=1000)
    loan_affordability_help = "Loan-to-Income Ratio evaluates the requested loan amount against your total annual earnings to measure affordability"
    loan_amount = st.slider(
        "Desired Loan Amount (ऋण रकम) (NPR)",
        min_value=50000,
        max_value=5000000,
        value=500000,
        step=50000,
        help=loan_affordability_help,
    )
    remittance_status = st.slider(
        "Remittance Status",
        min_value=0,
        max_value=1,
        value=1,
        step=1,
        help="0 = No remittance, 1 = Remittance received",
    )
    land_area = st.slider("Land Area (Ropani)", min_value=0.0, max_value=20.0, value=1.0, step=0.1)
    applicant_age = st.slider("Applicant Age", min_value=18, max_value=80, value=30, step=1)
    st.caption(f"Remittance selected: {'Received' if remittance_status == 1 else 'Not received'}")
    
    # Calculate loan-to-income ratio
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
    if st.button("🔍 Check CIB Records (API Simulation)", use_container_width=True):
        with st.spinner("Connecting to CIB Nepal..."):
            time.sleep(1.5)
        st.info("CIB Score: 720 | Blacklist: No | Outstanding Loans: 0 | Default History (24M): Clean")
        st.success("CIB check complete – profile ready for decision.")
        st.session_state.cib_verified = True
    if st.session_state.cib_verified:
        st.info(
            "**CIB Score:** 720 | **Blacklist:** No | **Outstanding Loans:** 0 | **Default History (24M):** Clean"
        )
    st.divider()
    st.subheader("Identity Verification")
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

assessment = assess_applicant(applicant_age, monthly_income, remittance_status, land_area, loan_to_income_ratio)

st.markdown(
    f'''
    <div class="hero-card">
        <div class="brand-strip">Global IME Bank Banking Dashboard</div>
        <h1 style="margin:0.6rem 0 0.35rem 0;">XAI-RAS v{APP_VERSION}</h1>
        <div class="small-note">Weighted risk score: Income 45%, Remittance 20%, Land Area 20%, Age 15%.</div>
        <div class="summary-badge">Score 0 = perfect | Score 100 = high risk</div>
    </div>
    ''',
    unsafe_allow_html=True,
)

top_col, bottom_col = st.columns([1.1, 1.0], gap="large")

with top_col:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.subheader("Decision Summary")
    metrics_col_1, metrics_col_2, metrics_col_3 = st.columns(3, gap="medium")
    with metrics_col_1:
        st.markdown(
            f'''
            <div class="metric-box">
                <div class="metric-label">Risk Score</div>
                <div class="metric-value" style="color:{RED};">{assessment['score']:.1f}/100</div>
                <div class="metric-subtext">Policy midpoint: 50</div>
            </div>
            ''',
            unsafe_allow_html=True,
        )
    with metrics_col_2:
        st.markdown(
            f'''
            <div class="metric-box">
                <div class="metric-label">Model Confidence</div>
                <div class="metric-value" style="color:{BLUE};">{assessment['confidence']:.1f}%</div>
                <div class="metric-subtext">Higher means lower risk pressure</div>
            </div>
            ''',
            unsafe_allow_html=True,
        )
    with metrics_col_3:
        band_color = RED if assessment["band"] == "HIGH" else BLUE if assessment["band"] == "LOW" else "#8a6400"
        st.markdown(
            f'''
            <div class="metric-box">
                <div class="metric-label">Risk Band</div>
                <div class="metric-value" style="color:{band_color};">{assessment['label']}</div>
                <div class="metric-subtext">{technical_status(assessment['score'])}</div>
            </div>
            ''',
            unsafe_allow_html=True,
        )
    st.write("")
    render_score_chart(assessment["contributions"])
    st.caption("Positive bars increase risk; negative bars reduce risk. The chart is centered on a neutral baseline.")
    st.markdown("</div>", unsafe_allow_html=True)

with bottom_col:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    if view_mode == "Bank Officer View":
        st.subheader("Bank Officer View")
        st.caption("Technical metrics, explainability, and audit controls for credit officers.")
        audit_report_text = generate_audit_report(monthly_income, loan_amount)
        st.dataframe(
            assessment["factor_table"][ ["Factor", "Risk Signal", "Weight", "Contribution", "Direction"] ].style.format(
                {"Risk Signal": "{:.2f}", "Weight": "{:.2f}", "Contribution": "{:+.1f}"}
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
                <div class="metric-subtext">Interpretation uses the configured weights only.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.download_button(
            "Download Audit Report",
            data=audit_report_text,
            file_name="Global_IME_Audit_Report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.subheader("सजिलो नतिजा")
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
        if assessment["label"] in {"MEDIUM RISK", "HIGH RISK"}:
            st.markdown("**What-If सुझाव**")
            st.info(
                f"सुझाव: यदि तपाईंले ऋणको रकम Rs. {suggested_lower_amount:,.0f} मा घटाउनुभयो भने, तपाईंको स्वीकृत हुने सम्भावना बढ्नेछ।"
            )
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

st.markdown('<div class="panel-card">', unsafe_allow_html=True)
st.subheader("तपाईंको लागि हाम्रो सुझाव (Our Advice)")

if loan_to_income_ratio > 5:
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