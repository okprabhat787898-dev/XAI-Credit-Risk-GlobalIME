import streamlit as st
import pandas as pd

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="XAI-RAS: Global IME Bank", layout="wide")


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(value, maximum))


def calculate_risk(age: int, income: int, remittance: str, land_area: float):
    """Simple, explainable score for demo purposes."""
    income_risk = _clamp((70000 - income) / 70000)
    age_risk = _clamp(abs(age - 35) / 35)
    land_risk = _clamp((2.5 - land_area) / 2.5)
    remittance_risk = 0.25 if remittance == "No" else 0.05

    reason_scores = {
        "Low Monthly Income": income_risk,
        "No Remittance Inflow": remittance_risk,
        "Limited Land Collateral": land_risk,
        "Age Profile": age_risk,
    }

    # Weighted sum gives a normalized risk score from 0 to 1.
    risk_score = (
        (income_risk * 0.45)
        + (remittance_risk * 0.20)
        + (land_risk * 0.20)
        + (age_risk * 0.15)
    )
    return _clamp(risk_score), reason_scores

# --- CUSTOMER/OFFICER TOGGLE (From your paper's Dual-Audience requirement) ---
st.sidebar.title("View Mode")
mode = st.sidebar.radio("Select View:", ["Bank Officer (Technical)", "Customer (Plain Nepali)"])

if mode == "Customer (Plain Nepali)":
    st.title("ऋण जोखिम मूल्याङ्कन प्रणाली (XAI-RAS)")
    st.info("तपाईंको ऋण आवेदनको स्थिति यहाँ जाँच गर्नुहोस्।")
else:
    st.title("XAI-RAS: Smart Credit Approval Dashboard")
    st.write("Autonomous Credit & Lending Orchestrator | Global IME Bank Hackathon 2026")

# --- DASHBOARD LAYOUT ---
input_col, output_col = st.columns([1.05, 1.45], gap="large")

with input_col:
    st.subheader("Borrower Input Panel")
    age = st.number_input("Age", min_value=18, max_value=80, value=30)
    income = st.number_input("Monthly Income (NPR)", min_value=0, value=50000, step=1000)
    remittance = st.selectbox("Receives Remittance?", ["Yes", "No"])
    land_area = st.number_input("Agricultural Land Area (Ropani)", min_value=0.0, value=1.0, step=0.1)
    run_assessment = st.button("Run Risk Assessment", use_container_width=True)

with output_col:
    st.subheader("Decision Dashboard")
    st.caption("Professional summary of model output, confidence, and explainability drivers")

    if run_assessment:
        risk_score, reason_scores = calculate_risk(age, income, remittance, land_area)
        risk_percent = int(round(risk_score * 100))
        confidence = 1 - risk_score

        if risk_score < 0.35:
            risk_band = "LOW RISK"
            status = "Approved Path"
            status_block = st.success
        elif risk_score < 0.65:
            risk_band = "MEDIUM RISK"
            status = "Manual Review"
            status_block = st.warning
        else:
            risk_band = "HIGH RISK"
            status = "Likely Reject"
            status_block = st.error

        m1, m2, m3 = st.columns(3)
        m1.metric("Risk Score", f"{risk_percent}/100")
        m2.metric("Risk Band", risk_band)
        m3.metric("Model Confidence", f"{confidence:.1%}")

        status_block(f"Decision Track: {status}")

        if mode == "Customer (Plain Nepali)":
            if risk_band == "LOW RISK":
                st.info("बधाई छ! तपाईंको ऋण आवेदन स्वीकृत हुने उच्च सम्भावना छ।")
            elif risk_band == "MEDIUM RISK":
                st.info("तपाईंको आवेदन थप समीक्षा (HITL) का लागि पठाइएको छ।")
            else:
                st.info("माफ गर्नुहोला, तपाईंको आवेदन अहिलेको लागि अस्वीकृत भएको छ।")

        st.markdown("### Top 3 Reasons For Decision")
        top3 = (
            pd.DataFrame(
                sorted(reason_scores.items(), key=lambda item: item[1], reverse=True)[:3],
                columns=["Reason", "Impact"],
            )
            .set_index("Reason")
        )
        st.bar_chart(top3)
    else:
        st.info("Enter borrower details and click 'Run Risk Assessment' to view the dashboard.")
