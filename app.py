import streamlit as st
import pandas as pd
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="XAI-RAS: Global IME Bank", layout="wide")

# --- CUSTOMER/OFFICER TOGGLE (From your paper's Dual-Audience requirement) ---
st.sidebar.title("View Mode")
mode = st.sidebar.radio("Select View:", ["Bank Officer (Technical)", "Customer (Plain Nepali)"])

if mode == "Customer (Plain Nepali)":
    st.title("ऋण जोखिम मूल्याङ्कन प्रणाली (XAI-RAS)")
    st.info("तपाईंको ऋण आवेदनको स्थिति यहाँ जाँच गर्नुहोस्।")
else:
    st.title("XAI-RAS: Smart Credit Approval Dashboard")
    st.write("Autonomous Credit & Lending Orchestrator | Global IME Bank Hackathon 2026")

# --- INPUT PANEL (Figure D1 in your paper) ---
st.header("Borrower Input Panel")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=80, value=30)
    income = st.number_input("Monthly Income (NPR)", min_value=0, value=50000)
    
with col2:
    remittance = st.selectbox("Receives Remittance?", ["Yes", "No"])
    land_area = st.number_input("Agricultural Land Area (Ropani)", min_value=0.0, value=1.0)

# --- PREDICTION LOGIC (Placeholder for your Ensemble Model) ---
if st.button("Run Risk Assessment"):
    st.divider()
    
    # Mock calculation for the hackathon demo
    risk_score = np.random.random() 
    
    # --- RESULT PANEL (Figure D2 in your paper) ---
    st.header("Risk Prediction Result")
    
    if risk_score < 0.3:
        st.success(f"LOW RISK (Confidence: {1-risk_score:.2%})")
        if mode == "Customer (Plain Nepali)":
            st.write("बधाई छ! तपाईंको ऋण आवेदन स्वीकृत हुने उच्च सम्भावना छ।")
    elif risk_score < 0.7:
        st.warning(f"MEDIUM RISK (Confidence: {1-risk_score:.2%})")
        st.write("Requires Human-in-the-Loop (HITL) Review.")
    else:
        st.error(f"HIGH RISK (Confidence: {1-risk_score:.2%})")
        if mode == "Customer (Plain Nepali)":
            st.write("माफ गर्नुहोला, तपाईंको आवेदन अहिलेको लागि अस्वीकृत भएको छ।")

    # --- SHAP EXPLANATION PANEL (Figure D3 in your paper) ---
        st.header("Explainability (XAI) Drivers")
        st.write("The following factors influenced this decision:")
    
    # Simple visual representation of SHAP
        factors = {"Income": 0.4, "Remittance": 0.3, "Age": 0.1, "Land Area": 0.2}
        st.bar_chart(pd.DataFrame(factors, index=["Impact"]))
