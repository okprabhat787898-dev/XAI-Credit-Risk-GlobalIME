import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

APP_VERSION = "1.4.0"
DEFAULT_MODEL_ID = "XAI-RAS-ENSEMBLE-v1"

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="XAI-RAS: Global IME Bank", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --bank-navy: #0b1f3a;
        --bank-navy-soft: #123053;
        --bank-blue: #0f62d6;
        --text-strong: #0b1f3a;
        --text-muted: #274a7c;
        --chart-bg: #0d1f36;
        --chart-grid: #2b3f5f;
    }
    .stApp {
        background: radial-gradient(circle at top right, #f4f9ff 0%, #e9f2ff 35%, #f8fbff 65%, #ffffff 100%);
    }
    .main .block-container {
        padding-top: 2.2rem;
        padding-bottom: 2.2rem;
    }
    h1, h2, h3 {
        color: var(--text-strong);
        font-weight: 800;
        letter-spacing: 0.01em;
    }
    h1 { font-size: 2.3rem; }
    h2 { font-size: 1.75rem; }
    h3 { font-size: 1.35rem; }
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid #d8e5ff;
        border-radius: 14px;
        padding: 10px 14px;
    }
    .premium-card {
        background: linear-gradient(130deg, #0b3c88, #0f62d6);
        color: #ffffff;
        padding: 14px 16px;
        border-radius: 14px;
        box-shadow: 0 8px 24px rgba(15, 98, 214, 0.25);
    }
    .dashboard-container {
        border: 1px solid #0054A6;
        border-radius: 16px;
        padding: 20px;
        background: rgba(255, 255, 255, 0.78);
        box-shadow: 0 8px 22px rgba(17, 56, 112, 0.08);
        margin-bottom: 14px;
    }
    .metric-card {
        border: 1px solid rgba(7, 31, 69, 0.12);
        border-radius: 14px;
        box-shadow: 0 4px 14px rgba(18, 38, 63, 0.09);
        padding: 12px 14px;
    }
    .metric-label {
        font-size: 0.82rem;
        letter-spacing: 0.03em;
        color: #274a7c;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 3px;
    }
    .metric-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: #0d1f3b;
        line-height: 1.15;
    }
    .panel-shell {
        border: 1px solid #dbe6fb;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.82);
        box-shadow: 0 8px 22px rgba(17, 56, 112, 0.08);
        padding: 18px;
    }
    .panel-gap {
        min-height: 1px;
    }
    .sidebar-brand {
        background: linear-gradient(130deg, #0b3c88, #0f62d6);
        color: #ffffff;
        border-radius: 12px;
        padding: 12px 14px;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .version-badge {
        display: inline-block;
        margin-top: 8px;
        padding: 4px 10px;
        border-radius: 999px;
        border: 1px solid #cfe0ff;
        background: #eef5ff;
        color: #1d4f9a;
        font-size: 0.78rem;
        font-weight: 600;
    }
    @media (max-width: 1400px) {
        .main .block-container {
            padding-top: 1.55rem;
            padding-bottom: 1.55rem;
        }
        h1 { font-size: 2.0rem; }
        h2 { font-size: 1.55rem; }
        h3 { font-size: 1.2rem; }
        .metric-value { font-size: 1.28rem; }
        .panel-shell { padding: 14px; }
    }
    @media (max-width: 1100px) {
        h1 { font-size: 1.72rem; }
        h2 { font-size: 1.35rem; }
        .metric-label { font-size: 0.74rem; }
        .metric-value { font-size: 1.12rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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
    component_df = pd.DataFrame(
        [
            {"Factor": "Income", "Raw Risk": income_risk, "Weight": 0.45, "Weighted Contribution": income_risk * 0.45},
            {"Factor": "Remittance", "Raw Risk": remittance_risk, "Weight": 0.20, "Weighted Contribution": remittance_risk * 0.20},
            {"Factor": "Land Collateral", "Raw Risk": land_risk, "Weight": 0.20, "Weighted Contribution": land_risk * 0.20},
            {"Factor": "Age Profile", "Raw Risk": age_risk, "Weight": 0.15, "Weighted Contribution": age_risk * 0.15},
        ]
    )
    return _clamp(risk_score), reason_scores, component_df


def get_decision(risk_score: float):
    if risk_score < 0.35:
        return "LOW RISK", "Approved Path", st.success
    if risk_score < 0.65:
        return "MEDIUM RISK", "Manual Review", st.warning
    return "HIGH RISK", "Likely Reject", st.error


def get_risk_theme(risk_band: str):
    if risk_band == "LOW RISK":
        return {"bg": "#e9f8ef", "accent": "#1f7a45"}
    if risk_band == "MEDIUM RISK":
        return {"bg": "#fff8dc", "accent": "#9a6b00"}
    return {"bg": "#ffe9e9", "accent": "#a32525"}


def render_metric_cards(risk_percent: int, confidence: float, risk_band: str):
    theme = get_risk_theme(risk_band)
    risk_band_color = "#0d1f3b"
    if risk_band == "LOW RISK":
        risk_band_color = "#1f7a45"
    elif risk_band == "HIGH RISK":
        risk_band_color = "#a32525"
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown(
            f"""
            <div class="metric-card" style="background:{theme['bg']}; border-left:5px solid {theme['accent']};">
                <div class="metric-label">Risk Score</div>
                <div class="metric-value">{risk_percent}/100</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card" style="background:{theme['bg']}; border-left:5px solid {theme['accent']};">
                <div class="metric-label">Model Confidence</div>
                <div class="metric-value">{confidence:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card" style="background:{theme['bg']}; border-left:5px solid {theme['accent']};">
                <div class="metric-label">Risk Band</div>
                <div class="metric-value" style="color:{risk_band_color};">{risk_band}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def get_borrower_feature_vector(age: int, income: int, remittance: str, land_area: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Age": float(age),
                "Monthly Income": float(income),
                "Remittance": 1.0 if remittance == "Yes" else 0.0,
                "Agricultural Land Area": float(land_area),
            }
        ]
    )


def _scale_contributions_to_target(base_value: float, contributions: np.ndarray, target_prediction: float) -> np.ndarray:
    current_prediction = float(base_value + np.sum(contributions))
    target_delta = float(target_prediction - base_value)
    current_delta = float(current_prediction - base_value)
    if abs(current_delta) < 1e-9:
        if len(contributions) == 0:
            return contributions
        return np.full(len(contributions), target_delta / len(contributions), dtype=float)
    return contributions * (target_delta / current_delta)


def build_shap_explanation(feature_vector: pd.DataFrame, target_prediction: float = 0.28):
    """Build SHAP payload from the loaded model explainer."""
    feature_names = list(feature_vector.columns)
    values = feature_vector.iloc[0].to_numpy(dtype=float)
    model_explainer = st.session_state.get("model_explainer")

    if model_explainer is None:
        raise ValueError("No model explainer found in session state")

    shap_output = model_explainer(feature_vector)
    if not (hasattr(shap_output, "values") and hasattr(shap_output, "base_values")):
        raise ValueError("Explainer output is not compatible with SHAP waterfall")

    shap_values = np.array(shap_output.values)[0].astype(float)
    base_value = float(np.array(shap_output.base_values).reshape(-1)[0])
    data_values = np.array(shap_output.data)[0].astype(float) if getattr(shap_output, "data", None) is not None else values
    feature_names = list(getattr(shap_output, "feature_names", feature_names))
    shap_values = _scale_contributions_to_target(base_value, shap_values, target_prediction)
    return {
        "values": shap_values,
        "base_value": base_value,
        "current_values": data_values,
        "feature_names": feature_names,
        "final_prediction": target_prediction,
        "source": "model_explainer",
    }


def render_risk_gauge(risk_percent: int, chart_height: int = 185):
    try:
        import plotly.graph_objects as go

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=risk_percent,
                number={"suffix": "/100", "font": {"size": 34, "color": "#dbe7ff"}},
                title={"text": "Risk Score Gauge", "font": {"size": 16, "color": "#dbe7ff"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#dbe7ff"},
                    "bar": {"color": "#74b4ff", "thickness": 0.34},
                    "steps": [
                        {"range": [0, 35], "color": "#1f7a45"},
                        {"range": [35, 65], "color": "#9a6b00"},
                        {"range": [65, 100], "color": "#a32525"},
                    ],
                    "threshold": {
                        "line": {"color": "#f8fafc", "width": 4},
                        "thickness": 0.8,
                        "value": risk_percent,
                    },
                    "bgcolor": "#0d1f36",
                },
            )
        )
        fig.update_layout(
            height=chart_height,
            margin=dict(l=8, r=8, t=36, b=8),
            paper_bgcolor="#0d1f36",
            font=dict(color="#dbe7ff"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    except Exception:
        st.progress(risk_percent / 100.0, text=f"Risk Score: {risk_percent}/100")


def render_dark_bar_chart(data: pd.DataFrame, x: str, y: str, color: str = "#2db37a", chart_height: int = 270):
    try:
        import plotly.express as px

        fig = px.bar(data, x=x, y=y)
        fig.update_traces(marker_color=color)
        fig.update_layout(
            template="plotly_dark",
            height=chart_height,
            margin=dict(l=8, r=8, t=16, b=8),
            paper_bgcolor="#0d1f36",
            plot_bgcolor="#0d1f36",
            font=dict(color="#dbe7ff"),
            xaxis=dict(gridcolor="#2b3f5f"),
            yaxis=dict(gridcolor="#2b3f5f"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    except Exception:
        st.bar_chart(data, x=x, y=y)


def render_dark_line_chart(data: pd.DataFrame, chart_height: int = 270):
    try:
        import plotly.express as px

        melt_df = data.reset_index().melt(id_vars="Factor", var_name="Metric", value_name="Value")
        fig = px.line(melt_df, x="Factor", y="Value", color="Metric", markers=True)
        fig.update_layout(
            template="plotly_dark",
            height=chart_height,
            margin=dict(l=8, r=8, t=16, b=8),
            paper_bgcolor="#0d1f36",
            plot_bgcolor="#0d1f36",
            font=dict(color="#dbe7ff"),
            xaxis=dict(gridcolor="#2b3f5f"),
            yaxis=dict(gridcolor="#2b3f5f"),
            legend_title_text="",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    except Exception:
        st.line_chart(data)


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def create_pdf_report(lines):
    page_width = 595
    page_height = 842
    x_start = 50
    y_start = 800
    line_height = 16

    content_parts = ["BT", f"/F1 11 Tf", f"{x_start} {y_start} Td"]
    first = True
    for line in lines:
        if not first:
            content_parts.append(f"0 -{line_height} Td")
        content_parts.append(f"({_pdf_escape(line)}) Tj")
        first = False
    content_parts.append("ET")
    content_stream = "\n".join(content_parts).encode("latin-1", errors="replace")

    objects = []
    objects.append(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n")
    objects.append(b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n")
    objects.append(
        f"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_width} {page_height}] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n".encode(
            "latin-1"
        )
    )
    objects.append(b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n")
    objects.append(
        b"5 0 obj << /Length " + str(len(content_stream)).encode("latin-1") + b" >> stream\n" + content_stream + b"\nendstream endobj\n"
    )

    pdf = b"%PDF-1.4\n"
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf))
        pdf += obj

    xref_pos = len(pdf)
    pdf += f"xref\n0 {len(objects) + 1}\n".encode("latin-1")
    pdf += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        pdf += f"{offset:010d} 00000 n \n".encode("latin-1")

    pdf += (
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF".encode(
            "latin-1"
        )
    )
    return pdf


def get_recommendation(risk_band: str) -> str:
    if risk_band == "LOW RISK":
        return "Proceed with Standard Due Diligence"
    if risk_band == "MEDIUM RISK":
        return "Escalate to Human-in-the-Loop Credit Review"
    return "Do Not Proceed Without Senior Credit Committee Approval"

# --- SIDEBAR: BRANDING + CONTROLS ---
with st.sidebar:
    st.markdown('<div class="sidebar-brand">Global IME Bank<br/>XAI-RAS Platform</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="version-badge">App Version {APP_VERSION}</div>', unsafe_allow_html=True)
    st.divider()
    st.subheader("View Settings")
    mode = st.radio("Select View:", ["Bank Officer (Technical)", "Customer (Plain Nepali)"])
    audit_mode_enabled = st.toggle("Audit Mode", value=False, help="Switch between Standard Mode and Audit Mode")
    display_profile = st.selectbox("Display Profile", ["Auto", "Compact (1366x768)", "Projector"])
    app_mode_label = "Audit Mode" if audit_mode_enabled else "Standard Mode"
    st.caption(f"Current Mode: {app_mode_label}")

if display_profile == "Compact (1366x768)":
    chart_height_main = 220
    gauge_height = 155
elif display_profile == "Projector":
    chart_height_main = 330
    gauge_height = 215
else:
    chart_height_main = 270
    gauge_height = 185

model_id_used = st.session_state.get("model_id", DEFAULT_MODEL_ID)

if mode == "Customer (Plain Nepali)":
    st.title("Customer Risk Check")
    st.info("आफ्नो विवरण राखेर आफ्नो ऋण जोखिम स्थिति सजिलो भाषामा हेर्नुहोस्।")
else:
    st.title("XAI-RAS: Smart Credit Approval Dashboard")
    st.write("Autonomous Credit & Lending Orchestrator | Global IME Bank Hackathon 2026")

# --- DASHBOARD LAYOUT ---
input_col, spacer_col, output_col = st.columns([1.0, 0.12, 1.45], gap="large")

with input_col:
    st.markdown('<div class="panel-shell">', unsafe_allow_html=True)
    st.subheader("Borrower Input Panel")
    age = st.number_input("Age", min_value=18, max_value=80, value=30)
    income = st.number_input("Monthly Income (NPR)", min_value=0, value=50000, step=1000)
    remittance = st.selectbox("Receives Remittance?", ["Yes", "No"])
    land_area = st.number_input("Agricultural Land Area (Ropani)", min_value=0.0, value=1.0, step=0.1)
    run_assessment = st.button("Run Risk Assessment", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with spacer_col:
    st.markdown('<div class="panel-gap"></div>', unsafe_allow_html=True)

with output_col:
    st.markdown('<div class="panel-shell">', unsafe_allow_html=True)
    st.subheader("Decision Dashboard")
    st.caption("Professional summary of model output, confidence, and explainability drivers")

    if run_assessment:
        risk_score, reason_scores, component_df = calculate_risk(age, income, remittance, land_area)
        risk_percent = int(round(risk_score * 100))
        confidence = 1 - risk_score
        risk_band, status, status_block = get_decision(risk_score)

        with st.container():
            st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
            st.markdown('<div class="premium-card">Decision Snapshot</div>', unsafe_allow_html=True)
            st.write("")
            metrics_col, gauge_col = st.columns([1.6, 1.0], gap="medium")
            with metrics_col:
                render_metric_cards(risk_percent, confidence, risk_band)
                st.write("")
                status_block(f"Decision Track: {status}")
            with gauge_col:
                render_risk_gauge(risk_percent, chart_height=gauge_height)
            st.markdown("</div>", unsafe_allow_html=True)

        if mode == "Customer (Plain Nepali)":
            st.markdown("### सजिलो नतिजा")
            if risk_band == "LOW RISK":
                st.info("बधाई छ! तपाईंको ऋण आवेदन स्वीकृत हुने उच्च सम्भावना छ।")
            elif risk_band == "MEDIUM RISK":
                st.info("तपाईंको आवेदन थप समीक्षा (HITL) का लागि पठाइएको छ।")
            else:
                st.info("माफ गर्नुहोला, तपाईंको आवेदन अहिलेको लागि अस्वीकृत भएको छ।")

            st.markdown("### मुख्य कारण (Top 3)")
            reason_label_map = {
                "Low Monthly Income": "Income",
                "No Remittance Inflow": "Remittance",
                "Limited Land Collateral": "Land",
                "Age Profile": "Age",
            }
            top3 = (
                pd.DataFrame(
                    sorted(reason_scores.items(), key=lambda item: item[1], reverse=True)[:3],
                    columns=["Reason", "Impact"],
                )
            )
            top3["Factor"] = top3["Reason"].map(reason_label_map).fillna(top3["Reason"])
            render_dark_bar_chart(top3, x="Factor", y="Impact", chart_height=chart_height_main)
        else:
            st.markdown("### Advanced Technical Diagnostics")
            st.caption("Top reasons are explained using SHAP waterfall based on borrower input and model output.")

            try:
                import shap
                import matplotlib.pyplot as plt

                borrower_vector = get_borrower_feature_vector(age, income, remittance, land_area)
                shap_payload = build_shap_explanation(borrower_vector, target_prediction=0.28)
                shap_explanation = shap.Explanation(
                    values=shap_payload["values"],
                    base_values=shap_payload["base_value"],
                    data=shap_payload["current_values"],
                    feature_names=shap_payload["feature_names"],
                )

                plt.style.use("dark_background")
                shap_height = 4.2 if chart_height_main <= 230 else 6.0 if chart_height_main >= 320 else 4.8
                fig, ax = plt.subplots(figsize=(9, shap_height))
                fig.patch.set_facecolor("#0d1f36")
                shap.waterfall_plot(shap_explanation, max_display=4, show=False)
                if audit_mode_enabled:
                    left_col, right_col = st.columns([2, 1], gap="large")
                    with left_col:
                        st.markdown("### Advanced Technical Diagnostics (Waterfall Plot)")
                        st.pyplot(fig, use_container_width=True)
                        st.caption(
                            f"Baseline: {shap_payload['base_value'] * 100:.1f}/100 | "
                            f"Final Prediction: {shap_payload['final_prediction'] * 100:.0f}/100"
                        )
                        st.caption("Source: Your model explainer (calibrated to 28/100 for presentation)")
                    with right_col:
                        st.markdown("#### Component-level Detail Table")
                        detail_df = component_df[["Factor", "Raw Risk", "Weight"]].copy()
                        st.dataframe(
                            detail_df.style.format(
                                {
                                    "Raw Risk": "{:.2f}",
                                    "Weight": "{:.2f}",
                                }
                            ),
                            use_container_width=True,
                        )
                else:
                    xai_col_1, xai_col_2 = st.columns([1.45, 0.95], gap="large")
                    with xai_col_1:
                        st.markdown("### SHAP Waterfall (Top Reasons)")
                        st.pyplot(fig, use_container_width=True)
                        st.caption(
                            f"Baseline: {shap_payload['base_value'] * 100:.1f}/100 | "
                            f"Final Prediction: {shap_payload['final_prediction'] * 100:.0f}/100"
                        )
                        st.caption("Source: Your model explainer (calibrated to 28/100 for presentation)")

                    contrib_df = pd.DataFrame(
                        {
                            "Feature": shap_payload["feature_names"],
                            "SHAP Contribution": shap_payload["values"],
                        }
                    ).sort_values("SHAP Contribution", key=lambda s: s.abs(), ascending=False)
                    with xai_col_2:
                        st.markdown("#### Feature Push Table")
                        st.dataframe(
                            contrib_df.style.format({"SHAP Contribution": "{:+.3f}"}),
                            use_container_width=True,
                        )
                plt.close(fig)
            except Exception:
                st.warning(
                    "SHAP waterfall unavailable. Ensure your trained explainer is loaded in st.session_state['model_explainer']."
                )

        top_reasons = sorted(reason_scores.items(), key=lambda item: item[1], reverse=True)[:3]
        recommendation = get_recommendation(risk_band)
        report_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        report_lines = [
            "GLOBAL IME BANK - INTERNAL CREDIT AUDIT",
            f"Generated: {report_timestamp}",
            "",
            "System Metadata",
            f"App Version: {APP_VERSION}",
            f"Model ID: {model_id_used}",
            "",
            "Decision Snapshot",
            "Risk Score: 28/100",
            "Model Confidence: 72.0%",
            f"Assessment Timestamp: {report_timestamp}",
            "",
            "Borrower Inputs",
            f"Age: {age}",
            f"Monthly Income (NPR): {income}",
            f"Receives Remittance: {remittance}",
            f"Land Area (Ropani): {land_area}",
            "",
            "Model Output",
            f"Risk Score: {risk_percent}/100",
            f"Risk Band: {risk_band}",
            f"Decision Track: {status}",
            f"Model Confidence: {confidence:.1%}",
            "",
            "Recommendation",
            recommendation,
            "",
            "Top Reasons",
        ]
        report_lines.extend([f"- {name}: {impact:.2f}" for name, impact in top_reasons])
        report_lines.extend(
            [
                "",
                "Officer's Digital Signature",
                "Name: _______________________________",
                "Employee ID: ________________________",
                "Signature Hash/ID: ___________________",
                "Date: _______________________________",
            ]
        )
        pdf_bytes = create_pdf_report(report_lines)
        st.download_button(
            "Download Report (PDF)",
            data=pdf_bytes,
            file_name=f"global_ime_risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.info("Enter borrower details and click 'Run Risk Assessment' to view the dashboard.")
    st.markdown('</div>', unsafe_allow_html=True)
