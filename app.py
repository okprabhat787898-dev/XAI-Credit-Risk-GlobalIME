import streamlit as st
import pandas as pd
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="XAI-RAS: Global IME Bank", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top right, #f4f9ff 0%, #e9f2ff 35%, #f8fbff 65%, #ffffff 100%);
    }
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


def render_risk_gauge(risk_percent: int):
    try:
        import plotly.graph_objects as go

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=risk_percent,
                number={"suffix": "/100", "font": {"size": 42}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#0f62d6", "thickness": 0.35},
                    "steps": [
                        {"range": [0, 35], "color": "#d6f5e8"},
                        {"range": [35, 65], "color": "#fff4cc"},
                        {"range": [65, 100], "color": "#ffe0e0"},
                    ],
                    "threshold": {
                        "line": {"color": "#0b3c88", "width": 4},
                        "thickness": 0.75,
                        "value": risk_percent,
                    },
                },
                title={"text": "Risk Gauge", "font": {"size": 24}},
            )
        )
        fig.update_layout(height=330, margin=dict(l=15, r=15, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.progress(risk_percent / 100.0, text=f"Risk Score: {risk_percent}/100")


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

# --- CUSTOMER/OFFICER TOGGLE (From your paper's Dual-Audience requirement) ---
st.sidebar.title("View Mode")
mode = st.sidebar.radio("Select View:", ["Bank Officer (Technical)", "Customer (Plain Nepali)"])

if mode == "Customer (Plain Nepali)":
    st.title("Customer Risk Check")
    st.info("आफ्नो विवरण राखेर आफ्नो ऋण जोखिम स्थिति सजिलो भाषामा हेर्नुहोस्।")
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
        risk_score, reason_scores, component_df = calculate_risk(age, income, remittance, land_area)
        risk_percent = int(round(risk_score * 100))
        confidence = 1 - risk_score
        risk_band, status, status_block = get_decision(risk_score)

        chart_col, metrics_col = st.columns([1.2, 1.0], gap="large")
        with chart_col:
            render_risk_gauge(risk_percent)
        with metrics_col:
            st.markdown('<div class="premium-card">Decision Snapshot</div>', unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            m1.metric("Risk Score", f"{risk_percent}/100")
            m2.metric("Model Confidence", f"{confidence:.1%}")
            st.metric("Risk Band", risk_band)
            status_block(f"Decision Track: {status}")

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
            st.bar_chart(top3, x="Factor", y="Impact")
        else:
            st.markdown("### Advanced Technical Diagnostics")
            tech_col_1, tech_col_2 = st.columns(2, gap="large")
            with tech_col_1:
                st.caption("Weighted contributions to final risk score")
                weighted = component_df[["Factor", "Weighted Contribution"]].set_index("Factor")
                st.bar_chart(weighted)
            with tech_col_2:
                st.caption("Raw risk vs weighted contribution")
                compare = component_df[["Factor", "Raw Risk", "Weighted Contribution"]].set_index("Factor")
                st.line_chart(compare)

            st.caption("Component-level detail table")
            st.dataframe(
                component_df.style.format(
                    {
                        "Raw Risk": "{:.2f}",
                        "Weight": "{:.2f}",
                        "Weighted Contribution": "{:.2f}",
                    }
                ),
                use_container_width=True,
            )

        top_reasons = sorted(reason_scores.items(), key=lambda item: item[1], reverse=True)[:3]
        report_lines = [
            "Global IME - XAI-RAS Risk Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
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
            "Top Reasons",
        ]
        report_lines.extend([f"- {name}: {impact:.2f}" for name, impact in top_reasons])
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
