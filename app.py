import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime

APP_VERSION = "1.4.0"
DEFAULT_MODEL_ID = "XAI-RAS-MOCK-RULES-v1"
FEATURE_COLUMNS = ["Age", "Monthly Income", "Remittance", "Agricultural Land Area"]
FEATURE_NAME_MAP = {
    "Monthly Income": "Income",
    "Remittance": "Remittance",
    "Agricultural Land Area": "Land",
    "Age": "Age",
}
RISK_WEIGHTS = {
    "Income": 0.45,
    "Remittance": 0.20,
    "Land": 0.20,
    "Age": 0.15,
}
INCOME_REFERENCE = 70000.0
LAND_REFERENCE = 2.5
AGE_REFERENCE = 35.0

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


def _income_risk(income: int) -> float:
    return _clamp((INCOME_REFERENCE - float(income)) / INCOME_REFERENCE)


def _remittance_risk(remittance: str) -> float:
    return 0.0 if remittance == "Yes" else 1.0


def _land_risk(land_area: float) -> float:
    return _clamp((LAND_REFERENCE - float(land_area)) / LAND_REFERENCE)


def _age_risk(age: int) -> float:
    return _clamp(abs(float(age) - AGE_REFERENCE) / AGE_REFERENCE)


def calculate_risk(age: int, income: int, remittance: str, land_area: float):
    """Weighted, explainable score using the local rules engine."""
    income_risk = _income_risk(income)
    remittance_risk = _remittance_risk(remittance)
    land_risk = _land_risk(land_area)
    age_risk = _age_risk(age)

    weighted_income = income_risk * RISK_WEIGHTS["Income"]
    weighted_remittance = remittance_risk * RISK_WEIGHTS["Remittance"]
    weighted_land = land_risk * RISK_WEIGHTS["Land"]
    weighted_age = age_risk * RISK_WEIGHTS["Age"]

    reason_scores = {
        "Low Monthly Income": weighted_income,
        "No Remittance Inflow": weighted_remittance,
        "Limited Land Collateral": weighted_land,
        "Age Profile": weighted_age,
    }

    risk_score = weighted_income + weighted_remittance + weighted_land + weighted_age
    component_df = pd.DataFrame(
        [
            {"Factor": "Income", "Raw Risk": income_risk, "Weight": RISK_WEIGHTS["Income"], "Weighted Contribution": weighted_income},
            {"Factor": "Remittance", "Raw Risk": remittance_risk, "Weight": RISK_WEIGHTS["Remittance"], "Weighted Contribution": weighted_remittance},
            {"Factor": "Land Collateral", "Raw Risk": land_risk, "Weight": RISK_WEIGHTS["Land"], "Weighted Contribution": weighted_land},
            {"Factor": "Age Profile", "Raw Risk": age_risk, "Weight": RISK_WEIGHTS["Age"], "Weighted Contribution": weighted_age},
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
        ],
        columns=FEATURE_COLUMNS,
    )


@dataclass
class MockShapOutput:
    values: np.ndarray
    base_values: float
    data: np.ndarray
    feature_names: list[str]


@dataclass
class MockShapExplanation:
    values: np.ndarray
    base_values: float
    data: np.ndarray
    feature_names: list[str]


class LendingModel:
    """Rule-based lending model that uses the configured borrower weights."""

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = dict(weights or RISK_WEIGHTS)

    def predict_row(self, row: pd.Series) -> float:
        income_risk = _income_risk(int(row["Monthly Income"]))
        remittance_risk = _remittance_risk("Yes" if float(row["Remittance"]) >= 0.5 else "No")
        land_risk = _land_risk(float(row["Agricultural Land Area"]))
        age_risk = _age_risk(int(row["Age"]))

        score = (
            income_risk * self.weights["Income"]
            + remittance_risk * self.weights["Remittance"]
            + land_risk * self.weights["Land"]
            + age_risk * self.weights["Age"]
        )
        return _clamp(score)

    def predict(self, feature_vector: pd.DataFrame) -> np.ndarray:
        return np.array([self.predict_row(row) for _, row in feature_vector.iterrows()], dtype=float)

    def predict_proba(self, feature_vector: pd.DataFrame) -> np.ndarray:
        scores = self.predict(feature_vector)
        return np.column_stack([1.0 - scores, scores])


class MockShapExplainer:
    """Local explainer that mirrors the SHAP output structure."""

    def __init__(self, model: LendingModel):
        self.model = model

    def __call__(self, feature_vector: pd.DataFrame) -> MockShapOutput:
        row = feature_vector.iloc[0]
        _ = self.model.predict(feature_vector)[0]
        income_risk = _income_risk(int(row["Monthly Income"]))
        remittance_risk = _remittance_risk("Yes" if float(row["Remittance"]) >= 0.5 else "No")
        land_risk = _land_risk(float(row["Agricultural Land Area"]))
        age_risk = _age_risk(int(row["Age"]))
        weights = self.model.weights
        shap_values = np.array(
            [
                income_risk * weights["Income"],
                remittance_risk * weights["Remittance"],
                land_risk * weights["Land"],
                age_risk * weights["Age"],
            ],
            dtype=float,
        )
        return MockShapOutput(
            values=shap_values,
            base_values=0.0,
            data=row.to_numpy(dtype=float).reshape(1, -1),
            feature_names=list(feature_vector.columns),
        )


class MockShapModule:
    Explainer = MockShapExplainer
    Explanation = MockShapExplanation

    @staticmethod
    def waterfall_plot(explanation, max_display: int = 4, show: bool = True):
        import matplotlib.pyplot as plt

        values = np.array(getattr(explanation, "values", []), dtype=float).reshape(-1)
        feature_names = list(getattr(explanation, "feature_names", []))
        data = np.array(getattr(explanation, "data", []), dtype=float).reshape(-1)

        if not feature_names:
            feature_names = [f"Feature {index + 1}" for index in range(len(values))]

        order = np.argsort(np.abs(values))[::-1][:max_display]
        selected_values = values[order]
        selected_names = [feature_names[index] for index in order]
        selected_data = [data[index] if index < len(data) else np.nan for index in order]
        colors = ["#1f7a45" if value >= 0 else "#a32525" for value in selected_values]

        ax = plt.gca()
        ax.barh(range(len(selected_values)), selected_values[::-1], color=colors[::-1], alpha=0.95)
        ax.set_yticks(range(len(selected_names)))
        ax.set_yticklabels(
            [
                f"{FEATURE_NAME_MAP.get(name, name)} = {value:g}"
                for name, value in zip(selected_names[::-1], selected_data[::-1])
            ],
            color="#dbe7ff",
        )
        ax.axvline(0, color="#dbe7ff", linewidth=1.0)
        ax.set_facecolor("#0d1f36")
        ax.figure.patch.set_facecolor("#0d1f36")
        ax.tick_params(axis="x", colors="#dbe7ff")
        ax.tick_params(axis="y", colors="#dbe7ff")
        ax.set_title("SHAP Waterfall", color="#dbe7ff")
        ax.set_xlabel("Contribution", color="#dbe7ff")
        if show:
            plt.show()
        return ax


def build_shap_explanation(feature_vector: pd.DataFrame, target_prediction: float | None = None):
    """Build SHAP payload from the local mock explainer."""
    feature_names = list(feature_vector.columns)
    values = feature_vector.iloc[0].to_numpy(dtype=float)
    model_explainer = st.session_state.get("model_explainer")

    if model_explainer is None:
        raise ValueError("No model explainer found in session state")

    shap_output = model_explainer(feature_vector)
    if not (hasattr(shap_output, "values") and hasattr(shap_output, "base_values")):
        raise ValueError("Explainer output is not compatible with SHAP waterfall")

    shap_array = np.array(shap_output.values)
    if shap_array.ndim == 3:
        shap_values = shap_array[0, :, -1].astype(float)
    else:
        shap_values = shap_array.reshape(-1).astype(float)

    base_array = np.array(shap_output.base_values)
    base_value = float(base_array.reshape(-1)[-1])
    data_values = np.array(shap_output.data)[0].astype(float) if getattr(shap_output, "data", None) is not None else values
    raw_names = list(getattr(shap_output, "feature_names", feature_names))
    feature_names = [FEATURE_NAME_MAP.get(name, name) for name in raw_names]
    final_prediction = float(base_value + np.sum(shap_values))
    return {
        "values": shap_values,
        "base_value": base_value,
        "current_values": data_values,
        "feature_names": feature_names,
        "final_prediction": final_prediction,
        "source": "mock_explainer",
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


def initialize_model_and_explainer():
    """Initialize the local mock explainer once per session."""
    if st.session_state.get("model_explainer") is not None and st.session_state.get("model") is not None:
        return

    st.session_state["model"] = LendingModel()
    st.session_state["model_explainer"] = MockShapExplainer(st.session_state["model"])
    st.session_state["model_id"] = DEFAULT_MODEL_ID
    st.session_state.pop("model_explainer_error", None)

# --- SIDEBAR: BRANDING + CONTROLS ---
initialize_model_and_explainer()

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
                try:
                    import shap as shap_module
                except Exception:
                    shap_module = MockShapModule()
                import matplotlib.pyplot as plt

                borrower_vector = get_borrower_feature_vector(age, income, remittance, land_area)
                shap_payload = build_shap_explanation(borrower_vector, target_prediction=0.28)
                shap_explanation = shap_module.Explanation(
                    values=shap_payload["values"],
                    base_values=shap_payload["base_value"],
                    data=shap_payload["current_values"],
                    feature_names=shap_payload["feature_names"],
                )

                plt.style.use("dark_background")
                shap_height = 4.2 if chart_height_main <= 230 else 6.0 if chart_height_main >= 320 else 4.8
                fig, ax = plt.subplots(figsize=(9, shap_height))
                fig.patch.set_facecolor("#0d1f36")
                shap_module.waterfall_plot(shap_explanation, max_display=4, show=False)
                if audit_mode_enabled:
                    left_col, right_col = st.columns([2, 1], gap="large")
                    with left_col:
                        st.markdown("### Advanced Technical Diagnostics (Waterfall Plot)")
                        st.pyplot(fig, use_container_width=True)
                        st.caption(
                            f"Baseline: {shap_payload['base_value'] * 100:.1f}/100 | "
                            f"Final Prediction: {shap_payload['final_prediction'] * 100:.0f}/100"
                        )
                        st.caption("Source: Local mock explainer using Income 45%, Remittance 20%, Land 20%, Age 15%")
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
                        st.caption("Source: Local mock explainer using Income 45%, Remittance 20%, Land 20%, Age 15%")

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
            except Exception as exc:
                st.session_state["model_explainer_error"] = str(exc)
                explainer_error = st.session_state.get("model_explainer_error", "Unknown error")
                st.warning(
                    f"SHAP waterfall unavailable. {explainer_error}"
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
            f"Risk Score: {risk_percent}/100",
            f"Model Confidence: {confidence:.1%}",
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