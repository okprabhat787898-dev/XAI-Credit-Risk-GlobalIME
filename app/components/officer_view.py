"""Technical (bank-officer) view for XAI-RAS dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from explainer.shap_explainer import SHAPExplainer


def render_prediction_card(
    default_prob: float,
    decision: str,
    risk_tier: str,
) -> None:
    """Display prediction result card."""
    colour = {"APPROVE": "green", "REVIEW": "orange", "REJECT": "red"}.get(
        decision, "grey"
    )
    st.markdown(
        f"""
        <div style="
            border: 2px solid {colour};
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
            margin-bottom: 1rem;
        ">
            <h2 style="color:{colour}; margin:0">{decision}</h2>
            <p style="margin:0">Default Probability: <b>{default_prob:.1%}</b></p>
            <p style="margin:0">Risk Tier: <b>{risk_tier}</b></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_shap_waterfall(local_df: pd.DataFrame, base_value: float) -> None:
    """Render a waterfall chart of local SHAP values."""
    df = local_df.copy()
    df["color"] = df["shap_value"].apply(lambda v: "#d62728" if v > 0 else "#2ca02c")

    fig = go.Figure(
        go.Waterfall(
            orientation="h",
            measure=["relative"] * len(df),
            y=df["feature"],
            x=df["shap_value"],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "#2ca02c"}},
            increasing={"marker": {"color": "#d62728"}},
            totals={"marker": {"color": "#1f77b4"}},
        )
    )
    fig.update_layout(
        title="Local SHAP Waterfall (Top Factors)",
        xaxis_title="SHAP Value (impact on default probability)",
        yaxis_title="Feature",
        height=350,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_global_importance(importance_df: pd.DataFrame) -> None:
    """Render global mean |SHAP| bar chart."""
    fig = px.bar(
        importance_df.head(15).sort_values("mean_abs_shap"),
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        title="Global Feature Importance (Mean |SHAP|)",
        labels={
            "mean_abs_shap": "Mean |SHAP| Value",
            "feature": "Feature",
        },
        color="mean_abs_shap",
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        height=450,
        margin=dict(l=10, r=10, t=40, b=10),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_portfolio_metrics(df: pd.DataFrame, pipeline) -> None:
    """Show portfolio-level risk metrics for a batch of applications."""
    probs = pipeline.predict_proba(df)[:, 1]
    pred_labels = pipeline.predict(df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Applications", len(df))
    col2.metric("Predicted Defaults", int(pred_labels.sum()))
    col3.metric("Approval Rate", f"{(pred_labels == 0).mean():.1%}")
    col4.metric("Avg. Default Probability", f"{probs.mean():.1%}")

    risk_bins = pd.cut(
        probs,
        bins=[0, 0.2, 0.4, 0.6, 1.0],
        labels=["Low (<20%)", "Medium (20-40%)", "High (40-60%)", "Very High (>60%)"],
    )
    dist = risk_bins.value_counts().reset_index()
    dist.columns = ["Risk Tier", "Count"]
    fig = px.pie(
        dist,
        names="Risk Tier",
        values="Count",
        title="Portfolio Risk Distribution",
        color="Risk Tier",
        color_discrete_map={
            "Low (<20%)": "#2ca02c",
            "Medium (20-40%)": "#ffbb33",
            "High (40-60%)": "#ff8800",
            "Very High (>60%)": "#d62728",
        },
    )
    st.plotly_chart(fig, use_container_width=True)


def render_officer_view(
    application: pd.DataFrame,
    pipeline,
    explainer: SHAPExplainer,
    portfolio_df: pd.DataFrame | None = None,
) -> None:
    """Render the full bank-officer technical view.

    Parameters
    ----------
    application:
        Single-row DataFrame for the loan application to analyse.
    pipeline:
        Fitted stacking pipeline.
    explainer:
        Fitted SHAPExplainer instance.
    portfolio_df:
        Optional batch DataFrame for portfolio analytics.
    """
    default_prob = float(pipeline.predict_proba(application)[0, 1])

    if default_prob < 0.30:
        decision, risk_tier = "APPROVE", "Low Risk"
    elif default_prob < 0.50:
        decision, risk_tier = "REVIEW", "Medium Risk"
    else:
        decision, risk_tier = "REJECT", "High Risk"

    st.subheader("🏦 Bank Officer – Technical View")
    render_prediction_card(default_prob, decision, risk_tier)

    local_df = explainer.local_explanation(application, top_n=8)
    base_val = explainer.explain(application)["base_value"]

    col_left, col_right = st.columns(2)
    with col_left:
        render_shap_waterfall(local_df, base_val)
    with col_right:
        st.markdown("#### Top Risk Factors")
        for _, row in local_df.iterrows():
            icon = "⬆️" if row["direction"] == "increases_risk" else "⬇️"
            st.write(
                f"{icon} **{row['feature']}**: SHAP = `{row['shap_value']:.4f}`"
            )

    with st.expander("🔬 Application Details"):
        st.dataframe(application.T.rename(columns={application.index[0]: "Value"}))

    if portfolio_df is not None:
        st.markdown("---")
        st.subheader("📊 Portfolio Analytics")
        render_portfolio_metrics(portfolio_df, pipeline)

    # NRB compliance note
    st.info(
        "**NRB AI Guidelines 2025 – Compliance Note:** "
        "This decision is fully explainable via SHAP values. "
        "All protected attributes (caste, gender, religion) are excluded from the model. "
        "An audit trail is maintained for regulatory review."
    )
