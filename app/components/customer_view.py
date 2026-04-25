"""Customer-facing view with plain Nepali explanations for XAI-RAS."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from explainer.shap_explainer import SHAPExplainer

# ---------------------------------------------------------------------------
# Plain-language Nepali templates
# ---------------------------------------------------------------------------
_APPROVE_MSG = """
## ✅ तपाईंको ऋण आवेदन **स्वीकृत** भयो!

बधाई छ! Global IME Bank ले तपाईंको ऋण आवेदन स्वीकृत गर्न सिफारिस गरेको छ।

**कारण:**
"""

_REVIEW_MSG = """
## ⏳ तपाईंको ऋण आवेदन **समीक्षाधीन** छ।

तपाईंको आवेदन थप समीक्षाका लागि पठाइएको छ। बैंक अधिकारी चाँडै सम्पर्क गर्नेछन्।

**मुख्य कारण:**
"""

_REJECT_MSG = """
## ❌ तपाईंको ऋण आवेदन **अस्वीकृत** भयो।

माफ गर्नुस्! यस पटक तपाईंको ऋण आवेदन स्वीकृत हुन सकेन।

**मुख्य कारण:**
"""

_DIRECTION_NP = {
    "increases_risk": "जोखिम बढाउँछ ⬆️",
    "decreases_risk": "जोखिम घटाउँछ ⬇️",
}

# ---- Tips per feature -------------------------------------------------------
_IMPROVEMENT_TIPS: dict[str, str] = {
    "credit_score": (
        "आफ्नो क्रेडिट स्कोर सुधार गर्न समयमै ऋण तिर्नुहोस् र नयाँ ऋण लिन सावधान हुनुहोस्।"
    ),
    "loan_repayment_history": (
        "पुरानो ऋणहरू समयमै तिर्नाले तपाईंको भुक्तान इतिहास राम्रो हुन्छ।"
    ),
    "loan_to_income_ratio": (
        "ऋण रकम घटाउनुस् वा आफ्नो आम्दानी बढाउनुस् ताकि ऋण-आय अनुपात सुधारियोस्।"
    ),
    "existing_loans": (
        "पहिलेका ऋणहरू चुक्ता गरेपछि फेरि आवेदन दिनुस्।"
    ),
    "collateral_value": (
        "बढी मूल्यको धितो प्रदान गर्नाले ऋण स्वीकृति सम्भावना बढ्छ।"
    ),
    "cooperative_member": (
        "सहकारीमा सदस्यता लिनाले ऋण विश्वसनीयता बढ्छ।"
    ),
}

_DEFAULT_TIP = "बैंकका ग्राहक सेवा प्रतिनिधिसँग थप जानकारीका लागि सम्पर्क गर्नुहोस्।"


def _get_tip(feature: str) -> str:
    from explainer.shap_explainer import _base_feature
    base = _base_feature(feature)
    return _IMPROVEMENT_TIPS.get(base, _DEFAULT_TIP)


def render_customer_view(
    application: pd.DataFrame,
    pipeline,
    explainer: SHAPExplainer,
) -> None:
    """Render the customer-facing Nepali explanation panel.

    Parameters
    ----------
    application:
        Single-row DataFrame for the loan application.
    pipeline:
        Fitted stacking pipeline.
    explainer:
        Fitted SHAPExplainer instance.
    """
    default_prob = float(pipeline.predict_proba(application)[0, 1])

    if default_prob < 0.30:
        decision = "APPROVE"
        st.markdown(_APPROVE_MSG)
    elif default_prob < 0.50:
        decision = "REVIEW"
        st.markdown(_REVIEW_MSG)
    else:
        decision = "REJECT"
        st.markdown(_REJECT_MSG)

    local_df = explainer.local_explanation(application, top_n=5)

    # Show bullet points in Nepali
    for _, row in local_df.iterrows():
        direction_np = _DIRECTION_NP.get(row["direction"], row["direction"])
        label = row["feature_label_np"]
        st.markdown(f"- **{label}** — {direction_np}")

    if decision in ("REVIEW", "REJECT"):
        st.markdown("---")
        st.markdown("### 💡 सुधारका सुझावहरू")
        high_risk_features = local_df[local_df["direction"] == "increases_risk"]
        for _, row in high_risk_features.iterrows():
            tip = _get_tip(row["feature"])
            st.markdown(f"**{row['feature_label_np']}:** {tip}")

    st.markdown("---")
    st.markdown(
        "📞 **थप जानकारी:** Global IME Bank को नजिकको शाखामा सम्पर्क गर्नुहोस् "
        "वा हेल्पलाइन **1660-01-54321** मा फोन गर्नुहोस्।"
    )
    st.caption(
        "यो निर्णय Nepal Rastra Bank AI Guidelines 2025 अनुसार पारदर्शी "
        "AI प्रणाली (XAI-RAS) द्वारा गरिएको हो।"
    )
