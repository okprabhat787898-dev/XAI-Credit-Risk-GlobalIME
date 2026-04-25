# XAI-RAS: Explainable AI Risk Assessment System

> **Global IME Bank AI/ML Hackathon 2026** – Theme: *Autonomous Credit & Lending Orchestrator*

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

XAI-RAS automates credit approval for the Nepalese market with **100% transparency** via SHAP values, a **Stacked Ensemble Model** (Random Forest + LightGBM), and a **dual-audience Streamlit dashboard** in English and Nepali.

---

## ✨ Key Features

| Feature | Details |
|---------|---------|
| **Stacked Ensemble** | Random Forest + LightGBM → Logistic Regression meta-learner |
| **Explainable AI (XAI)** | SHAP TreeExplainer – per-application & portfolio-level insights |
| **Hyper-local features** | Remittance income, agricultural seasonality, cooperative membership, land (Ropani), gold (Tola) |
| **NRB Compliance** | Nepal Rastra Bank AI Guidelines 2025 – no protected attributes, full audit trail |
| **Dual Dashboard** | Technical view for bank officers + plain **Nepali** explanations for customers |
| **Deployment** | Docker & Streamlit Cloud ready |

---

## 🏗️ Architecture

```
Input Features (25+)
       │
  ColumnTransformer  ←  StandardScaler (numeric) + OneHotEncoder (categorical)
       │
  StackingClassifier
  ├── RandomForestClassifier   (base learner 1)
  └── LGBMClassifier           (base learner 2)
        └── LogisticRegression (meta-learner)
       │
  SHAPExplainer (TreeExplainer on LightGBM)
       │
  ┌──────────────────────────────────────┐
  │          Streamlit Dashboard         │
  │  ├── Officer View  (technical/SHAP)  │
  │  ├── Customer View (नेपाली)          │
  │  └── Portfolio Analytics             │
  └──────────────────────────────────────┘
```

---

## 📁 Project Structure

```
XAI-Credit-Risk-GlobalIME/
├── app/
│   ├── main.py                  # Streamlit entry-point
│   └── components/
│       ├── officer_view.py      # Technical bank-officer view
│       └── customer_view.py     # Nepali customer explanation view
├── data/
│   └── generator.py             # Synthetic Nepal-specific data generator
├── models/
│   ├── ensemble.py              # Stacked Ensemble pipeline
│   ├── train.py                 # Training & model persistence
│   └── artifacts/               # Saved model files (gitignored)
├── explainer/
│   └── shap_explainer.py        # SHAP wrapper + Nepali labels
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_explainer.py
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the dashboard

```bash
streamlit run app/main.py
```

The app will be available at `http://localhost:8501`.  
On first launch the model is trained automatically (~30 seconds) and cached.

### 3. Run with Docker

```bash
docker compose up --build
```

---

## 🧪 Tests

```bash
pytest tests/ -v
```

---

## 🌏 Hyper-local Features

| Feature | Unit | Description |
|---------|------|-------------|
| `remittance_monthly_income` | NPR | Income from overseas remittance |
| `remittance_country` | categorical | Destination country |
| `agricultural_annual_income` | NPR | Farm income, scaled by seasonal factor |
| `seasonal_factor` | 0–1 | Agricultural activity index (rice transplanting / harvest months) |
| `land_area_ropani` | Ropani (508 m²) | Land holding size |
| `gold_holdings_tola` | Tola (11.66 g) | Gold collateral |
| `cooperative_member` | boolean | Savings cooperative membership |
| `cooperative_savings` | NPR | Accumulated cooperative savings |

---

## ⚖️ NRB AI Guidelines 2025 – Compliance

- Protected attributes (caste, gender, religion, ethnicity) are **excluded**.
- Every decision is accompanied by **SHAP-based reasoning** (explainability mandate).
- An **audit trail** is maintained for regulatory review.
- The system flags high-risk decisions for **human-in-the-loop** review.

---

## 🛠️ Tech Stack

- **Python 3.9** | scikit-learn | LightGBM | SHAP | Streamlit | Pandas | Plotly

---

## 📄 License

MIT License – © 2026 Global IME Bank Hackathon Contributors
