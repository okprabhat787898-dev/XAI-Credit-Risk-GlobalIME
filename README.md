# XAI-RAS v1.4.0 | Global IME Bank

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)

An explainable AI (XAI) risk assessment dashboard for transparent and practical credit decision support at Global IME Bank.

यो प्रणाली Global IME Bank का लागि बनाइएको Explainable AI (XAI) आधारित जोखिम मूल्याङ्कन ड्यासबोर्ड हो।

## Summary

This repository contains a Streamlit-based demo of an explainable credit-risk assessment system. The app uses a trained Random Forest classifier for inference, and SHAP for local feature explanations. The UI includes an Intake view, an Analysis view with model explainability, an Audit view for compliance exports, and a Pilot Study module with 20 synthetic profiles for quick evaluation.

## What’s changed

- All UI risk scores come from a trained model via `model.predict_proba()` (no simulated sigmoid or heuristic fallback for scoring).
- Explainability is provided by SHAP (`shap.TreeExplainer`) when a background dataset is available.
- A clearly-marked Mock CIB (Demo) flow is present; it is explicitly configurable and never represents a real CIB integration.
- New Pilot Study module: `data/pilot_20_profiles.csv` and a UI tab that scores the 20 profiles and produces a downloadable report.

## Model Artifacts (deployed)

The app expects the following artifacts under the `artifacts/` directory at startup:

- `artifacts/model.joblib` — the trained scikit-learn model (RandomForestClassifier)
- `artifacts/feature_names.json` — ordered list of feature names used at training time (optional but recommended)
- `artifacts/background_X.joblib` — background dataset used to initialize SHAP TreeExplainer (optional but recommended)

The app enforces a hard failure if `artifacts/model.joblib` is missing; explainability fallbacks are explicit and visible in the UI.

## Training pipeline

- `train_model.py` prepares a training dataset (reads `data/` CSV if present; otherwise it synthesizes a dataset), performs feature engineering using `feature_engineering.py`, trains a `RandomForestClassifier`, and persists artifacts into `artifacts/`.
- Produced artifacts include the model, the exact feature ordering (`feature_names.json`), and a small background sample (`background_X.joblib`) for SHAP.
- If you wish to retrain with real data, place a CSV in `data/` and re-run `train_model.py`.

## Explainability (SHAP)

- The app uses SHAP for local explanations. For tree-based models (RandomForest) we instantiate `shap.TreeExplainer(model, data=background_X)` when `background_X` is available.
- SHAP is pinned to `shap==0.41.0` in `requirements.txt`. Interactive force plots are optional and require `streamlit-shap`.
- If SHAP fails to initialize (missing package or memory issues), the app shows an explicit error/guidance and falls back to a simple contribution summary (no silent behavior).

## Mock CIB (Demo) Flow

- By default the app ships with a clearly-labelled mock CIB flow used for demo/testing. All labels and messages have been changed to `Mock CIB (Demo)` to avoid any suggestion of a real integration.
- Configuration:
  - Edit `app.py` and set `MOCK_CIB_MODE = True` or `False` to toggle demo mode.
  - The mock flow is cosmetic and sets a session flag; it does not call any external CIB APIs.
- If you implement a real CIB integration, replace the mock controls with a secure API client and ensure you follow regulatory and security requirements (mutual TLS or OAuth, consent capture, PII handling, logging/audit, rate limits, etc.).

## Pilot Study — 20 Profiles

- A demo module lives in the UI tab `🧪 Pilot Study (20 Profiles)` and uses `data/pilot_20_profiles.csv` (synthetic, no real PII).
- Features:
  - Runs the trained model on all 20 profiles (batch scoring via `predict_proba`).
  - Shows summary metrics (approval rate, risk distribution histogram/bar chart).
  - Lets operators select any profile and load it into the main Intake UI (populates session state values).
  - Computes top-3 SHAP contributions per profile when SHAP is available and adds them to the downloadable CSV.
- Report: The Pilot tab provides a downloadable CSV report containing inputs, model probability, model score (0–100), decision bucket (Approved / Manual Review / Reject), and the top SHAP features when available.

## How to run locally

1. Create and activate a Python 3.10+ environment.

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux / macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Ensure artifacts exist (either from `train_model.py` or pre-built):

```bash
python train_model.py
```

This will create `artifacts/model.joblib`, `artifacts/feature_names.json`, and `artifacts/background_X.joblib` (or synthetic dataset if no `data/` CSV is found).

3. Run the Streamlit app locally:

```bash
streamlit run app.py
```

Open the URL shown in the terminal (typically http://localhost:8501).

## How to deploy on Streamlit Cloud

1. Push the repository (including the `artifacts/` folder) to a Git remote (GitHub/GitLab).
2. In Streamlit Cloud, create a new app and point it to the repository + branch.
3. Set the Python version to 3.10+ in the Streamlit Cloud settings if available.
4. Ensure `requirements.txt` includes pinned versions (`shap==0.41.0`, `streamlit-shap==0.1.0` if you need force plots) so the cloud environment installs the correct packages.
5. Important: Include `artifacts/model.joblib` and `artifacts/background_X.joblib` in the repository or a secure artifact store reachable by the deployed app. The app fails hard if the model artifact is missing.
6. Optionally set an environment variable or edit `app.py` to set `MOCK_CIB_MODE=False` if you will implement and enable a real CIB client. For demo deployments keep `MOCK_CIB_MODE=True` so the UI makes clear no real CIB calls are being made.

Notes on cloud limits:
- SHAP (TreeExplainer) can be memory-hungry depending on the background dataset and model complexity. Keep `background_X` small (a representative sample) to reduce memory impact.

## Files of interest

- `app.py` — Streamlit application and UI logic
- `train_model.py` — training script and artifact generation
- `feature_engineering.py` — shared feature engineering used by training and inference
- `artifacts/` — model and explainability artifacts produced by training
- `data/pilot_20_profiles.csv` — synthetic pilot profiles used by the Pilot Study UI

## Troubleshooting

- Missing model: The app will stop and show an error if `artifacts/model.joblib` is not found. Run `python train_model.py` to create artifacts.
- SHAP errors: If SHAP fails to initialize, the app shows an explicit message and suggests installing the pinned package versions. For interactive force plots also install `streamlit-shap`.

## Contribution

Contributions are welcome. For changes, please:

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes with clear messages.
4. Open a pull request describing the improvement.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

For project and collaboration inquiries open an issue in the repository.

## Version

Current release: v1.4.0
