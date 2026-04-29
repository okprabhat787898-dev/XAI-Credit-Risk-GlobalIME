Required columns for dataset

Place a CSV at `data/credit_data.csv` with the following columns (headers):

- Monthly_Income (numeric)
- Loan_Amount (numeric)
- Age (integer)
- Primary_Income_Source (Salary|Business|Agriculture|Remittance|Other)
- Essential_Expenses (numeric)
- Monthly_Installment (numeric)
- Remittance_Status (0 or 1)
- Land_Area (numeric, optional)
- CIB_Score (numeric, optional)
- Risk (0 or 1)  <-- binary target label (1 = high risk)

If you don't have a real dataset, the training script `train_model.py` will generate a synthetic dataset and save it to `data/synthetic_credit.csv`.

How to add real data

1. Create `data/credit_data.csv` with the required columns and place it in the `data/` directory.
2. Run `python train_model.py` to train and create artifacts in `artifacts/`.

Artifacts produced by training

- `artifacts/model.joblib` — trained RandomForest model
- `artifacts/feature_names.json` — ordered list of feature column names used by the model
- `artifacts/background_X.joblib` — training features DataFrame used as SHAP background

Notes

- Feature ordering is important; `feature_names.json` is used by the app to align user inputs with model features.
- The synthetic generator produces 2,000 rows with realistic ranges and random noise for development/testing.