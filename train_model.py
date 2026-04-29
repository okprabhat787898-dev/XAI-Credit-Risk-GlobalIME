from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from feature_engineering import add_feature_engineering


def generate_synthetic_dataset(n_rows: int = 2000) -> pd.DataFrame:
    """Generate a synthetic credit dataset with realistic ranges and noise."""
    rng = np.random.default_rng(42)
    monthly_income = rng.normal(60000, 20000, size=n_rows).clip(5000, 200000).round().astype(int)
    loan_amount = (monthly_income * rng.uniform(0.5, 10.0, size=n_rows)).round().astype(int)
    age = rng.integers(18, 75, size=n_rows)
    primary_income_source = rng.choice(
        ["Salary", "Business", "Agriculture", "Remittance", "Other"], size=n_rows, p=[0.5, 0.2, 0.15, 0.1, 0.05]
    )
    essential_expenses = (monthly_income * rng.uniform(0.08, 0.45, size=n_rows)).round().astype(int)
    monthly_installment = (loan_amount / rng.uniform(12, 120, size=n_rows)).round().astype(int)
    remittance_status = rng.choice([0, 1], size=n_rows, p=[0.8, 0.2])
    land_area = np.round(rng.exponential(1.5, size=n_rows), 2)
    cib_score = rng.integers(300, 851, size=n_rows)

    # Create a probabilistic target influenced by DTI, CIB, age, and noise
    dti = loan_amount / np.where(monthly_income == 0, 1.0, monthly_income)
    logits = -1.5 + 1.8 * dti - 0.003 * (cib_score - 600) + 0.01 * (age - 35) + rng.normal(0, 0.6, size=n_rows)
    prob = 1 / (1 + np.exp(-logits))
    risk = (prob > 0.5).astype(int)

    df = pd.DataFrame(
        {
            "Monthly_Income": monthly_income,
            "Loan_Amount": loan_amount,
            "Age": age,
            "Primary_Income_Source": primary_income_source,
            "Essential_Expenses": essential_expenses,
            "Monthly_Installment": monthly_installment,
            "Remittance_Status": remittance_status,
            "Land_Area": land_area,
            "CIB_Score": cib_score,
            "Risk": risk,
        }
    )

    return df


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    data_dir = repo_root / "data"
    artifacts_dir = repo_root / "artifacts"
    reports_dir = repo_root / "reports"

    data_dir.mkdir(exist_ok=True)
    artifacts_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    real_csv = data_dir / "credit_data.csv"
    synthetic_csv = data_dir / "synthetic_credit.csv"

    if real_csv.exists():
        print(f"Loading real dataset from {real_csv}")
        df = pd.read_csv(real_csv)
    else:
        if synthetic_csv.exists():
            print(f"No real dataset found — loading existing synthetic dataset {synthetic_csv}")
            df = pd.read_csv(synthetic_csv)
        else:
            print("No dataset found — generating synthetic dataset (2000 rows)")
            df = generate_synthetic_dataset(2000)
            df.to_csv(synthetic_csv, index=False)
            print(f"Synthetic dataset written to {synthetic_csv}")

    # Verify minimal required columns exist
    required = {
        "Monthly_Income",
        "Loan_Amount",
        "Age",
        "Primary_Income_Source",
        "Essential_Expenses",
        "Monthly_Installment",
        "Risk",
    }
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise KeyError(f"Input dataset missing required columns: {missing}")

    # Feature engineering
    engineered = add_feature_engineering(df)
    engineered["Loan_Coverage_Ratio"] = (
        (engineered["Stress_Tested_Income"].astype(float) - engineered["Essential_Expenses"].astype(float))
        / engineered["Monthly_Installment"].replace(0, pd.NA)
    )

    # Select features (more than 3) including engineered features
    feature_cols = [
        "Monthly_Income",
        "Loan_Amount",
        "Age",
        "Stress_Tested_Income",
        "Loan_Coverage_Ratio",
        "Remittance_Status",
        "CIB_Score",
    ]

    X = engineered[feature_cols].fillna(0)
    y = engineered["Risk"].astype(int)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    # Persist artifacts
    model_path = artifacts_dir / "model.joblib"
    background_path = artifacts_dir / "background_X.joblib"
    feature_names_path = artifacts_dir / "feature_names.json"

    joblib.dump(model, model_path)
    joblib.dump(X, background_path)
    with open(feature_names_path, "w", encoding="utf-8") as fh:
        json.dump(list(X.columns), fh, ensure_ascii=False, indent=2)

    # Save engineered training features for reporting
    engineered.to_csv(reports_dir / "training_engineered_features.csv", index=False)

    print("Training complete")
    print(f"Model saved to: {model_path}")
    print(f"Feature names saved to: {feature_names_path}")
    print(f"SHAP background X saved to: {background_path}")


if __name__ == "__main__":
    main()
