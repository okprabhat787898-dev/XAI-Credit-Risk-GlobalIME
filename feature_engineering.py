from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PRIMARY_INCOME_SOURCE_COL = "Primary_Income_Source"
MONTHLY_INCOME_COL = "Monthly_Income"
ESSENTIAL_EXPENSES_COL = "Essential_Expenses"
MONTHLY_INSTALLMENT_COL = "Monthly_Installment"
STRESS_TESTED_INCOME_COL = "Stress_Tested_Income"
LOAN_COVERAGE_RATIO_COL = "Loan_Coverage_Ratio"


def add_feature_engineering(
    df: pd.DataFrame,
    primary_income_source_col: str = PRIMARY_INCOME_SOURCE_COL,
    monthly_income_col: str = MONTHLY_INCOME_COL,
    essential_expenses_col: str = ESSENTIAL_EXPENSES_COL,
    monthly_installment_col: str = MONTHLY_INSTALLMENT_COL,
) -> pd.DataFrame:
    """Add stress-tested income and loan coverage ratio features.

    Rules:
    - If Primary_Income_Source is Agriculture, reduce Monthly_Income by 30%.
    - Loan_Coverage_Ratio = (Net Income - Essential Expenses) / Monthly Installment.
    """
    engineered_df = df.copy()

    if monthly_income_col not in engineered_df.columns:
        raise KeyError(f"Missing required column: {monthly_income_col}")
    if essential_expenses_col not in engineered_df.columns:
        raise KeyError(f"Missing required column: {essential_expenses_col}")
    if monthly_installment_col not in engineered_df.columns:
        raise KeyError(f"Missing required column: {monthly_installment_col}")
    if primary_income_source_col not in engineered_df.columns:
        raise KeyError(f"Missing required column: {primary_income_source_col}")

    stress_tested_income = engineered_df[monthly_income_col].astype(float)
    agriculture_mask = engineered_df[primary_income_source_col].astype(str).str.strip().str.lower() == "agriculture"
    stress_tested_income.loc[agriculture_mask] = stress_tested_income.loc[agriculture_mask] * 0.70

    engineered_df[STRESS_TESTED_INCOME_COL] = stress_tested_income

    net_income_minus_expenses = (
        engineered_df[STRESS_TESTED_INCOME_COL].astype(float) - engineered_df[essential_expenses_col].astype(float)
    )
    installment = engineered_df[monthly_installment_col].astype(float)
    engineered_df[LOAN_COVERAGE_RATIO_COL] = net_income_minus_expenses.div(installment.replace(0, pd.NA))

    return engineered_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Create stress-tested income and loan coverage ratio features.")
    parser.add_argument("input_csv", type=Path, help="Path to the input CSV file")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path for the engineered CSV output",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    engineered_df = add_feature_engineering(df)

    if args.output_csv:
        engineered_df.to_csv(args.output_csv, index=False)
    else:
        print(engineered_df.to_string(index=False))


if __name__ == "__main__":
    main()
