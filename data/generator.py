"""Data generation module for XAI-RAS.

Generates synthetic loan application data with Nepal-specific features
such as remittance income, agricultural seasonality, and local financial
instruments.
"""

import numpy as np
import pandas as pd


# Nepal-specific constants
NEPAL_DISTRICTS = [
    "Kathmandu", "Lalitpur", "Bhaktapur", "Pokhara", "Chitwan",
    "Butwal", "Biratnagar", "Birgunj", "Dharan", "Hetauda",
    "Nepalgunj", "Dhangadhi", "Bharatpur", "Janakpur", "Itahari",
]

REMITTANCE_COUNTRIES = [
    "Qatar", "UAE", "Saudi Arabia", "Malaysia", "India",
    "South Korea", "Japan", "USA", "Australia", "Kuwait",
]

LOAN_PURPOSES = [
    "agriculture", "business", "education", "home_construction",
    "vehicle", "personal", "remittance_backed", "cooperative",
]

OCCUPATION_TYPES = [
    "farmer", "salaried", "self_employed", "remittance_dependent",
    "trader", "government_employee", "ngo_worker",
]

# Months with high agricultural activity in Nepal (rice transplanting / harvest)
HIGH_AGRICULTURE_MONTHS = {3, 4, 5, 6, 10, 11}  # Chaitra-Ashadh & Kartik-Mangsir


def _agricultural_seasonality(month: int, rng: np.random.Generator) -> float:
    """Return a seasonality multiplier (0–1) for the given calendar month."""
    if month in HIGH_AGRICULTURE_MONTHS:
        return float(rng.uniform(0.7, 1.0))
    return float(rng.uniform(0.2, 0.6))


def generate_loan_applications(
    n_samples: int = 1000,
    random_state: int = 42,
    default_rate: float = 0.25,
) -> pd.DataFrame:
    """Generate a synthetic loan-application dataset for the Nepalese market.

    Parameters
    ----------
    n_samples:
        Number of loan application records to generate.
    random_state:
        Random seed for reproducibility.
    default_rate:
        Approximate proportion of applications that end in default.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per application and a binary ``default`` column.
    """
    rng = np.random.default_rng(random_state)

    # ---- Demographic features -------------------------------------------
    age = rng.integers(18, 65, size=n_samples)
    district = rng.choice(NEPAL_DISTRICTS, size=n_samples)
    occupation = rng.choice(OCCUPATION_TYPES, size=n_samples)
    dependents = rng.integers(0, 8, size=n_samples)

    # ---- Income features ------------------------------------------------
    base_monthly_income = rng.uniform(8_000, 150_000, size=n_samples)  # NPR

    # Remittance income (common in Nepal; ~30% of households receive it)
    receives_remittance = rng.random(size=n_samples) < 0.30
    remittance_country = np.where(
        receives_remittance,
        rng.choice(REMITTANCE_COUNTRIES, size=n_samples),
        "None",
    )
    remittance_monthly_income = np.where(
        receives_remittance,
        rng.uniform(20_000, 80_000, size=n_samples),
        0.0,
    )

    # Agricultural income with seasonal multiplier
    application_month = rng.integers(1, 13, size=n_samples)
    seasonal_factors = np.array([
        _agricultural_seasonality(int(m), rng) for m in application_month
    ])
    has_agricultural_income = rng.random(size=n_samples) < 0.45
    agricultural_annual_income = np.where(
        has_agricultural_income,
        rng.uniform(50_000, 400_000, size=n_samples) * seasonal_factors,
        0.0,
    )

    total_monthly_income = (
        base_monthly_income
        + remittance_monthly_income
        + agricultural_annual_income / 12.0
    )

    # ---- Assets ---------------------------------------------------------
    # Land area in Ropani (1 Ropani ≈ 508.7 m²)
    land_area_ropani = np.where(
        has_agricultural_income,
        rng.uniform(1.0, 50.0, size=n_samples),
        0.0,
    )

    # Gold holdings in Tola (1 Tola ≈ 11.66 g)
    gold_holdings_tola = rng.exponential(scale=5.0, size=n_samples)

    # ---- Banking / credit history --------------------------------------
    credit_score = np.clip(
        rng.normal(loc=600, scale=100, size=n_samples), 300, 900
    ).astype(int)
    existing_loans = rng.integers(0, 5, size=n_samples)
    loan_repayment_history = np.clip(
        rng.normal(loc=0.85, scale=0.15, size=n_samples), 0.0, 1.0
    )

    # Cooperative membership (common savings mechanism in Nepal)
    cooperative_member = rng.random(size=n_samples) < 0.40
    cooperative_savings = np.where(
        cooperative_member,
        rng.uniform(5_000, 100_000, size=n_samples),
        0.0,
    )

    # ---- Loan request details ------------------------------------------
    loan_amount = rng.uniform(50_000, 5_000_000, size=n_samples)  # NPR
    loan_purpose = rng.choice(LOAN_PURPOSES, size=n_samples)
    loan_tenure_months = rng.choice([12, 24, 36, 48, 60, 84, 120], size=n_samples)
    collateral_value = rng.uniform(0, 10_000_000, size=n_samples)  # NPR

    # Debt-to-income ratio
    loan_to_income_ratio = loan_amount / (total_monthly_income * loan_tenure_months)

    # ---- Default label (engineered with domain-driven probabilities) ---
    default_prob = (
        0.35 * (1 - loan_repayment_history)
        + 0.25 * np.clip(loan_to_income_ratio, 0, 1)
        + 0.15 * (existing_loans / 5.0)
        - 0.10 * (credit_score / 900.0)
        - 0.05 * (cooperative_member.astype(float))
        - 0.05 * (receives_remittance.astype(float))
        - 0.03 * (collateral_value / loan_amount).clip(0, 1)
        + rng.normal(0, 0.05, size=n_samples)
    )
    # Shift to match target default_rate
    threshold = np.percentile(default_prob, (1 - default_rate) * 100)
    default = (default_prob >= threshold).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "district": district,
            "occupation": occupation,
            "dependents": dependents,
            "base_monthly_income": base_monthly_income.round(2),
            "receives_remittance": receives_remittance.astype(int),
            "remittance_country": remittance_country,
            "remittance_monthly_income": remittance_monthly_income.round(2),
            "application_month": application_month,
            "seasonal_factor": seasonal_factors.round(4),
            "has_agricultural_income": has_agricultural_income.astype(int),
            "agricultural_annual_income": agricultural_annual_income.round(2),
            "total_monthly_income": total_monthly_income.round(2),
            "land_area_ropani": land_area_ropani.round(2),
            "gold_holdings_tola": gold_holdings_tola.round(2),
            "credit_score": credit_score,
            "existing_loans": existing_loans,
            "loan_repayment_history": loan_repayment_history.round(4),
            "cooperative_member": cooperative_member.astype(int),
            "cooperative_savings": cooperative_savings.round(2),
            "loan_amount": loan_amount.round(2),
            "loan_purpose": loan_purpose,
            "loan_tenure_months": loan_tenure_months,
            "collateral_value": collateral_value.round(2),
            "loan_to_income_ratio": loan_to_income_ratio.round(6),
            "default": default,
        }
    )
    return df


CATEGORICAL_FEATURES = [
    "district",
    "occupation",
    "remittance_country",
    "loan_purpose",
]

NUMERIC_FEATURES = [
    "age",
    "dependents",
    "base_monthly_income",
    "receives_remittance",
    "remittance_monthly_income",
    "seasonal_factor",
    "has_agricultural_income",
    "agricultural_annual_income",
    "total_monthly_income",
    "land_area_ropani",
    "gold_holdings_tola",
    "credit_score",
    "existing_loans",
    "loan_repayment_history",
    "cooperative_member",
    "cooperative_savings",
    "loan_amount",
    "loan_tenure_months",
    "collateral_value",
    "loan_to_income_ratio",
    "application_month",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "default"
