"""Tests for the synthetic data generator."""

import pandas as pd
import pytest

from data.generator import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    generate_loan_applications,
)


def test_generate_returns_dataframe():
    df = generate_loan_applications(n_samples=50, random_state=0)
    assert isinstance(df, pd.DataFrame)


def test_generate_correct_row_count():
    df = generate_loan_applications(n_samples=100, random_state=1)
    assert len(df) == 100


def test_target_column_binary():
    df = generate_loan_applications(n_samples=200, random_state=2)
    assert set(df[TARGET_COLUMN].unique()).issubset({0, 1})


def test_default_rate_approximately_correct():
    df = generate_loan_applications(n_samples=2000, random_state=3, default_rate=0.25)
    actual_rate = df[TARGET_COLUMN].mean()
    assert 0.15 <= actual_rate <= 0.35, f"Default rate {actual_rate:.2f} out of expected range"


def test_all_expected_columns_present():
    df = generate_loan_applications(n_samples=50, random_state=4)
    for col in ALL_FEATURES + [TARGET_COLUMN]:
        assert col in df.columns, f"Missing column: {col}"


def test_numeric_features_no_nulls():
    df = generate_loan_applications(n_samples=100, random_state=5)
    for col in NUMERIC_FEATURES:
        assert df[col].isnull().sum() == 0, f"Null values in {col}"


def test_seasonal_factor_range():
    df = generate_loan_applications(n_samples=500, random_state=6)
    assert (df["seasonal_factor"] >= 0.0).all()
    assert (df["seasonal_factor"] <= 1.0).all()


def test_remittance_income_zero_when_no_remittance():
    df = generate_loan_applications(n_samples=500, random_state=7)
    no_remittance = df[df["receives_remittance"] == 0]
    assert (no_remittance["remittance_monthly_income"] == 0.0).all()


def test_agricultural_income_zero_when_no_agri():
    df = generate_loan_applications(n_samples=500, random_state=8)
    no_agri = df[df["has_agricultural_income"] == 0]
    assert (no_agri["agricultural_annual_income"] == 0.0).all()


def test_credit_score_in_valid_range():
    df = generate_loan_applications(n_samples=200, random_state=9)
    assert (df["credit_score"] >= 300).all()
    assert (df["credit_score"] <= 900).all()


def test_reproducibility():
    df1 = generate_loan_applications(n_samples=50, random_state=42)
    df2 = generate_loan_applications(n_samples=50, random_state=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_different_seeds_give_different_data():
    df1 = generate_loan_applications(n_samples=50, random_state=1)
    df2 = generate_loan_applications(n_samples=50, random_state=2)
    assert not df1[NUMERIC_FEATURES[0]].equals(df2[NUMERIC_FEATURES[0]])
