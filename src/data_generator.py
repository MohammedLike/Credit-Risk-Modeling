"""
Synthetic Credit Risk Data Generator
Generates realistic loan-level data with correlated features for PD/LGD/EAD modeling.
"""
import numpy as np
import pandas as pd
from scipy import stats


def generate_credit_data(n_samples=50_000, default_rate=0.08, seed=42):
    """Generate synthetic credit portfolio data with realistic correlations."""
    rng = np.random.RandomState(seed)

    # --- Borrower Demographics ---
    age = rng.normal(42, 12, n_samples).clip(21, 75).astype(int)
    annual_income = np.exp(rng.normal(10.8, 0.6, n_samples)).clip(15_000, 500_000)
    employment_length = rng.exponential(5, n_samples).clip(0, 40).round(1)
    home_ownership = rng.choice(
        ["RENT", "OWN", "MORTGAGE", "OTHER"],
        n_samples,
        p=[0.30, 0.15, 0.50, 0.05],
    )

    # --- Credit History ---
    fico_score = rng.normal(690, 60, n_samples).clip(300, 850).astype(int)
    num_credit_lines = rng.poisson(8, n_samples).clip(1, 40)
    credit_utilization = rng.beta(2, 5, n_samples).clip(0, 1)
    delinq_2yrs = rng.poisson(0.3, n_samples).clip(0, 10)
    months_since_delinq = np.where(
        delinq_2yrs > 0, rng.exponential(15, n_samples).clip(1, 120), np.nan
    )
    inquiries_6mo = rng.poisson(1.2, n_samples).clip(0, 15)
    public_records = rng.poisson(0.1, n_samples).clip(0, 5)
    revolving_balance = annual_income * credit_utilization * rng.uniform(0.1, 0.5, n_samples)

    # --- Loan Characteristics ---
    loan_purpose = rng.choice(
        ["debt_consolidation", "credit_card", "home_improvement",
         "major_purchase", "small_business", "medical", "other"],
        n_samples,
        p=[0.35, 0.20, 0.12, 0.10, 0.10, 0.05, 0.08],
    )
    loan_amount = np.exp(rng.normal(9.5, 0.8, n_samples)).clip(1_000, 100_000)
    interest_rate = (
        15.0
        - 0.015 * (fico_score - 600)
        + 2.0 * credit_utilization
        + 0.3 * delinq_2yrs
        + rng.normal(0, 1.5, n_samples)
    ).clip(3.0, 30.0)
    term_months = rng.choice([36, 60], n_samples, p=[0.6, 0.4])
    dti_ratio = (
        (loan_amount * interest_rate / 100 / 12) / (annual_income / 12)
        + rng.uniform(0.05, 0.25, n_samples)
    ).clip(0.01, 0.65)

    # --- Macroeconomic Features ---
    gdp_growth = rng.normal(0.025, 0.015, n_samples).clip(-0.05, 0.08)
    unemployment_rate = rng.normal(0.055, 0.02, n_samples).clip(0.02, 0.15)
    fed_funds_rate = rng.normal(0.03, 0.015, n_samples).clip(0.001, 0.08)
    house_price_index = rng.normal(200, 30, n_samples).clip(120, 350)

    # --- Default Probability (latent variable model) ---
    z = (
        -3.5
        - 0.008 * (fico_score - 600)
        + 1.5 * credit_utilization
        + 0.15 * delinq_2yrs
        + 0.8 * (dti_ratio - 0.20)
        - 0.002 * (annual_income / 1000 - 50)
        + 0.4 * (np.where(home_ownership == "RENT", 1, 0))
        + 0.3 * (np.where(loan_purpose == "small_business", 1, 0))
        - 5.0 * gdp_growth
        + 3.0 * unemployment_rate
        + 0.05 * inquiries_6mo
        + 0.2 * public_records
        - 0.01 * employment_length
        + rng.normal(0, 0.5, n_samples)
    )
    pd_true = 1 / (1 + np.exp(-z))
    # Scale to match desired default rate
    threshold = np.percentile(pd_true, 100 * (1 - default_rate))
    default = (pd_true >= threshold).astype(int)

    # --- LGD (for defaulted loans) ---
    lgd = np.where(
        default == 1,
        rng.beta(2, 3, n_samples).clip(0.05, 0.95),
        np.nan,
    )

    # --- EAD ---
    credit_limit = loan_amount * rng.uniform(1.0, 1.5, n_samples)
    drawn_amount = loan_amount * rng.uniform(0.6, 1.0, n_samples)
    ead = np.where(
        default == 1,
        drawn_amount + (credit_limit - drawn_amount) * rng.beta(3, 5, n_samples),
        drawn_amount,
    )

    df = pd.DataFrame({
        "age": age,
        "annual_income": annual_income.round(2),
        "employment_length": employment_length,
        "home_ownership": home_ownership,
        "fico_score": fico_score,
        "num_credit_lines": num_credit_lines,
        "credit_utilization": credit_utilization.round(4),
        "delinq_2yrs": delinq_2yrs,
        "months_since_delinq": np.round(months_since_delinq, 1),
        "inquiries_6mo": inquiries_6mo,
        "public_records": public_records,
        "revolving_balance": revolving_balance.round(2),
        "loan_purpose": loan_purpose,
        "loan_amount": loan_amount.round(2),
        "interest_rate": interest_rate.round(2),
        "term_months": term_months,
        "dti_ratio": dti_ratio.round(4),
        "gdp_growth": gdp_growth.round(4),
        "unemployment_rate": unemployment_rate.round(4),
        "fed_funds_rate": fed_funds_rate.round(4),
        "house_price_index": house_price_index.round(2),
        "credit_limit": credit_limit.round(2),
        "drawn_amount": drawn_amount.round(2),
        "ead": ead.round(2),
        "lgd": np.round(lgd, 4),
        "default": default,
    })

    return df


def add_derived_features(df):
    """Add engineered features to the dataset."""
    df = df.copy()
    df["loan_to_income"] = df["loan_amount"] / df["annual_income"]
    df["balance_to_income"] = df["revolving_balance"] / df["annual_income"]
    df["income_per_credit_line"] = df["annual_income"] / df["num_credit_lines"]
    df["fico_bucket"] = pd.cut(
        df["fico_score"],
        bins=[299, 579, 669, 739, 799, 851],
        labels=["Poor", "Fair", "Good", "Very Good", "Excellent"],
    )
    df["high_utilization"] = (df["credit_utilization"] > 0.5).astype(int)
    df["has_delinquency"] = (df["delinq_2yrs"] > 0).astype(int)
    df["has_public_record"] = (df["public_records"] > 0).astype(int)
    df["monthly_payment_est"] = (
        df["loan_amount"]
        * (df["interest_rate"] / 100 / 12)
        / (1 - (1 + df["interest_rate"] / 100 / 12) ** (-df["term_months"]))
    )
    df["payment_to_income"] = df["monthly_payment_est"] / (df["annual_income"] / 12)
    df["real_rate"] = df["interest_rate"] / 100 - df["gdp_growth"]
    return df
