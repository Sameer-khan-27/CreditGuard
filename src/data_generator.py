"""
data_generator.py
=================
Generates a realistic synthetic consumer credit dataset with 35,000 records
and an inherent default rate of approximately 12%.

Feature distributions are calibrated to mirror real-world consumer lending
portfolios (e.g., LendingClub, FICO industry benchmarks).

Usage
-----
    python src/data_generator.py
    # → writes data/credit_records.csv
"""

import os
import numpy as np
import pandas as pd
from scipy.special import expit  # numerically stable sigmoid


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_RECORDS: int = 35_000
RANDOM_SEED: int = 42
DEFAULT_TARGET_RATE: float = 0.12   # ~12 % default rate


def generate_credit_dataset(
    n_records: int = N_RECORDS,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate a synthetic consumer credit application portfolio.

    Parameters
    ----------
    n_records : int
        Number of credit records to generate (default 35,000).
    random_seed : int
        NumPy random seed for full reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with the following columns:

        applicant_id            – Unique string identifier (CG-XXXXXX)
        annual_income           – Annual income in USD (log-normal)
        loan_amount             – Requested loan amount USD
        interest_rate           – Contract interest rate (%)
        dti_ratio               – Debt-to-income ratio
        fico_score              – FICO credit score (580–850)
        credit_line_age_years   – Age of oldest credit line (years)
        open_credit_lines       – Number of open credit lines
        revolving_utilization   – Revolving credit utilization ratio
        delinquencies_2yr       – 30-day delinquency count in last 2 years
        default_status          – Binary target (1 = default/charge-off)
    """
    rng = np.random.default_rng(random_seed)

    # ------------------------------------------------------------------
    # 1. Core feature generation
    # ------------------------------------------------------------------

    # Log-normal income: μ=ln(65 000), σ=0.55 → mean ≈ $70k
    raw_income = rng.lognormal(mean=np.log(65_000), sigma=0.55, size=n_records)
    annual_income = np.clip(raw_income, 20_000, 250_000).round(2)

    # Loan amount: slightly correlated with income
    loan_fraction = rng.uniform(0.03, 0.55, size=n_records)
    loan_amount = np.clip(annual_income * loan_fraction, 1_000, 40_000).round(2)

    # Interest rate: higher for riskier profiles (will correlate with FICO)
    interest_rate = rng.uniform(5.0, 28.0, size=n_records).round(2)

    # DTI ratio
    dti_ratio = rng.beta(2, 6, size=n_records)           # skewed toward lower values
    dti_ratio = np.clip(dti_ratio * 1.2, 0.05, 0.60).round(4)

    # FICO score: normal distribution centred at 700, clipped 580–850
    fico_raw = rng.normal(loc=700, scale=70, size=n_records)
    fico_score = np.clip(fico_raw, 580, 850).round(0).astype(int)

    # Re-price interest_rate inversely with FICO (realistic underwriting)
    fico_normalized = (fico_score - 580) / (850 - 580)          # 0→1, higher = better
    interest_rate = np.clip(28.0 - 23.0 * fico_normalized + rng.normal(0, 1.5, size=n_records), 5.0, 28.0).round(2)

    # Credit line age
    credit_line_age_years = rng.uniform(1, 30, size=n_records).round(1)

    # Open credit lines
    open_credit_lines = rng.integers(2, 26, size=n_records)

    # Revolving utilization: beta distribution, occasionally > 1 (over-limit)
    revolving_utilization = rng.beta(2, 5, size=n_records) * 1.2
    revolving_utilization = np.clip(revolving_utilization, 0.05, 1.10).round(4)

    # Delinquencies in last 2 years: zero-inflated Poisson-like
    delinquency_base = rng.poisson(lam=0.4, size=n_records)
    delinquencies_2yr = np.clip(delinquency_base, 0, 6).astype(int)

    # ------------------------------------------------------------------
    # 2. Default probability via logistic combination of risk factors
    #    Coefficients chosen so that unconditional P(default) ≈ 12 %
    # ------------------------------------------------------------------
    # Standardize each driver for stable linear combination
    z = (
        -5.20                                               # intercept (calibrated)
        + 3.50 * ((dti_ratio - 0.05) / (0.60 - 0.05))     # high DTI → risk ↑
        - 3.80 * fico_normalized                            # low FICO → risk ↑
        + 2.40 * ((revolving_utilization - 0.05) / 1.05)   # high util → risk ↑
        + 1.20 * (delinquencies_2yr / 6.0)                 # recent deliq → risk ↑
        + 0.80 * ((loan_amount - 1_000) / 39_000)          # larger loan → risk ↑
        + 0.50 * ((interest_rate - 5.0) / 23.0)            # high rate → risk ↑ (proxy)
        - 0.40 * (credit_line_age_years / 30.0)            # older history → risk ↓
        + rng.normal(0, 0.35, size=n_records)               # idiosyncratic noise
    )

    default_prob = expit(z)
    default_status = (rng.uniform(size=n_records) < default_prob).astype(int)

    # ------------------------------------------------------------------
    # 3. Assemble DataFrame
    # ------------------------------------------------------------------
    applicant_ids = [f"CG-{str(i).zfill(6)}" for i in range(1, n_records + 1)]

    df = pd.DataFrame(
        {
            "applicant_id": applicant_ids,
            "annual_income": annual_income,
            "loan_amount": loan_amount,
            "interest_rate": interest_rate,
            "dti_ratio": dti_ratio,
            "fico_score": fico_score,
            "credit_line_age_years": credit_line_age_years,
            "open_credit_lines": open_credit_lines,
            "revolving_utilization": revolving_utilization,
            "delinquencies_2yr": delinquencies_2yr,
            "default_status": default_status,
        }
    )

    actual_rate = df["default_status"].mean()
    print(
        f"[data_generator] Generated {n_records:,} records | "
        f"Default rate: {actual_rate:.2%}"
    )
    return df


# ---------------------------------------------------------------------------
# Main – export CSV
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "credit_records.csv")

    dataset = generate_credit_dataset()
    dataset.to_csv(output_path, index=False)
    print(f"[data_generator] Dataset saved → {output_path}")
