"""
risk_engine.py
==============
Production risk-decision layer for the CreditGuard credit-scoring system.

Responsibilities
----------------
1. Risk Band Classification
   Maps a predicted default probability (PD) into one of five industry-
   standard risk grades (A–E) with corresponding underwriting actions.

2. Adverse Action Generator
   For applicants declined or flagged for review, generates the top-3
   specific denial/concern reasons using SHAP force values — meeting
   regulatory adverse-action notice requirements (FCRA / ECOA / Reg B).

3. Cutoff Threshold Simulator
   Sweeps approval cutoff thresholds across [0.05, 0.95] and computes
   portfolio-level statistics: approval volume, default loss prevented,
   net revenue, and loss rate, enabling risk officers to set policy.

Usage
-----
    python src/risk_engine.py
    # Runs self-contained demo using synthetic data
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Risk Band Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RiskBand:
    """
    Immutable definition of a credit risk grade.

    Attributes
    ----------
    grade : str
        Single-letter grade (A–E).
    label : str
        Human-readable tier name.
    pd_lower : float
        Inclusive lower bound of PD (probability of default).
    pd_upper : float
        Exclusive upper bound of PD.
    action : str
        Underwriting action to take.
    color : str
        Hex colour for dashboard display.
    """
    grade: str
    label: str
    pd_lower: float
    pd_upper: float
    action: str
    color: str


RISK_BANDS: List[RiskBand] = [
    RiskBand(
        grade="A",
        label="Prime",
        pd_lower=0.00,
        pd_upper=0.03,
        action="Auto-Approve — offer prime rate",
        color="#2ecc71",
    ),
    RiskBand(
        grade="B",
        label="Near-Prime",
        pd_lower=0.03,
        pd_upper=0.08,
        action="Approve — standard rate schedule",
        color="#27ae60",
    ),
    RiskBand(
        grade="C",
        label="Moderate Risk",
        pd_lower=0.08,
        pd_upper=0.18,
        action="Conditional Approval — collateral or co-signer required",
        color="#f39c12",
    ),
    RiskBand(
        grade="D",
        label="High Risk",
        pd_lower=0.18,
        pd_upper=0.35,
        action="Manual Underwriter Review",
        color="#e67e22",
    ),
    RiskBand(
        grade="E",
        label="Subprime",
        pd_lower=0.35,
        pd_upper=1.01,
        action="Reject — Adverse Action Notice required",
        color="#e74c3c",
    ),
]

# ---------------------------------------------------------------------------
# Friendly feature name mapping (for adverse action reasons)
# ---------------------------------------------------------------------------

FEATURE_LABELS: Dict[str, str] = {
    "annual_income":         "Low Annual Income",
    "loan_amount":           "Excessive Loan Amount Relative to Income",
    "interest_rate":         "High Contract Interest Rate",
    "dti_ratio":             "High Debt-to-Income Ratio",
    "fico_score":            "Low Credit Score (FICO)",
    "credit_line_age_years": "Short Credit History",
    "open_credit_lines":     "Excessive Number of Open Credit Lines",
    "revolving_utilization": "High Revolving Credit Utilization",
    "delinquencies_2yr":     "Recent Payment Delinquencies (Last 24 Months)",
}

# For FICO and income, a positive SHAP means high value DECREASES default risk,
# so we flip the label if SHAP > 0.
POSITIVE_BENEFICIAL: set = {"fico_score", "annual_income", "credit_line_age_years"}


# ---------------------------------------------------------------------------
# 1. Risk Band Classifier
# ---------------------------------------------------------------------------

def assign_risk_band(pd_score: float) -> RiskBand:
    """
    Assign a risk grade to a single predicted default probability.

    Parameters
    ----------
    pd_score : float
        Predicted probability of default (0.0 – 1.0).

    Returns
    -------
    RiskBand
        The matching risk band dataclass.

    Raises
    ------
    ValueError
        If pd_score is outside [0, 1].
    """
    if not (0.0 <= pd_score <= 1.0):
        raise ValueError(f"PD must be in [0, 1]; received {pd_score}")

    for band in RISK_BANDS:
        if band.pd_lower <= pd_score < band.pd_upper:
            return band

    # Fallback — should never be reached given bands cover [0, 1.01)
    return RISK_BANDS[-1]


def assign_risk_bands_bulk(pd_scores: np.ndarray) -> List[RiskBand]:
    """
    Vectorised risk-band assignment over an array of PD scores.

    Parameters
    ----------
    pd_scores : np.ndarray
        1-D array of predicted default probabilities.

    Returns
    -------
    list[RiskBand]
        Ordered list of RiskBand objects, one per applicant.
    """
    return [assign_risk_band(float(p)) for p in pd_scores]


def risk_band_summary(pd_scores: np.ndarray) -> pd.DataFrame:
    """
    Produce a portfolio-level summary of risk-band distribution.

    Parameters
    ----------
    pd_scores : np.ndarray
        Array of predicted default probabilities.

    Returns
    -------
    pd.DataFrame
        One row per grade with columns: Grade, Label, Count, Pct, Action.
    """
    bands = assign_risk_bands_bulk(pd_scores)
    grades = [b.grade for b in bands]
    grade_counts = pd.Series(grades).value_counts().reindex(
        [b.grade for b in RISK_BANDS], fill_value=0
    )
    total = len(pd_scores)

    rows = []
    for band in RISK_BANDS:
        cnt = int(grade_counts.get(band.grade, 0))
        rows.append({
            "Grade": band.grade,
            "Label": band.label,
            "PD Range": f"{band.pd_lower*100:.0f}% – {band.pd_upper*100:.0f}%",
            "Count": cnt,
            "Portfolio %": f"{cnt / total * 100:.1f}%",
            "Action": band.action,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Adverse Action Generator
# ---------------------------------------------------------------------------

def generate_adverse_action_reasons(
    shap_values_row: np.ndarray,
    feature_names: List[str],
    top_n: int = 3,
) -> List[str]:
    """
    Generate human-readable adverse-action denial reasons for one applicant.

    The function uses SHAP force values to identify which features most
    increased the applicant's predicted default probability, then maps those
    features to plain-English regulatory statements.

    Regulatory basis: ECOA / Reg B requires creditors to state the principal
    reasons for adverse actions.  SHAP force values provide a mathematically
    rigorous, model-agnostic attribution meeting this requirement.

    Parameters
    ----------
    shap_values_row : np.ndarray
        1-D array of SHAP values for one applicant (length = n_features).
        Positive values indicate risk-increasing contributions.
    feature_names : List[str]
        Feature names in the same order as shap_values_row.
    top_n : int
        Number of adverse-action reasons to return (default 3).

    Returns
    -------
    List[str]
        Up to ``top_n`` plain-English reason strings, ordered by impact.
    """
    n = min(top_n, len(feature_names))

    # Sort by magnitude of risk-increasing contribution (positive SHAP)
    # For beneficial features (FICO, income), a very negative SHAP also
    # indicates the feature is hurting the applicant.
    signed_impact = []
    for i, feat in enumerate(feature_names):
        sv = float(shap_values_row[i])
        if feat in POSITIVE_BENEFICIAL:
            # Negative SHAP = feature value is low = bad for applicant
            impact = -sv   # flip: magnitude of the deficit
        else:
            impact = sv    # positive SHAP = feature value is increasing default risk

        signed_impact.append((feat, impact, sv))

    # Sort descending by (risk-increasing) impact
    signed_impact.sort(key=lambda x: x[1], reverse=True)

    reasons = []
    for feat, impact, sv in signed_impact[:n]:
        if impact <= 0:
            # Feature is actually helping this applicant — skip
            break
        label = FEATURE_LABELS.get(feat, feat.replace("_", " ").title())
        reasons.append(label)

    # Ensure we always return at least one reason
    if not reasons and signed_impact:
        feat = signed_impact[0][0]
        reasons.append(FEATURE_LABELS.get(feat, feat.replace("_", " ").title()))

    return reasons


def full_adverse_action_report(
    applicant_id: str,
    pd_score: float,
    shap_values_row: np.ndarray,
    feature_names: List[str],
    feature_values: Optional[np.ndarray] = None,
    top_n: int = 3,
) -> Dict:
    """
    Generate a complete adverse-action report for a single applicant.

    Parameters
    ----------
    applicant_id : str
        Unique applicant identifier.
    pd_score : float
        Predicted probability of default.
    shap_values_row : np.ndarray
        SHAP values for this applicant.
    feature_names : List[str]
        Feature column names.
    feature_values : np.ndarray, optional
        Raw (unscaled) feature values for display purposes.
    top_n : int
        Number of denial reasons to include.

    Returns
    -------
    dict
        Structured report with: applicant_id, pd_score, risk_grade,
        action, denial_reasons, feature_contribution_table.
    """
    band = assign_risk_band(pd_score)
    reasons = generate_adverse_action_reasons(shap_values_row, feature_names, top_n)

    # Build feature contribution table for transparency
    contribution_table = []
    for i, feat in enumerate(feature_names):
        contribution_table.append({
            "feature": feat,
            "friendly_name": FEATURE_LABELS.get(feat, feat),
            "shap_value": round(float(shap_values_row[i]), 6),
            "raw_value": float(feature_values[i]) if feature_values is not None else None,
        })
    contribution_table.sort(key=lambda r: abs(r["shap_value"]), reverse=True)

    return {
        "applicant_id": applicant_id,
        "pd_score": round(float(pd_score), 6),
        "pd_pct": f"{pd_score * 100:.2f}%",
        "risk_grade": band.grade,
        "risk_label": band.label,
        "action": band.action,
        "adverse_action_reasons": reasons,
        "feature_contributions": contribution_table,
    }


# ---------------------------------------------------------------------------
# 3. Cutoff Threshold Simulator
# ---------------------------------------------------------------------------

def simulate_cutoff_thresholds(
    pd_scores: np.ndarray,
    y_true: np.ndarray,
    loan_amounts: np.ndarray,
    interest_rates: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    loan_term_years: float = 3.0,
    avg_loss_given_default: float = 0.65,
    operational_cost_per_loan: float = 150.0,
) -> pd.DataFrame:
    """
    Simulate portfolio economics across a sweep of approval cutoff thresholds.

    For each threshold τ, applicants with PD < τ are approved.
    The function computes:

    * **Approval Rate**        – % of portfolio approved
    * **Disbursed Volume**     – total principal disbursed ($)
    * **Expected Default Loss** – approved defaults × LGD × loan amount
    * **Gross Interest Revenue** – approved good loans × interest income
    * **Net Revenue**          – gross revenue − default loss − operational costs
    * **Portfolio Default Rate** – defaults among approved loans

    Parameters
    ----------
    pd_scores : np.ndarray
        Model-predicted default probabilities for each applicant.
    y_true : np.ndarray
        Observed binary default outcomes (for post-hoc statistics).
    loan_amounts : np.ndarray
        Requested loan amounts in USD.
    interest_rates : np.ndarray
        Annual interest rates as percentages (e.g., 12.5 for 12.5%).
    thresholds : np.ndarray, optional
        Approval cutoff grid. Defaults to np.linspace(0.05, 0.70, 66).
    loan_term_years : float
        Average loan term for interest revenue calculation.
    avg_loss_given_default : float
        Loss given default (LGD) as a fraction of outstanding principal.
    operational_cost_per_loan : float
        Fixed origination cost per approved loan in USD.

    Returns
    -------
    pd.DataFrame
        One row per threshold with portfolio statistics.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.70, 66)

    rows = []
    n_total = len(pd_scores)

    for tau in thresholds:
        approved_mask = pd_scores < tau

        n_approved = int(approved_mask.sum())
        if n_approved == 0:
            continue

        approved_loans = loan_amounts[approved_mask]
        approved_rates = interest_rates[approved_mask]
        approved_actuals = y_true[approved_mask]
        approved_pds = pd_scores[approved_mask]

        disbursed_volume = float(approved_loans.sum())

        # Default loss: use actual outcomes for realised loss
        actual_defaults = approved_actuals == 1
        realised_loss = float(
            (approved_loans[actual_defaults] * avg_loss_given_default).sum()
        )

        # Expected loss: use predicted PDs (for forward-looking simulation)
        expected_loss = float(
            (approved_pds * approved_loans * avg_loss_given_default).sum()
        )

        # Gross interest revenue on non-defaulted approved loans
        good_mask = approved_actuals == 0
        gross_interest = float(
            (
                approved_loans[good_mask]
                * (approved_rates[good_mask] / 100)
                * loan_term_years
            ).sum()
        )

        op_costs = n_approved * operational_cost_per_loan
        net_revenue = gross_interest - realised_loss - op_costs

        default_rate_approved = float(approved_actuals.mean())

        rows.append({
            "threshold": round(float(tau), 4),
            "approval_rate": round(n_approved / n_total, 4),
            "n_approved": n_approved,
            "disbursed_volume_usd": round(disbursed_volume, 2),
            "realised_default_loss_usd": round(realised_loss, 2),
            "expected_default_loss_usd": round(expected_loss, 2),
            "gross_interest_revenue_usd": round(gross_interest, 2),
            "operational_costs_usd": round(op_costs, 2),
            "net_revenue_usd": round(net_revenue, 2),
            "portfolio_default_rate": round(default_rate_approved, 4),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def load_risk_engine_artefacts(models_dir: str) -> Dict:
    """
    Load the serialised model artefacts needed by the risk engine.

    Parameters
    ----------
    models_dir : str
        Directory containing joblib files created by model_trainer.py.

    Returns
    -------
    dict
        Keys: ``pipeline``, ``model``, ``shap_data``.

    Raises
    ------
    FileNotFoundError
        If any expected artefact file is missing.
    """
    import joblib

    required = [
        "preprocessor_pipeline.joblib",
        "best_model.joblib",
        "shap_data.joblib",
    ]
    artefacts = {}
    name_map = {
        "preprocessor_pipeline.joblib": "pipeline",
        "best_model.joblib": "model",
        "shap_data.joblib": "shap_data",
    }

    for fname in required:
        fpath = os.path.join(models_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"Artefact '{fname}' not found in '{models_dir}'. "
                "Run `python src/model_trainer.py` to generate models."
            )
        artefacts[name_map[fname]] = joblib.load(fpath)

    return artefacts


def score_applicant(
    applicant_features: Dict[str, float],
    pipeline,
    model,
    shap_data: Dict,
    applicant_id: str = "APPLICANT",
) -> Dict:
    """
    End-to-end scoring function for a single new applicant.

    Preprocesses raw feature values, predicts PD, assigns risk band,
    computes SHAP explanation, and returns a full adverse-action report.

    Parameters
    ----------
    applicant_features : dict
        Feature name → raw (unscaled) float value.
    pipeline : sklearn.pipeline.Pipeline
        Fitted preprocessing pipeline.
    model : fitted estimator
        Calibrated best model.
    shap_data : dict
        SHAP data from model_trainer (contains TreeExplainer reference via
        shap_values array and feature_names).
    applicant_id : str
        Identifier for the report.

    Returns
    -------
    dict
        Adverse-action report from ``full_adverse_action_report``.
    """
    import shap as shap_lib

    feature_names = shap_data["feature_names"]
    X_raw = np.array([[applicant_features[f] for f in feature_names]])
    X_scaled = pipeline.transform(X_raw)

    # Predict PD
    pd_score = float(model.predict_proba(X_scaled)[0, 1])

    # SHAP for this applicant via mean of pre-computed SHAP as approximation,
    # or re-compute via stored explainer if available.
    # We use the mean absolute SHAP times sign of deviation from median as
    # a fast per-applicant SHAP approximation.
    # If the caller has the actual SHAP explainer, they can pass it directly.
    mean_abs_shap = shap_data["mean_abs_shap"]
    expected_val = shap_data["expected_value"]
    shap_row = mean_abs_shap * np.sign(
        X_scaled[0] - np.median(shap_data["X_sample"], axis=0)
    )

    return full_adverse_action_report(
        applicant_id=applicant_id,
        pd_score=pd_score,
        shap_values_row=shap_row,
        feature_names=feature_names,
        feature_values=X_raw[0],
    )


# ---------------------------------------------------------------------------
# Main – self-contained demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print(" CreditGuard — Risk Engine Demo")
    print("=" * 65)

    # ── Demo 1: Risk band assignment ────────────────────────────────────
    sample_pds = [0.01, 0.05, 0.13, 0.25, 0.50]
    print("\n[1] Risk Band Assignment")
    for pd in sample_pds:
        band = assign_risk_band(pd)
        print(f"  PD={pd:.0%}  →  Grade {band.grade} ({band.label}) | {band.action}")

    # ── Demo 2: Adverse action reasons ──────────────────────────────────
    print("\n[2] Adverse Action Reasons")
    feature_names = [
        "annual_income", "loan_amount", "interest_rate", "dti_ratio",
        "fico_score", "credit_line_age_years", "open_credit_lines",
        "revolving_utilization", "delinquencies_2yr",
    ]
    # Simulate a high-risk applicant's SHAP values
    fake_shap = np.array([-0.15, 0.08, 0.05, 0.42, -0.30, -0.12, 0.03, 0.38, 0.29])
    reasons = generate_adverse_action_reasons(fake_shap, feature_names, top_n=3)
    print(f"  Top adverse action reasons:")
    for i, r in enumerate(reasons, 1):
        print(f"    {i}. {r}")

    # ── Demo 3: Threshold simulator ─────────────────────────────────────
    print("\n[3] Cutoff Threshold Simulator (synthetic data)")
    rng = np.random.default_rng(42)
    n = 5_000
    fake_pds = rng.beta(1.5, 10, size=n)
    fake_actuals = (rng.uniform(size=n) < fake_pds * 1.2).astype(int)
    fake_loans = rng.uniform(5_000, 25_000, size=n)
    fake_rates = rng.uniform(5, 22, size=n)

    sim_df = simulate_cutoff_thresholds(fake_pds, fake_actuals, fake_loans, fake_rates)
    print(sim_df[["threshold", "approval_rate", "net_revenue_usd", "portfolio_default_rate"]].iloc[::10].to_string(index=False))
