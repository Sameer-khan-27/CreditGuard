"""
model_trainer.py
================
Trains, evaluates, and serialises three credit-risk classification models:

1. Cost-Weighted Logistic Regression  (interpretable baseline)
2. Balanced Random Forest             (ensemble, handles imbalance natively)
3. XGBoost Classifier                 (gradient boosting, typically best performer)

Evaluation metrics
------------------
* ROC-AUC          – discrimination power
* PR-AUC           – precision-recall, critical for imbalanced classes
* F1-Score         – harmonic mean of precision and recall
* Brier Score      – probability calibration quality (lower = better)
* Financial Cost   – $10,000 × FN  +  $500 × FP  (domain-specific loss)

Post-training
-------------
* CalibratedClassifierCV  – Platt scaling for well-calibrated probabilities
* SHAP TreeExplainer      – global feature importance for the best model
* All artefacts serialised to models/

Usage
-----
    python src/model_trainer.py
    # prerequisite: data/credit_records.csv must exist
    #   run `python src/data_generator.py` first if needed
"""

import os
import json
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    confusion_matrix,
)
from xgboost import XGBClassifier

# Internal imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessor import prepare_datasets

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
COST_FN = 10_000     # cost of missing a defaulter ($)
COST_FP = 500        # cost of wrongly rejecting a good borrower ($)
MODELS_DIR = str(Path(__file__).resolve().parent.parent / "models")
DATA_CSV = str(Path(__file__).resolve().parent.parent / "data" / "credit_records.csv")


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def get_model_definitions() -> Dict[str, Any]:
    """
    Return unfitted model instances with production-grade hyperparameters.

    The ``class_weight='balanced'`` and ``scale_pos_weight`` parameters are
    the cost-sensitive knobs that penalise false negatives more heavily.

    Returns
    -------
    dict
        Keys are human-readable model names; values are unfitted estimators.
    """
    n_default_ratio = 7.0   # approximate majority:minority ratio for XGB

    return {
        "Logistic Regression (Cost-Weighted)": LogisticRegression(
            class_weight="balanced",
            max_iter=1_000,
            solver="lbfgs",
            C=0.1,
            random_state=RANDOM_SEED,
        ),
        "Balanced Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            max_depth=12,
            min_samples_leaf=10,
            n_jobs=-1,
            random_state=RANDOM_SEED,
        ),
        "XGBoost Classifier": XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.80,
            colsample_bytree=0.80,
            scale_pos_weight=n_default_ratio,   # cost-sensitive weighting
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbosity=0,
        ),
    }


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def compute_financial_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_fn: int = COST_FN,
    cost_fp: int = COST_FP,
) -> float:
    """
    Compute the total portfolio financial cost given a confusion matrix.

    Cost formula
    ------------
        Total Cost = cost_fn × False Negatives  +  cost_fp × False Positives

    A False Negative (missed default) costs the lender the full outstanding
    principal (~$10,000 average loss). A False Positive (good borrower
    incorrectly rejected) costs in foregone net interest margin (~$500).

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_pred : np.ndarray
        Predicted binary labels (threshold-based).
    cost_fn : int
        Dollar cost per false negative.
    cost_fp : int
        Dollar cost per false positive.

    Returns
    -------
    float
        Total financial cost in dollars.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return float(cost_fn * fn + cost_fp * fp)


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Compute all evaluation metrics for a fitted model.

    Parameters
    ----------
    model : fitted estimator
        Must implement predict_proba.
    X_test : np.ndarray
        Scaled test feature matrix.
    y_test : np.ndarray
        True binary test labels.
    model_name : str
        Human-readable label for logging.
    threshold : float
        Decision boundary for converting probabilities to binary predictions.

    Returns
    -------
    dict
        ROC-AUC, PR-AUC, F1, Brier score, financial cost, confusion matrix,
        and raw probability arrays for curve plotting.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    roc_auc = roc_auc_score(y_test, y_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    f1 = f1_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_proba)
    fin_cost = compute_financial_cost(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(
        f"  [{model_name}] "
        f"ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}  "
        f"F1={f1:.4f}  Brier={brier:.4f}  FinCost=${fin_cost:,.0f}"
    )

    return {
        "model_name": model_name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1": f1,
        "brier_score": brier,
        "financial_cost": fin_cost,
        "confusion_matrix": cm.tolist(),
        "y_proba": y_proba,
        "y_pred": y_pred,
        "precision_curve": precision,
        "recall_curve": recall,
    }


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def compute_shap_values(
    model,
    X_data: np.ndarray,
    feature_names: list,
    sample_size: int = 2_000,
    random_state: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Compute SHAP TreeExplainer values for a tree-based model.

    Samples up to ``sample_size`` rows from the test set to keep runtime
    manageable while still providing statistically representative explanations.

    Parameters
    ----------
    model : fitted tree estimator
        XGBoost or Random Forest with predict_proba support.
    X_data : np.ndarray
        Feature matrix to explain (test set recommended).
    feature_names : list
        Ordered list of feature names.
    sample_size : int
        Max number of records to explain.
    random_state : int
        Seed for sampling.

    Returns
    -------
    dict
        ``shap_values``      – np.ndarray of shape (n_samples, n_features).
        ``expected_value``   – Model base value (log-odds intercept).
        ``feature_names``    – Feature name list.
        ``X_sample``         – The feature matrix rows that were explained.
        ``mean_abs_shap``    – Mean absolute SHAP per feature (global importance).
    """
    rng = np.random.default_rng(random_state)
    n = min(sample_size, len(X_data))
    idx = rng.choice(len(X_data), size=n, replace=False)
    X_sample = X_data[idx]

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_sample)

    # XGBoost returns a 2-D array for binary classification;
    # RandomForest returns a list [class0_array, class1_array]
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]   # class 1 (default)

    mean_abs = np.abs(shap_vals).mean(axis=0)

    return {
        "shap_values": shap_vals,
        "expected_value": float(
            explainer.expected_value[1]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value
        ),
        "feature_names": feature_names,
        "X_sample": X_sample,
        "mean_abs_shap": mean_abs,
    }


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate_model(model, X_train: np.ndarray, y_train: np.ndarray):
    """
    Wrap a fitted model with Platt scaling (sigmoid calibration).

    CalibratedClassifierCV with cv='prefit' calibrates an already-fitted
    estimator using cross-validated isotonic or sigmoid regression, improving
    probability calibration without refitting on the full training data.

    Parameters
    ----------
    model : fitted estimator
        The best-performing model after benchmarking.
    X_train : np.ndarray
        Training features (used for calibration fitting).
    y_train : np.ndarray
        Training labels.

    Returns
    -------
    fitted CalibratedClassifierCV
    """
    calibrated = CalibratedClassifierCV(estimator=model, cv="prefit", method="sigmoid")
    calibrated.fit(X_train, y_train)
    print("[model_trainer] Probability calibration applied (Platt scaling)")
    return calibrated


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def save_artefacts(
    pipeline,
    best_model,
    shap_data: Dict[str, Any],
    metrics: Dict[str, Dict],
    models_dir: str = MODELS_DIR,
) -> None:
    """
    Persist all trained artefacts to the models/ directory.

    Files written
    -------------
    preprocessor_pipeline.joblib  – sklearn preprocessing pipeline
    best_model.joblib              – calibrated best model
    shap_data.joblib               – SHAP values and metadata
    metrics_summary.json           – serialisable metrics for the dashboard

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Fitted preprocessing pipeline.
    best_model : fitted estimator
        Best calibrated model.
    shap_data : dict
        SHAP values and associated metadata.
    metrics : dict
        Nested dict of evaluation metrics keyed by model name.
    models_dir : str
        Output directory path.
    """
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(pipeline, os.path.join(models_dir, "preprocessor_pipeline.joblib"))
    joblib.dump(best_model, os.path.join(models_dir, "best_model.joblib"))

    # SHAP: save everything except the raw arrays separately for speed
    joblib.dump(shap_data, os.path.join(models_dir, "shap_data.joblib"))

    # Serialise metrics (convert numpy types to native Python for JSON)
    serialisable: Dict[str, Any] = {}
    for name, m in metrics.items():
        serialisable[name] = {
            "roc_auc": round(float(m["roc_auc"]), 4),
            "pr_auc": round(float(m["pr_auc"]), 4),
            "f1": round(float(m["f1"]), 4),
            "brier_score": round(float(m["brier_score"]), 4),
            "financial_cost": round(float(m["financial_cost"]), 2),
            "confusion_matrix": m["confusion_matrix"],
        }

    with open(os.path.join(models_dir, "metrics_summary.json"), "w") as fh:
        json.dump(serialisable, fh, indent=2)

    print(f"[model_trainer] All artefacts saved → {models_dir}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def train_and_evaluate(data: Dict[str, Any]) -> Tuple[Any, Dict, Dict]:
    """
    Full training-and-evaluation loop for all three models.

    Trains each model on **both** the baseline and SMOTE training sets and
    evaluates each on the held-out test set.  The model with the highest
    ROC-AUC on SMOTE data is selected as the production model.

    Parameters
    ----------
    data : dict
        Output of ``preprocessor.prepare_datasets()``.

    Returns
    -------
    Tuple
        (best_model, best_metrics, all_metrics_dict)
        ``best_model``   – calibrated best estimator
        ``best_metrics`` – evaluation dict for the best model
        ``all_metrics``  – dict of metrics for all models (both variants)
    """
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_train_smote = data["X_train_smote"]
    y_train_smote = data["y_train_smote"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    model_defs = get_model_definitions()
    all_metrics: Dict[str, Dict] = {}

    best_roc = 0.0
    best_model = None
    best_key = ""

    print("\n[model_trainer] ── Baseline (imbalanced) training set ──")
    for name, clf in model_defs.items():
        clf.fit(X_train, y_train)
        key = f"{name} [Baseline]"
        m = evaluate_model(clf, X_test, y_test, key)
        all_metrics[key] = m

    print("\n[model_trainer] ── SMOTE-resampled training set ──")
    smote_models: Dict[str, Any] = {}
    for name, clf in get_model_definitions().items():   # fresh instances
        clf.fit(X_train_smote, y_train_smote)
        key = f"{name} [SMOTE]"
        m = evaluate_model(clf, X_test, y_test, key)
        all_metrics[key] = m
        smote_models[key] = clf

        if m["roc_auc"] > best_roc:
            best_roc = m["roc_auc"]
            best_model = clf
            best_key = key

    print(f"\n[model_trainer] Best model: {best_key} (ROC-AUC={best_roc:.4f})")

    # Calibrate the best model probabilities
    best_model_calibrated = calibrate_model(best_model, X_train_smote, y_train_smote)

    # Re-evaluate calibrated model and update metrics
    best_metrics = evaluate_model(
        best_model_calibrated, X_test, y_test, f"{best_key} [Calibrated]"
    )
    all_metrics[f"{best_key} [Calibrated]"] = best_metrics

    return best_model, best_metrics, all_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print(" CreditGuard — Model Trainer")
    print("=" * 65)

    # 1. Prepare data
    data = prepare_datasets(DATA_CSV)

    # 2. Train and evaluate
    best_model, best_metrics, all_metrics = train_and_evaluate(data)

    # 3. SHAP for best model (uncalibrated – TreeExplainer needs raw tree)
    #    We retrieve the underlying XGB from calibrated wrapper if needed
    underlying = best_model
    if hasattr(best_model, "estimator"):
        underlying = best_model.estimator

    print("\n[model_trainer] Computing SHAP values (this may take ~60 s)…")
    shap_data = compute_shap_values(
        underlying,
        data["X_test"],
        data["feature_names"],
    )
    print("[model_trainer] SHAP computation complete.")

    # 4. Calibrate best model
    best_calibrated = calibrate_model(best_model, data["X_train_smote"], data["y_train_smote"])

    # 5. Save everything
    save_artefacts(
        pipeline=data["pipeline"],
        best_model=best_calibrated,
        shap_data=shap_data,
        metrics=all_metrics,
        models_dir=MODELS_DIR,
    )

    print("\n[model_trainer] Training complete. ✓")
