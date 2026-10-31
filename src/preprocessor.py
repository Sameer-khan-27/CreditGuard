"""
preprocessor.py
===============
Builds a scikit-learn preprocessing pipeline for the CreditGuard dataset.

Responsibilities
----------------
* Load the raw CSV (or accept a DataFrame directly).
* Perform a stratified 80/20 train-test split.
* Apply median imputation for missing numerical values.
* Apply RobustScaler (robust to outliers in income, loan amount, etc.).
* Compare two training-set variants:
    - Baseline   : imbalanced (as-is)
    - SMOTE      : over-sampled minority class to restore balance
* Serialize the fitted pipeline to models/preprocessor_pipeline.joblib.

Usage
-----
    python src/preprocessor.py
    # → writes models/preprocessor_pipeline.joblib
    # → prints class distribution comparison
"""

import os
import warnings
from pathlib import Path
from typing import Tuple, Dict, Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "annual_income",
    "loan_amount",
    "interest_rate",
    "dti_ratio",
    "fico_score",
    "credit_line_age_years",
    "open_credit_lines",
    "revolving_utilization",
    "delinquencies_2yr",
]
TARGET_COL = "default_status"
TEST_SIZE = 0.20
RANDOM_SEED = 42
SMOTE_SAMPLING_STRATEGY = 0.40   # minority will become 40 % of majority after oversampling


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load the credit records CSV from disk.

    Parameters
    ----------
    csv_path : str
        Absolute or relative path to ``credit_records.csv``.

    Returns
    -------
    pd.DataFrame
        Raw credit records dataframe.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the given path.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'. "
            "Run `python src/data_generator.py` first."
        )
    df = pd.read_csv(path)
    print(f"[preprocessor] Loaded {len(df):,} records from {csv_path}")
    return df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate feature matrix X from the binary target vector y.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset including target column.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        (X, y) where X contains FEATURE_COLS and y is TARGET_COL.
    """
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    return X, y


def train_test_stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform a stratified train-test split preserving the class ratio.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary target vector.
    test_size : float
        Fraction of data reserved for testing (default 0.20).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(
        f"[preprocessor] Train: {len(X_train):,} | Test: {len(X_test):,} | "
        f"Train default rate: {y_train.mean():.2%} | Test default rate: {y_test.mean():.2%}"
    )
    return X_train, X_test, y_train, y_test


def build_preprocessing_pipeline() -> Pipeline:
    """
    Construct the scikit-learn numerical preprocessing pipeline.

    Steps
    -----
    1. SimpleImputer  – fills NaN with feature median (robust to outliers).
    2. RobustScaler   – centres/scales using median and IQR (handles extreme income/loan values).

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted preprocessing pipeline.
    """
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
    return pipeline


def apply_smote(
    X_train: np.ndarray,
    y_train: pd.Series,
    sampling_strategy: float = SMOTE_SAMPLING_STRATEGY,
    random_state: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE to the training set to mitigate severe class imbalance.

    SMOTE synthesises new minority-class (default) observations by
    interpolating between existing ones in feature space, rather than
    simply duplicating records.

    Parameters
    ----------
    X_train : np.ndarray
        Preprocessed training feature matrix (already scaled).
    y_train : pd.Series or np.ndarray
        Binary training labels.
    sampling_strategy : float
        Desired ratio of minority to majority after resampling.
        E.g., 0.40 means the minority will be 40 % the size of the majority.
    random_state : int
        Seed for SMOTE's internal k-NN search.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (X_resampled, y_resampled) with the minority class augmented.
    """
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        k_neighbors=5,
    )
    X_res, y_res = smote.fit_resample(X_train, y_train)
    before = np.bincount(y_train.astype(int))
    after = np.bincount(y_res.astype(int))
    print(
        f"[preprocessor] SMOTE | Before: {before[0]:,} non-default / {before[1]:,} default "
        f"| After: {after[0]:,} non-default / {after[1]:,} default"
    )
    return X_res, y_res


def prepare_datasets(
    csv_path: str,
) -> Dict[str, Any]:
    """
    Full preprocessing pipeline: load → split → scale → SMOTE.

    This is the single entry-point called by ``model_trainer.py`` and
    the Streamlit app loader.

    Parameters
    ----------
    csv_path : str
        Path to the raw credit records CSV.

    Returns
    -------
    dict with keys:
        ``pipeline``        – Fitted sklearn preprocessing Pipeline.
        ``X_train``         – Scaled training features (baseline, imbalanced).
        ``y_train``         – Training labels (baseline).
        ``X_train_smote``   – Scaled + SMOTE-resampled training features.
        ``y_train_smote``   – SMOTE-resampled training labels.
        ``X_test``          – Scaled test features.
        ``y_test``          – Test labels.
        ``feature_names``   – List of feature column names.
    """
    # 1. Load
    df = load_data(csv_path)

    # 2. Split features / target
    X, y = split_features_target(df)

    # 3. Train-test split
    X_train_raw, X_test_raw, y_train, y_test = train_test_stratified_split(X, y)

    # 4. Fit preprocessing on train, transform both
    pipeline = build_preprocessing_pipeline()
    X_train_scaled = pipeline.fit_transform(X_train_raw)
    X_test_scaled = pipeline.transform(X_test_raw)

    # 5. SMOTE on the training set only (no data leakage)
    X_train_smote, y_train_smote = apply_smote(X_train_scaled, y_train)

    return {
        "pipeline": pipeline,
        "X_train": X_train_scaled,
        "y_train": y_train.to_numpy(),
        "X_train_smote": X_train_smote,
        "y_train_smote": y_train_smote,
        "X_test": X_test_scaled,
        "y_test": y_test.to_numpy(),
        "feature_names": FEATURE_COLS,
    }


def save_pipeline(pipeline: Pipeline, output_dir: str = "models") -> str:
    """
    Persist the fitted preprocessing pipeline to disk using joblib.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Fitted preprocessing pipeline.
    output_dir : str
        Directory to write ``preprocessor_pipeline.joblib``.

    Returns
    -------
    str
        Absolute path to the saved pipeline file.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "preprocessor_pipeline.joblib")
    joblib.dump(pipeline, out_path)
    print(f"[preprocessor] Pipeline saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main – standalone run for quick validation
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    csv_path = str(base_dir / "data" / "credit_records.csv")
    models_dir = str(base_dir / "models")

    data = prepare_datasets(csv_path)
    save_pipeline(data["pipeline"], models_dir)

    print("\n[preprocessor] Class distribution comparison")
    print(f"  Baseline   – {np.bincount(data['y_train'])}")
    print(f"  SMOTE      – {np.bincount(data['y_train_smote'])}")
    print(f"  Test set   – {np.bincount(data['y_test'])}")
