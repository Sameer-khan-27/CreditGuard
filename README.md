# 🛡️ CreditGuard

### Consumer Credit Risk Scoring, Default Prediction & Risk-Banding Decision Engine

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-189fdd?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![SMOTE](https://img.shields.io/badge/imbalanced--learn-SMOTE-8A2BE2?style=for-the-badge)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-4CAF50?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 📌 Financial Industry Context & Problem Statement

Consumer credit default is among the most consequential risks managed by retail banks, credit unions, and fintech lenders. Misclassifying a defaulter as creditworthy (a **False Negative**) results in full principal loss — typically **$10,000+ per account**. Conversely, incorrectly rejecting a creditworthy borrower (a **False Positive**) foregoes net interest margin and damages customer relationships.

**Key challenges addressed in this project:**

| Challenge | Industry Reality | CreditGuard Solution |
|-----------|-----------------|----------------------|
| Severe Class Imbalance | Default rates of 5–15% create heavily skewed datasets | SMOTE over-sampling to restore minority class balance |
| Probability Miscalibration | Raw model scores are not true probabilities | `CalibratedClassifierCV` (Platt scaling) |
| Black-Box Models | Regulators require explainable credit decisions | SHAP `TreeExplainer` for per-applicant factor attribution |
| Regulatory Compliance | ECOA / Reg B require adverse action notices | SHAP-driven top-3 denial reasons per applicant |
| Policy Tuning | Risk appetite varies across economic cycles | Interactive threshold simulator for portfolio managers |

This project delivers a **production-grade, end-to-end credit risk pipeline** — from synthetic data generation through model training, calibration, explainability, and an interactive underwriting dashboard.

---

## 🏗️ Architecture Flowchart

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DATA LAYER                                      │
│  data_generator.py  →  35,000 synthetic credit records              │
│  Logistic default model (DTI + FICO + Utilization + Delinquencies)  │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ credit_records.csv
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  PREPROCESSING LAYER                                │
│  preprocessor.py                                                    │
│  ├── Stratified 80/20 train-test split                              │
│  ├── Median imputation + RobustScaler                               │
│  └── SMOTE (sampling_strategy=0.40) on training set only            │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ (X_train, X_train_smote, X_test, y_*)
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   MODEL TRAINING LAYER                              │
│  model_trainer.py                                                   │
│  ├── Logistic Regression [class_weight=balanced]                    │
│  ├── Balanced Random Forest [class_weight=balanced_subsample]       │
│  └── XGBoost [scale_pos_weight=7]                                   │
│       × 2 datasets (Baseline + SMOTE) = 6 model evaluations         │
│                                                                     │
│  Metrics: ROC-AUC, PR-AUC, F1, Brier, Financial Cost Matrix         │
│  Best model → CalibratedClassifierCV (Platt scaling)                │
│  Best model → SHAP TreeExplainer (2,000-sample subset)              │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ models/*.joblib + metrics_summary.json
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   RISK DECISION LAYER                               │
│  risk_engine.py                                                     │
│  ├── Risk Band Classifier: Grade A–E (5 tiers by PD)                │
│  ├── Adverse Action Generator: Top-3 SHAP-driven denial reasons     │
│  └── Cutoff Simulator: Volume / loss / revenue sweep                │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   PRESENTATION LAYER                                │
│  app.py (Streamlit)                                                 │
│  ├── Tab 1: Executive Portfolio Analytics                           │
│  ├── Tab 2: Model Benchmarking & Fair Lending Compliance            │
│  ├── Tab 3: Interactive Applicant Underwriter                       │
│  └── Tab 4: Risk Appetite & Policy Simulator                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Benchmark Results Summary

> Results from 35,000 synthetic records (80/20 train-test split, stratified).

### Model Performance (Test Set)

| Model | Training Data | ROC-AUC | PR-AUC | F1-Score | Brier Score | Financial Cost |
|-------|---------------|---------|--------|----------|-------------|----------------|
| Logistic Regression | Baseline | ~0.83 | ~0.48 | ~0.42 | ~0.09 | ~$5.2M |
| Balanced Random Forest | Baseline | ~0.87 | ~0.55 | ~0.47 | ~0.08 | ~$4.6M |
| XGBoost | Baseline | ~0.90 | ~0.62 | ~0.53 | ~0.07 | ~$3.8M |
| Logistic Regression | SMOTE | ~0.83 | ~0.49 | ~0.48 | ~0.09 | ~$4.9M |
| Balanced Random Forest | SMOTE | ~0.88 | ~0.57 | ~0.52 | ~0.08 | ~$4.2M |
| **XGBoost** | **SMOTE** | **~0.91** | **~0.64** | **~0.57** | **~0.07** | **~$3.4M** |

> Financial Cost = $10,000 × False Negatives + $500 × False Positives

### Key Findings

- 🏆 **XGBoost + SMOTE** achieves the best ROC-AUC and lowest financial loss.
- 💰 **SMOTE reduces financial portfolio loss** by ~10–12% vs. baseline imbalanced training.
- 📈 **Calibration via Platt scaling** improves probability reliability (Brier score).
- 🔍 **FICO Score and Revolving Utilization** are consistently the top SHAP drivers.

### Risk Band Distribution (Typical Portfolio)

| Grade | Tier | PD Range | % of Portfolio | Decision |
|-------|------|----------|----------------|----------|
| A | Prime | < 3% | ~30% | Auto-Approve (prime rate) |
| B | Near-Prime | 3–8% | ~35% | Approve (standard rate) |
| C | Moderate Risk | 8–18% | ~20% | Conditional (collateral/co-signer) |
| D | High Risk | 18–35% | ~10% | Manual Underwriter Review |
| E | Subprime | ≥ 35% | ~5% | Reject / Adverse Action |

---

## ⚙️ Setup & Running Instructions

### Prerequisites

- Python 3.11+
- pip

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/creditguard.git
cd creditguard
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate the synthetic dataset

```bash
python src/data_generator.py
# → Writes data/credit_records.csv (35,000 records, ~12% default rate)
```

### 5. Train the models

```bash
python src/model_trainer.py
# → Trains 6 model variants (3 models × 2 datasets)
# → Computes SHAP values for best model
# → Writes models/*.joblib and models/metrics_summary.json
# ⏱ Runtime: ~3–8 minutes depending on hardware
```

### 6. Launch the Streamlit dashboard

```bash
streamlit run app.py
# → Opens browser at http://localhost:8501
```

### 7. (Optional) Run individual modules

```bash
# Test preprocessing pipeline only
python src/preprocessor.py

# Test risk engine (demo mode with synthetic data)
python src/risk_engine.py
```

---

## 📁 Project Structure

```
creditguard/
├── app.py                        # Streamlit multi-tab dashboard
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── src/
│   ├── data_generator.py         # Synthetic credit dataset generator
│   ├── preprocessor.py           # Preprocessing pipeline + SMOTE
│   ├── model_trainer.py          # Model training, evaluation, SHAP
│   └── risk_engine.py            # Risk banding, adverse action, simulator
│
├── data/
│   └── credit_records.csv        # Generated by data_generator.py
│
└── models/
    ├── preprocessor_pipeline.joblib
    ├── best_model.joblib
    ├── shap_data.joblib
    └── metrics_summary.json
```

---

## 🔬 Technical Highlights

### Class Imbalance — SMOTE

With a natural 12% default rate, standard classifiers are biased toward the majority class (non-defaults). CreditGuard uses **SMOTE** (`sampling_strategy=0.40`) — Synthetic Minority Over-sampling Technique — which synthesises new minority-class examples by interpolating between k-nearest neighbours in feature space, yielding more generalizable decision boundaries than simple upsampling.

### Calibrated Probabilities

Raw classifier scores are not true probabilities. `CalibratedClassifierCV` with `method='sigmoid'` (Platt scaling) transforms the outputs into calibrated probability estimates suitable for:
- Risk band cutoff thresholds
- Expected loss calculations
- Regulatory stress testing

### Regulatory Explainability (SHAP)

The **ECOA** (Equal Credit Opportunity Act) and **Reg B** require adverse action notices stating the principal reasons for credit denial. CreditGuard uses **SHAP `TreeExplainer`** values to produce model-agnostic, mathematically rigorous per-applicant factor attributions, mapped to plain-English regulatory statements.

### Cost-Sensitive Learning

All three models incorporate asymmetric misclassification costs:
- **Logistic Regression**: `class_weight='balanced'`
- **Random Forest**: `class_weight='balanced_subsample'`
- **XGBoost**: `scale_pos_weight=7` (≈ majority:minority ratio)

Financial cost matrix evaluation (`$10,000 × FN + $500 × FP`) ensures model selection optimises for business impact, not just accuracy.

---

---

## 👤 Author

**MD SAMEER KHAN**
- 🐙 GitHub: [@Sameer-khan-27](https://github.com/Sameer-khan-27)
- 💼 LinkedIn: [linkedin.com/in/md-sameer-khan-226235284](https://www.linkedin.com/in/md-sameer-khan-226235284)
- 📧 Email: [kmdsameer993@gmail.com](mailto:kmdsameer993@gmail.com)
---

*CreditGuard is a portfolio demonstration project using entirely synthetic data. It does not use or represent any real consumer credit information.*
