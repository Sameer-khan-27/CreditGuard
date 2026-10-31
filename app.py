"""
app.py
======
CreditGuard — Interactive Streamlit Dashboard

Four-tab decision-support interface for credit underwriters and risk officers.

Tabs
----
1. Executive Portfolio Analytics
   – Default distribution, correlation matrix, default rate by FICO band.

2. Model Benchmarking & Fair Lending Compliance
   – ROC/PR curves, confusion matrices, SMOTE vs baseline cost comparison,
     global SHAP feature importance.

3. Interactive Applicant Underwriter
   – Sidebar sliders for applicant parameters; returns live PD, Risk Grade
     badge, and SHAP-driven adverse action reasons.

4. Risk Appetite & Policy Simulator
   – Approval cutoff slider; recalculates portfolio volume, default loss,
     and net revenue dynamically.

Setup
-----
    streamlit run app.py
"""

import json
import os
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessor import FEATURE_COLS
from risk_engine import (
    RISK_BANDS,
    assign_risk_band,
    assign_risk_bands_bulk,
    full_adverse_action_report,
    generate_adverse_action_reasons,
    risk_band_summary,
    simulate_cutoff_thresholds,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CreditGuard — Credit Risk Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .main-header {
            font-size: 2.2rem; font-weight: 800; color: #1f2328;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.0rem; color: #57606a; margin-bottom: 1.5rem;
        }
        .metric-card {
            background: #f7f8fa; border: 1px solid #e5e7eb;
            border-radius: 8px; padding: 1rem 1.2rem;
            text-align: center;
        }
        .grade-badge {
            display: inline-block; padding: 6px 18px;
            border-radius: 20px; font-size: 1.1rem;
            font-weight: 700; color: #fff; margin: 4px 0;
        }
        .section-title {
            font-size: 1.15rem; font-weight: 700;
            color: #1f2328; margin-top: 1.2rem; margin-bottom: 0.4rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data & Model Loaders (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading credit portfolio dataset…")
def load_portfolio_data() -> pd.DataFrame:
    csv_path = DATA_DIR / "credit_records.csv"
    if not csv_path.exists():
        st.error(
            "Dataset not found. Run `python src/data_generator.py` first."
        )
        st.stop()
    return pd.read_csv(csv_path)


@st.cache_resource(show_spinner="Loading trained models and SHAP values…")
def load_models():
    required = [
        MODELS_DIR / "preprocessor_pipeline.joblib",
        MODELS_DIR / "best_model.joblib",
        MODELS_DIR / "shap_data.joblib",
        MODELS_DIR / "metrics_summary.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        st.error(
            f"Model artefacts missing: {missing}\n\n"
            "Run `python src/model_trainer.py` to train models first."
        )
        st.stop()

    pipeline = joblib.load(MODELS_DIR / "preprocessor_pipeline.joblib")
    model = joblib.load(MODELS_DIR / "best_model.joblib")
    shap_data = joblib.load(MODELS_DIR / "shap_data.joblib")
    with open(MODELS_DIR / "metrics_summary.json") as fh:
        metrics = json.load(fh)

    return pipeline, model, shap_data, metrics


@st.cache_data(show_spinner="Running model predictions on portfolio…")
def get_portfolio_scores(_pipeline, _model, df: pd.DataFrame) -> np.ndarray:
    X = df[FEATURE_COLS].values
    X_scaled = _pipeline.transform(X)
    return _model.predict_proba(X_scaled)[:, 1]


# ---------------------------------------------------------------------------
# Helper: ROC curve data
# ---------------------------------------------------------------------------

def compute_roc_pr_from_metrics(metrics: dict) -> dict:
    """
    Extract pre-computed ROC / PR summary statistics from metrics dict.
    (Detailed curves require predictions; we use stored AUC values.)
    """
    rows = []
    for name, m in metrics.items():
        rows.append(
            {
                "Model": name,
                "ROC-AUC": m["roc_auc"],
                "PR-AUC": m["pr_auc"],
                "F1-Score": m["f1"],
                "Brier Score": m["brier_score"],
                "Financial Cost ($)": m["financial_cost"],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Grade colour map
# ---------------------------------------------------------------------------
GRADE_COLORS = {b.grade: b.color for b in RISK_BANDS}
GRADE_LABELS = {b.grade: b.label for b in RISK_BANDS}


# ===========================================================================
# MAIN APP
# ===========================================================================

def main():
    # Header
    st.markdown(
        '<div class="main-header">🛡️ CreditGuard</div>'
        '<div class="sub-header">Consumer Credit Risk Scoring, Default Prediction & Risk-Banding Decision Engine</div>',
        unsafe_allow_html=True,
    )

    # Load assets
    df = load_portfolio_data()
    pipeline, model, shap_data, metrics = load_models()
    pd_scores = get_portfolio_scores(pipeline, model, df)

    tabs = st.tabs(
        [
            "📊 Executive Portfolio Analytics",
            "🏆 Model Benchmarking",
            "🔍 Applicant Underwriter",
            "⚖️ Risk Appetite Simulator",
        ]
    )

    # =======================================================================
    # TAB 1 — Executive Portfolio Analytics
    # =======================================================================
    with tabs[0]:
        st.markdown('<div class="section-title">Portfolio Overview</div>', unsafe_allow_html=True)

        default_rate = df["default_status"].mean()
        avg_fico = df["fico_score"].mean()
        avg_loan = df["loan_amount"].mean()
        avg_dti = df["dti_ratio"].mean()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", f"{len(df):,}")
        col2.metric("Portfolio Default Rate", f"{default_rate:.1%}")
        col3.metric("Avg FICO Score", f"{avg_fico:.0f}")
        col4.metric("Avg Loan Amount", f"${avg_loan:,.0f}")

        st.divider()

        # ── Default Distribution
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<div class="section-title">Default Distribution</div>', unsafe_allow_html=True)
            dist_df = df["default_status"].value_counts().rename({0: "Fully Paid", 1: "Default"})
            fig_dist = px.pie(
                values=dist_df.values,
                names=dist_df.index,
                color_discrete_sequence=["#3b82d4", "#e74c3c"],
                hole=0.45,
            )
            fig_dist.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320)
            st.plotly_chart(fig_dist, use_container_width=True)

        with col_b:
            st.markdown('<div class="section-title">Risk Band Distribution</div>', unsafe_allow_html=True)
            bands = assign_risk_bands_bulk(pd_scores)
            grade_series = pd.Series([b.grade for b in bands]).value_counts().reindex(
                [b.grade for b in RISK_BANDS], fill_value=0
            )
            fig_band = px.bar(
                x=grade_series.index,
                y=grade_series.values,
                color=grade_series.index,
                color_discrete_map=GRADE_COLORS,
                labels={"x": "Risk Grade", "y": "Count"},
            )
            fig_band.update_layout(
                showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=320
            )
            st.plotly_chart(fig_band, use_container_width=True)

        st.divider()

        # ── Default Rate by FICO Band
        st.markdown('<div class="section-title">Default Rate by FICO Score Band</div>', unsafe_allow_html=True)
        df_copy = df.copy()
        df_copy["fico_band"] = pd.cut(
            df_copy["fico_score"],
            bins=[580, 620, 660, 700, 740, 780, 850],
            labels=["580-620", "621-660", "661-700", "701-740", "741-780", "781-850"],
        )
        fico_default = df_copy.groupby("fico_band", observed=True)["default_status"].mean().reset_index()
        fico_default.columns = ["FICO Band", "Default Rate"]
        fig_fico = px.bar(
            fico_default,
            x="FICO Band",
            y="Default Rate",
            color="Default Rate",
            color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
            text_auto=".1%",
        )
        fig_fico.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320)
        fig_fico.update_coloraxes(showscale=False)
        st.plotly_chart(fig_fico, use_container_width=True)

        # ── Correlation Matrix
        st.markdown('<div class="section-title">Feature Correlation Matrix</div>', unsafe_allow_html=True)
        corr_cols = ["dti_ratio", "fico_score", "loan_amount", "revolving_utilization",
                     "interest_rate", "delinquencies_2yr", "annual_income", "default_status"]
        corr_matrix = df[corr_cols].corr()
        fig_corr = px.imshow(
            corr_matrix,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            text_auto=".2f",
            aspect="auto",
        )
        fig_corr.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=440)
        st.plotly_chart(fig_corr, use_container_width=True)

        # ── DTI vs Default Rate scatter
        st.markdown('<div class="section-title">Loan Amount vs Default Probability</div>', unsafe_allow_html=True)
        scatter_df = df.copy()
        scatter_df["PD_Score"] = pd_scores
        scatter_df["Default"] = scatter_df["default_status"].map({0: "No", 1: "Yes"})
        sample_df = scatter_df.sample(min(3000, len(scatter_df)), random_state=42)
        fig_scatter = px.scatter(
            sample_df,
            x="loan_amount",
            y="PD_Score",
            color="Default",
            color_discrete_map={"No": "#3b82d4", "Yes": "#e74c3c"},
            opacity=0.5,
            labels={"loan_amount": "Loan Amount ($)", "PD_Score": "Predicted Default Probability"},
        )
        fig_scatter.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=360)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # =======================================================================
    # TAB 2 — Model Benchmarking & Fair Lending Compliance
    # =======================================================================
    with tabs[1]:
        st.markdown('<div class="section-title">Model Performance Benchmark</div>', unsafe_allow_html=True)

        metrics_df = compute_roc_pr_from_metrics(metrics)

        # Highlight best
        def highlight_best(s):
            if s.name in ["ROC-AUC", "PR-AUC", "F1-Score"]:
                is_best = s == s.max()
            elif s.name in ["Brier Score", "Financial Cost ($)"]:
                is_best = s == s.min()
            else:
                return [""] * len(s)
            return ["background-color: #d4edda; font-weight:bold" if v else "" for v in is_best]

        styled = metrics_df.style.apply(highlight_best, axis=0).format(
            {
                "ROC-AUC": "{:.4f}",
                "PR-AUC": "{:.4f}",
                "F1-Score": "{:.4f}",
                "Brier Score": "{:.4f}",
                "Financial Cost ($)": "${:,.0f}",
            }
        )
        st.dataframe(styled, use_container_width=True, height=300)

        st.divider()

        # ── ROC-AUC Bar Chart
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-title">ROC-AUC by Model</div>', unsafe_allow_html=True)
            fig_roc = px.bar(
                metrics_df.sort_values("ROC-AUC", ascending=True),
                x="ROC-AUC",
                y="Model",
                orientation="h",
                color="ROC-AUC",
                color_continuous_scale=["#f39c12", "#2ecc71"],
                text_auto=".4f",
                range_x=[0.5, 1.0],
            )
            fig_roc.update_layout(
                showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=380,
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        with col2:
            st.markdown('<div class="section-title">Financial Cost Comparison ($)</div>', unsafe_allow_html=True)
            fig_cost = px.bar(
                metrics_df.sort_values("Financial Cost ($)", ascending=False),
                x="Financial Cost ($)",
                y="Model",
                orientation="h",
                color="Financial Cost ($)",
                color_continuous_scale=["#2ecc71", "#e74c3c"],
                text_auto=",.0f",
            )
            fig_cost.update_layout(
                showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=380,
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_cost, use_container_width=True)

        st.divider()

        # ── SHAP Global Feature Importance
        st.markdown('<div class="section-title">Global SHAP Feature Importance (Best Model)</div>', unsafe_allow_html=True)
        feature_names = shap_data["feature_names"]
        mean_abs_shap = shap_data["mean_abs_shap"]

        shap_df = pd.DataFrame(
            {"Feature": feature_names, "Mean |SHAP|": mean_abs_shap}
        ).sort_values("Mean |SHAP|", ascending=True)

        friendly = {
            "annual_income": "Annual Income",
            "loan_amount": "Loan Amount",
            "interest_rate": "Interest Rate",
            "dti_ratio": "DTI Ratio",
            "fico_score": "FICO Score",
            "credit_line_age_years": "Credit Line Age (yrs)",
            "open_credit_lines": "Open Credit Lines",
            "revolving_utilization": "Revolving Utilization",
            "delinquencies_2yr": "Delinquencies (2yr)",
        }
        shap_df["Feature Label"] = shap_df["Feature"].map(friendly)

        fig_shap = px.bar(
            shap_df,
            x="Mean |SHAP|",
            y="Feature Label",
            orientation="h",
            color="Mean |SHAP|",
            color_continuous_scale=["#3b82d4", "#7c5cd8"],
            text_auto=".4f",
        )
        fig_shap.update_layout(
            showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=380,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        st.divider()

        # ── SMOTE vs Baseline Cost Reduction Table
        st.markdown('<div class="section-title">SMOTE Impact: Financial Loss Reduction</div>', unsafe_allow_html=True)
        smote_rows = [m for k, m in metrics.items() if "SMOTE" in k and "Calibrated" not in k]
        base_rows = [m for k, m in metrics.items() if "Baseline" in k]

        if smote_rows and base_rows:
            best_smote_cost = min(r["financial_cost"] for r in smote_rows)
            best_base_cost = min(r["financial_cost"] for r in base_rows)
            savings = best_base_cost - best_smote_cost
            pct_savings = savings / best_base_cost * 100 if best_base_cost > 0 else 0

            cost_compare = pd.DataFrame(
                [
                    {"Approach": "Baseline (Imbalanced)", "Best Financial Cost ($)": best_base_cost},
                    {"Approach": "SMOTE (Resampled)", "Best Financial Cost ($)": best_smote_cost},
                    {"Approach": "💰 Savings from SMOTE", "Best Financial Cost ($)": savings},
                ]
            )
            st.dataframe(
                cost_compare.style.format({"Best Financial Cost ($)": "${:,.0f}"}),
                use_container_width=True,
                hide_index=True,
            )
            st.success(f"SMOTE reduces portfolio financial loss by **{pct_savings:.1f}%** (${savings:,.0f} saved)")

    # =======================================================================
    # TAB 3 — Interactive Applicant Underwriter
    # =======================================================================
    with tabs[2]:
        st.markdown('<div class="section-title">Live Applicant Credit Decisioning</div>', unsafe_allow_html=True)
        st.caption(
            "Adjust applicant parameters using the sliders below. "
            "The engine will compute a default probability, assign a risk grade, "
            "and generate SHAP-based adverse-action reasons in real time."
        )

        st.divider()

        # ── Input sliders
        col_in1, col_in2, col_in3 = st.columns(3)

        with col_in1:
            st.markdown("**Income & Loan**")
            annual_income = st.slider("Annual Income ($)", 20_000, 250_000, 65_000, step=1_000)
            loan_amount = st.slider("Loan Amount ($)", 1_000, 40_000, 10_000, step=500)
            interest_rate = st.slider("Interest Rate (%)", 5.0, 28.0, 12.5, step=0.5)

        with col_in2:
            st.markdown("**Credit Profile**")
            fico_score = st.slider("FICO Score", 580, 850, 700, step=5)
            dti_ratio = st.slider("DTI Ratio", 0.05, 0.60, 0.25, step=0.01)
            revolving_utilization = st.slider("Revolving Utilization", 0.05, 1.10, 0.35, step=0.01)

        with col_in3:
            st.markdown("**Credit History**")
            credit_line_age_years = st.slider("Credit Line Age (years)", 1.0, 30.0, 8.0, step=0.5)
            open_credit_lines = st.slider("Open Credit Lines", 2, 25, 6)
            delinquencies_2yr = st.slider("Delinquencies (last 2 years)", 0, 6, 0)

        applicant_features = {
            "annual_income": float(annual_income),
            "loan_amount": float(loan_amount),
            "interest_rate": float(interest_rate),
            "dti_ratio": float(dti_ratio),
            "fico_score": float(fico_score),
            "credit_line_age_years": float(credit_line_age_years),
            "open_credit_lines": float(open_credit_lines),
            "revolving_utilization": float(revolving_utilization),
            "delinquencies_2yr": float(delinquencies_2yr),
        }

        # ── Score applicant
        X_raw = np.array([[applicant_features[f] for f in FEATURE_COLS]])
        X_scaled = pipeline.transform(X_raw)
        pd_score = float(model.predict_proba(X_scaled)[0, 1])
        band = assign_risk_band(pd_score)

        # SHAP for this applicant
        mean_abs_shap = shap_data["mean_abs_shap"]
        shap_row = mean_abs_shap * np.sign(
            X_scaled[0] - np.median(shap_data["X_sample"], axis=0)
        )
        reasons = generate_adverse_action_reasons(shap_row, FEATURE_COLS, top_n=3)

        st.divider()

        # ── Decision output
        col_res1, col_res2, col_res3 = st.columns(3)

        with col_res1:
            st.markdown('<div class="section-title">Default Probability</div>', unsafe_allow_html=True)
            st.metric(
                label="Predicted PD",
                value=f"{pd_score:.2%}",
                delta=f"{'▲ High Risk' if pd_score > 0.18 else '▼ Manageable Risk'}",
                delta_color="inverse",
            )
            # Gauge chart
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=pd_score * 100,
                    title={"text": "Default Probability (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": band.color},
                        "steps": [
                            {"range": [0, 3], "color": "#2ecc71"},
                            {"range": [3, 8], "color": "#27ae60"},
                            {"range": [8, 18], "color": "#f39c12"},
                            {"range": [18, 35], "color": "#e67e22"},
                            {"range": [35, 100], "color": "#e74c3c"},
                        ],
                    },
                    number={"suffix": "%", "font": {"size": 28}},
                )
            )
            fig_gauge.update_layout(height=260, margin=dict(t=30, b=0, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_res2:
            st.markdown('<div class="section-title">Risk Grade & Decision</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="grade-badge" style="background:{band.color}">'
                f"Grade {band.grade} — {band.label}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**Decision:** {band.action}")

            pd_pct_ranges = [
                ("< 3%", "Grade A — Prime", "#2ecc71"),
                ("3% – 8%", "Grade B — Near-Prime", "#27ae60"),
                ("8% – 18%", "Grade C — Moderate", "#f39c12"),
                ("18% – 35%", "Grade D — High Risk", "#e67e22"),
                ("> 35%", "Grade E — Subprime", "#e74c3c"),
            ]
            st.markdown("---")
            st.markdown("**Grade Reference:**")
            for rng_str, lbl, clr in pd_pct_ranges:
                marker = "◉" if band.label in lbl else "○"
                st.markdown(
                    f'<span style="color:{clr}">{marker}</span> '
                    f'**{rng_str}** — {lbl}',
                    unsafe_allow_html=True,
                )

        with col_res3:
            st.markdown('<div class="section-title">Adverse Action Reasons</div>', unsafe_allow_html=True)
            if band.grade in ("D", "E"):
                st.warning("⚠️ Adverse Action Notice — Top Risk Factors:")
            elif band.grade == "C":
                st.info("ℹ️ Conditional Approval — Key Risk Factors:")
            else:
                st.success("✅ Application within acceptable risk parameters.")

            # SHAP feature importance for this applicant
            shap_feat_df = pd.DataFrame(
                {
                    "Feature": [
                        {
                            "annual_income": "Annual Income",
                            "loan_amount": "Loan Amount",
                            "interest_rate": "Interest Rate",
                            "dti_ratio": "DTI Ratio",
                            "fico_score": "FICO Score",
                            "credit_line_age_years": "Credit Line Age",
                            "open_credit_lines": "Open Credit Lines",
                            "revolving_utilization": "Revolving Util.",
                            "delinquencies_2yr": "Delinquencies 2yr",
                        }.get(f, f)
                        for f in FEATURE_COLS
                    ],
                    "SHAP Impact": shap_row,
                }
            ).sort_values("SHAP Impact", key=abs, ascending=False)

            fig_shap_local = px.bar(
                shap_feat_df.head(9),
                x="SHAP Impact",
                y="Feature",
                orientation="h",
                color="SHAP Impact",
                color_continuous_scale=["#3b82d4", "#f7f8fa", "#e74c3c"],
                color_continuous_midpoint=0,
            )
            fig_shap_local.update_layout(
                margin=dict(t=0, b=0, l=0, r=0), height=280, coloraxis_showscale=False
            )
            st.plotly_chart(fig_shap_local, use_container_width=True)

            if reasons:
                st.markdown("**Specific Reasons:**")
                for i, r in enumerate(reasons, 1):
                    st.markdown(f"{i}. {r}")

    # =======================================================================
    # TAB 4 — Risk Appetite & Policy Simulator
    # =======================================================================
    with tabs[3]:
        st.markdown('<div class="section-title">Portfolio Risk Appetite & Threshold Simulation</div>', unsafe_allow_html=True)
        st.caption(
            "Adjust the approval threshold to explore the trade-off between "
            "portfolio volume, default loss, and net revenue. "
            "Applicants with predicted PD below the threshold are approved."
        )

        sim_threshold = st.slider(
            "Approval Cutoff Threshold (PD %)",
            min_value=1,
            max_value=70,
            value=15,
            step=1,
            help="Applicants with predicted default probability below this value are approved.",
        )
        threshold_val = sim_threshold / 100.0

        loan_amounts_arr = df["loan_amount"].values
        interest_rates_arr = df["interest_rate"].values
        y_true_arr = df["default_status"].values

        # Simulate full sweep
        sim_df = simulate_cutoff_thresholds(
            pd_scores=pd_scores,
            y_true=y_true_arr,
            loan_amounts=loan_amounts_arr,
            interest_rates=interest_rates_arr,
        )

        # Current threshold stats
        current = sim_df[sim_df["threshold"] >= threshold_val].head(1)
        if current.empty:
            current = sim_df.tail(1)
        current_row = current.iloc[0]

        # ── KPI cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Approval Rate", f"{current_row['approval_rate']:.1%}")
        kpi2.metric(
            "Disbursed Volume",
            f"${current_row['disbursed_volume_usd'] / 1e6:.1f}M",
        )
        kpi3.metric(
            "Default Loss",
            f"${current_row['realised_default_loss_usd'] / 1e6:.2f}M",
            delta=f"Portfolio Default Rate: {current_row['portfolio_default_rate']:.1%}",
            delta_color="inverse",
        )
        kpi4.metric(
            "Net Revenue",
            f"${current_row['net_revenue_usd'] / 1e6:.2f}M",
        )

        st.divider()

        col_sim1, col_sim2 = st.columns(2)

        with col_sim1:
            st.markdown('<div class="section-title">Net Revenue vs Approval Threshold</div>', unsafe_allow_html=True)
            fig_rev = go.Figure()
            fig_rev.add_trace(
                go.Scatter(
                    x=sim_df["threshold"] * 100,
                    y=sim_df["net_revenue_usd"] / 1e6,
                    mode="lines",
                    name="Net Revenue",
                    line=dict(color="#3b82d4", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(59,130,212,0.1)",
                )
            )
            fig_rev.add_vline(
                x=sim_threshold,
                line_dash="dash",
                line_color="#e74c3c",
                annotation_text=f"Current: {sim_threshold}%",
            )
            fig_rev.update_layout(
                xaxis_title="Approval Threshold (%)",
                yaxis_title="Net Revenue ($M)",
                margin=dict(t=10, b=10, l=10, r=10),
                height=340,
            )
            st.plotly_chart(fig_rev, use_container_width=True)

        with col_sim2:
            st.markdown('<div class="section-title">Default Loss & Approval Rate vs Threshold</div>', unsafe_allow_html=True)
            fig_dual = go.Figure()
            fig_dual.add_trace(
                go.Scatter(
                    x=sim_df["threshold"] * 100,
                    y=sim_df["realised_default_loss_usd"] / 1e6,
                    mode="lines",
                    name="Default Loss ($M)",
                    line=dict(color="#e74c3c", width=2),
                    yaxis="y1",
                )
            )
            fig_dual.add_trace(
                go.Scatter(
                    x=sim_df["threshold"] * 100,
                    y=sim_df["approval_rate"] * 100,
                    mode="lines",
                    name="Approval Rate (%)",
                    line=dict(color="#2ecc71", width=2, dash="dot"),
                    yaxis="y2",
                )
            )
            fig_dual.add_vline(
                x=sim_threshold,
                line_dash="dash",
                line_color="#57606a",
            )
            fig_dual.update_layout(
                xaxis_title="Approval Threshold (%)",
                yaxis=dict(title="Default Loss ($M)", titlefont=dict(color="#e74c3c")),
                yaxis2=dict(
                    title="Approval Rate (%)",
                    titlefont=dict(color="#2ecc71"),
                    overlaying="y",
                    side="right",
                ),
                margin=dict(t=10, b=10, l=10, r=10),
                height=340,
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_dual, use_container_width=True)

        st.divider()

        # ── Simulation table
        st.markdown('<div class="section-title">Threshold Simulation Table (Selected Rows)</div>', unsafe_allow_html=True)
        display_df = sim_df.copy()
        display_df["threshold_pct"] = (display_df["threshold"] * 100).round(1)
        display_df["approval_rate_pct"] = (display_df["approval_rate"] * 100).round(1)
        display_df["default_rate_pct"] = (display_df["portfolio_default_rate"] * 100).round(2)

        display_cols = {
            "threshold_pct": "Threshold (%)",
            "approval_rate_pct": "Approval Rate (%)",
            "n_approved": "# Approved",
            "disbursed_volume_usd": "Disbursed ($)",
            "realised_default_loss_usd": "Default Loss ($)",
            "net_revenue_usd": "Net Revenue ($)",
            "default_rate_pct": "Default Rate (%)",
        }
        table_df = display_df[list(display_cols.keys())].rename(columns=display_cols)
        # Show every 5th row for readability
        st.dataframe(
            table_df.iloc[::5].style.format(
                {
                    "Disbursed ($)": "${:,.0f}",
                    "Default Loss ($)": "${:,.0f}",
                    "Net Revenue ($)": "${:,.0f}",
                }
            ),
            use_container_width=True,
            height=320,
        )

    # Footer
    st.markdown("---")
    st.caption(
        "CreditGuard v1.0 | Built with Streamlit · XGBoost · SHAP · scikit-learn · imbalanced-learn"
    )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
