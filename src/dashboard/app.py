# src/dashboard/app.py
"""
ML Experiment Cost & ROI Tracker — Streamlit Dashboard

Supports two data sources:
  1. MLflow (Grid Search) — from runs_features.parquet
  2. Optuna (TPE)         — loaded live from optuna.db
  3. Both combined        — merged for cross-source comparison
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

# ─── Path Setup ───────────────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.analysis.pareto import calculate_pareto_front
from src.analysis.importance import calculate_shap_importance
from src.analysis.budget_optimizer import get_best_model_under_budget
from src.ingestion.optuna_connector import fetch_optuna_runs, get_study_summary
from src.processing.cost_calculator import calculate_run_costs, load_cost_config
from src.processing.feature_builder import build_features

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Cost Tracker",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Constants ────────────────────────────────────────────────────────────────
OPTUNA_DB     = project_root / "optuna.db"
OPTUNA_URI    = f"sqlite:///{OPTUNA_DB.as_posix()}"
STUDY_NAME    = "rf_breast_cancer_tpe"
PARAMS_FILE   = str(project_root / "params.yaml")
PARQUET_PATH  = project_root / "data" / "processed" / "runs_features.parquet"

FEATURE_COLS  = ['n_estimators', 'max_depth', 'min_samples_split']

SOURCE_COLORS = {
    "MLflow (Grid)": "#1f77b4",   # blue
    "Optuna (TPE)":  "#2ca02c",   # green
}
PARETO_COLORS = {
    "Pareto Optimal (Best)":    "#FF8C00",   # orange
    "Dominated (Sub-optimal)":  "#aec7e8",   # light blue
}


# ─── Data Loaders ─────────────────────────────────────────────────────────────
@st.cache_data
def load_mlflow_data() -> pd.DataFrame:
    """Loads and Pareto-tags MLflow grid search data from processed Parquet."""
    if not PARQUET_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(PARQUET_PATH)
    if 'source' not in df.columns:
        df['source'] = 'MLflow (Grid)'
    df_pareto = calculate_pareto_front(df)
    return df_pareto


@st.cache_data
def load_optuna_data(instance_type: str) -> pd.DataFrame:
    """Loads Optuna TPE trial data, runs it through the cost+feature pipeline."""
    if not OPTUNA_DB.exists():
        return pd.DataFrame()

    # Step 1: Fetch raw normalized trials
    df_raw = fetch_optuna_runs(OPTUNA_URI, STUDY_NAME)
    if df_raw.empty:
        return pd.DataFrame()

    # Step 2: Calculate costs (same pipeline as MLflow path)
    df_cost = calculate_run_costs(df_raw, instance_type, PARAMS_FILE)

    # Step 3: Build features (clean + type-fix)
    df_features = build_features(df_cost)

    # Step 4: Tag Pareto front
    df_pareto = calculate_pareto_front(df_features)

    return df_pareto


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://mlflow.org/docs/latest/_static/MLflow-logo-final-black.png", width=120)
    st.title("⚙️ Controls")
    st.markdown("---")

    # Data source selector
    st.subheader("📂 Data Source")
    optuna_available = OPTUNA_DB.exists()

    if optuna_available:
        data_source = st.radio(
            "Select experiment source:",
            ["MLflow (Grid Search)", "Optuna (TPE)", "Both Combined"],
            index=0
        )
    else:
        data_source = "MLflow (Grid Search)"
        st.info("💡 Optuna data not found.\nRun `generate_optuna_runs.py` to enable Optuna insights.")

    st.markdown("---")

    # Instance type selector (affects cost calculation for Optuna)
    st.subheader("💻 Instance Type")
    cost_config = load_cost_config(PARAMS_FILE)
    instance_options = list(cost_config.keys()) if cost_config else ["local_cpu"]
    instance_type = st.selectbox(
        "Cost rate for Optuna data:",
        instance_options,
        index=0,
        help="MLflow data uses the rate from when feature_builder.py was last run."
    )

    st.markdown("---")

    # Optuna study info (if available)
    if optuna_available:
        summary = get_study_summary(OPTUNA_URI, STUDY_NAME)
        if summary:
            st.subheader("🔬 Optuna Study Info")
            st.metric("Total Trials",  summary.get("n_trials", "—"))
            st.metric("Best Accuracy", f"{summary.get('best_value', 0):.4f}")

    st.markdown("---")
    if st.button("Refresh Data", help="Clear cache and re-fetch data from tracking DBs"):
        st.cache_data.clear()
        st.rerun()


# ─── Load Data Based on Source Selection ──────────────────────────────────────
df_mlflow = pd.DataFrame()
df_optuna  = pd.DataFrame()

if data_source in ["MLflow (Grid Search)", "Both Combined"]:
    df_mlflow = load_mlflow_data()

if data_source in ["Optuna (TPE)", "Both Combined"] and optuna_available:
    df_optuna = load_optuna_data(instance_type)

# Merge if needed
if data_source == "Both Combined":
    if not df_mlflow.empty and not df_optuna.empty:
        # Re-compute Pareto on the combined set
        df = pd.concat([df_mlflow, df_optuna], ignore_index=True)
        df = calculate_pareto_front(df)
    elif not df_mlflow.empty:
        df = df_mlflow
    else:
        df = df_optuna
elif data_source == "Optuna (TPE)":
    df = df_optuna
else:
    df = df_mlflow


# ─── Header ───────────────────────────────────────────────────────────────────
st.title("💸 ML Experiment Cost & ROI Tracker")
st.markdown(
    f"Analyzing **{data_source}** — "
    "Visualize the tradeoff between model accuracy and compute cost."
)

if df.empty:
    if data_source == "Optuna (TPE)":
        st.error("No Optuna data found! Run `python generate_optuna_runs.py` first.")
    else:
        st.error("No MLflow data found! Run `python src/processing/feature_builder.py` first.")
    st.stop()


# ─── Top-Level KPI Metrics ────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Experiments",    len(df))
col2.metric("Best Accuracy",        f"{df['accuracy'].max():.4f}")
col3.metric("Total Cost (Est.)",    f"${df['estimated_cost_usd'].sum():.6f}")
col4.metric("Pareto Optimal Runs",  int(df['is_pareto_optimal'].sum()))
if 'source' in df.columns and data_source == "Both Combined":
    optuna_count = int((df['source'] == 'Optuna (TPE)').sum())
    col5.metric("Optuna Trials", optuna_count)
else:
    col5.metric("Avg Accuracy",  f"{df['accuracy'].mean():.4f}")

st.divider()


# ─── Main Layout: Scatter + SHAP ──────────────────────────────────────────────
left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("🎯 Cost vs. Accuracy")

    # Choose coloring strategy
    if data_source == "Both Combined" and 'source' in df.columns:
        # Color by data source — shows how well each method explored the space
        st.markdown("Points colored by **data source** (Grid Search vs TPE). "
                    "Pareto-optimal runs are marked with a ⭐ star outline.")

        df['Pareto'] = df['is_pareto_optimal'].map(
            {True: '⭐ Pareto Optimal', False: 'Dominated'}
        )
        fig = px.scatter(
            df,
            x="estimated_cost_usd",
            y="accuracy",
            color="source",
            symbol="Pareto",
            color_discrete_map=SOURCE_COLORS,
            symbol_map={"⭐ Pareto Optimal": "star", "Dominated": "circle"},
            hover_data=["run_id", "n_estimators", "max_depth", "min_samples_split",
                        "is_pareto_optimal"],
            title="Grid Search vs Optuna TPE — Exploration Comparison",
            labels={"estimated_cost_usd": "Estimated Cost (USD)",
                    "accuracy": "Validation Accuracy"}
        )
    else:
        # Color by Pareto optimality — the default single-source view
        st.markdown("**Orange** = Pareto optimal (no other run is both cheaper AND more accurate). "
                    "**Blue** = dominated.")
        df['Efficiency'] = df['is_pareto_optimal'].map(
            {True: 'Pareto Optimal (Best)', False: 'Dominated (Sub-optimal)'}
        )
        fig = px.scatter(
            df,
            x="estimated_cost_usd",
            y="accuracy",
            color="Efficiency",
            color_discrete_map=PARETO_COLORS,
            hover_data=["run_id", "n_estimators", "max_depth", "min_samples_split"],
            title="Experiment ROI Sandbox",
            labels={"estimated_cost_usd": "Estimated Cost (USD)",
                    "accuracy": "Validation Accuracy"}
        )

    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

with right_col:
    st.subheader("🧠 Hyperparameter Impact (SHAP)")
    st.markdown("A Random Forest meta-model trained on your experiment history, "
                "explained by SHAP.")

    @st.cache_data
    def get_shap_importance(df_hash, features, target):
        """Cached SHAP computation — only recalculates when data changes."""
        return calculate_shap_importance(df, features, target)

    with st.spinner("Calculating SHAP values..."):
        # Build feature list dynamically (use what's available)
        available_features = [f for f in FEATURE_COLS if f in df.columns]

        # Use a hashable key so cache invalidates when data changes
        df_hash = len(df)
        _, _, importance_df = get_shap_importance(df_hash, available_features, 'accuracy')

    if importance_df is not None:
        fig_shap = px.bar(
            importance_df,
            x="Mean_Absolute_SHAP",
            y="Hyperparameter",
            orientation='h',
            title="SHAP Feature Importance",
            color="Mean_Absolute_SHAP",
            color_continuous_scale="Oranges",
            labels={"Mean_Absolute_SHAP": "Mean |SHAP| Impact on Accuracy"}
        )
        fig_shap.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_shap, use_container_width=True)
    else:
        st.warning("Not enough runs to compute SHAP (need ≥ 5).")

st.divider()


# ─── Optuna Convergence Chart (only when Optuna data is loaded) ───────────────
if not df_optuna.empty and 'trial_number' in df_optuna.columns:
    st.subheader("📈 Optuna Convergence — How the TPE Sampler Learned")
    st.markdown(
        "Each point is one Optuna trial. The **green line** shows the best accuracy "
        "achieved *so far* — you can watch TPE hone in on the optimal region over time."
    )

    df_conv = df_optuna.sort_values('trial_number').copy()
    df_conv['best_so_far'] = df_conv['accuracy'].cummax()

    fig_conv = go.Figure()

    # Individual trial dots
    fig_conv.add_trace(go.Scatter(
        x=df_conv['trial_number'],
        y=df_conv['accuracy'],
        mode='markers',
        name='Individual Trial',
        marker=dict(color='#2ca02c', size=7, opacity=0.6,
                    line=dict(width=1, color='darkgreen')),
        hovertemplate="Trial %{x}<br>Accuracy: %{y:.4f}<extra></extra>"
    ))

    # Best-so-far line
    fig_conv.add_trace(go.Scatter(
        x=df_conv['trial_number'],
        y=df_conv['best_so_far'],
        mode='lines',
        name='Best Accuracy So Far',
        line=dict(color='#FF8C00', width=2.5, dash='solid'),
        hovertemplate="Trial %{x}<br>Best So Far: %{y:.4f}<extra></extra>"
    ))

    fig_conv.update_layout(
        title="Optuna TPE Convergence Curve",
        xaxis_title="Trial Number",
        yaxis_title="Validation Accuracy",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    st.plotly_chart(fig_conv, use_container_width=True)

    # Show best trial details
    best_trial = df_conv.loc[df_conv['accuracy'].idxmax()]
    with st.expander("🏆 Best Optuna Trial Details"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Trial #",          int(best_trial.get('trial_number', 0)))
        c2.metric("Accuracy",         f"{best_trial['accuracy']:.4f}")
        c3.metric("Cost (Est.)",      f"${best_trial['estimated_cost_usd']:.7f}")
        c4.metric("n_estimators",     int(best_trial.get('n_estimators', 0)))
        if 'max_depth' in best_trial:
            st.write(f"**max_depth:** {int(best_trial['max_depth'])}")
        if 'min_samples_split' in best_trial:
            st.write(f"**min_samples_split:** {int(best_trial['min_samples_split'])}")

    st.divider()


# ─── Budget Optimizer ─────────────────────────────────────────────────────────
st.subheader("💰 The Budget Optimizer")
st.markdown("Set a strict budget cap — the engine recommends the best model you can afford.")

max_cost = float(df['estimated_cost_usd'].max())
min_cost = float(df['estimated_cost_usd'].min())

selected_budget = st.slider(
    "Set Maximum Compute Budget ($)",
    min_value=min_cost,
    max_value=max_cost,
    value=max_cost / 2,
    format="%.7f"
)

best_run = get_best_model_under_budget(df, selected_budget)

if best_run is not None:
    src_label = best_run.get('source', 'Unknown')
    st.success(
        f"**Optimal Model Found!** [{src_label}] — "
        f"Achieved **{best_run['accuracy']:.4f}** accuracy "
        f"for **${best_run['estimated_cost_usd']:.7f}**."
    )

    rc1, rc2 = st.columns(2)
    with rc1:
        st.json({
            "Run ID": best_run['run_id'],
            "Source": src_label,
            "Hyperparameters": {
                "n_estimators":      int(best_run.get('n_estimators', 0)),
                "max_depth":         int(best_run.get('max_depth', 0)),
                "min_samples_split": int(best_run.get('min_samples_split', 0)),
            }
        })
    with rc2:
        # Show how many runs are affordable at this budget
        affordable = df[df['estimated_cost_usd'] <= selected_budget]
        fig_budget = px.histogram(
            affordable,
            x="accuracy",
            nbins=20,
            title=f"Accuracy Distribution (within ${selected_budget:.7f} budget)",
            color_discrete_sequence=["#2ca02c"],
            labels={"accuracy": "Validation Accuracy", "count": "# Runs"}
        )
        fig_budget.add_vline(
            x=best_run['accuracy'],
            line_dash="dash",
            line_color="orange",
            annotation_text="Best",
            annotation_position="top left"
        )
        st.plotly_chart(fig_budget, use_container_width=True)
else:
    st.warning("No models found under this budget. Try increasing the slider.")

st.divider()


# ─── Full Run Table ───────────────────────────────────────────────────────────
with st.expander("📋 Full Experiment Table"):
    display_cols = [c for c in [
        'run_id', 'source', 'accuracy', 'estimated_cost_usd',
        'n_estimators', 'max_depth', 'min_samples_split',
        'duration_hours', 'is_pareto_optimal'
    ] if c in df.columns]

    st.dataframe(
        df[display_cols].sort_values('accuracy', ascending=False),
        use_container_width=True
    )
    st.download_button(
        label="⬇️ Download as CSV",
        data=df[display_cols].to_csv(index=False),
        file_name="ml_experiment_results.csv",
        mime="text/csv"
    )