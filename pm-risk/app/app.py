# app/app.py
# Purpose:
#   Lightweight UI for single and batch predictions.
#   - LOCAL mode: loads model_artifacts/model.joblib directly.
#   - SAGEMAKER mode (optional): set env var SAGEMAKER_ENDPOINT_NAME and use a different UI file if desired.
#   This app shows the configured decision threshold from /src/config.json for consistency with the endpoint.

# --- ensure project root is on sys.path so "import src.*" works ---
import sys, os
ROOT = os.path.dirname(os.path.dirname(__file__))  # project root = parent of app/
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# -----------------------------------------------------------------

from src.config import load_config, threshold

# app/app.py - full Streamlit app
import streamlit as st, joblib, pandas as pd, numpy as np, os

st.set_page_config(page_title="PM Schedule Risk", page_icon="📊", layout="wide")
st.title("📊 Project Schedule Risk Predictor (Local Model)")

@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(__file__), "..", "model_artifacts", "model.joblib")
    if not os.path.exists(path):
        st.warning("Model not found. Train it first: `python src/train.py --data ./data/sample_projects_large.csv --out ./model_artifacts`")
        return None
    return joblib.load(path)

model = load_model()
from src.config import load_config, threshold
cfg = load_config(); thr = threshold(cfg)
st.caption(f'Decision threshold: {thr:.2f}')

st.sidebar.header("Single Prediction")
with st.sidebar.form("single_pred"):
    pm_experience_years = st.number_input("PM experience (years)", 0, 50, 5)
    team_size = st.number_input("Team size", 1, 200, 10)
    resource_utilization = st.slider("Resource utilization", 0.0, 1.5, 0.8, 0.01)
    overtime_hours = st.number_input("Overtime hours", 0, 200, 5)
    num_dependencies = st.number_input("Number of dependencies", 0, 200, 5)
    planned_days = st.number_input("Planned days", 1, 1000, 120)
    days_elapsed = st.number_input("Days elapsed", 1, 1000, 60)
    percent_complete = st.slider("Percent complete", 0.0, 1.0, 0.45, 0.01)
    milestone_delays = st.number_input("Milestone delays", 0, 50, 1)
    critical_path_length = st.number_input("Critical path length", 1, 500, 20)
    budget_spent_pct = st.slider("Budget spent %", 0.0, 1.5, 0.55, 0.01)
    change_requests = st.number_input("Change requests", 0, 100, 1)
    open_issues = st.number_input("Open issues", 0, 500, 4)
    avg_issue_age_days = st.number_input("Avg issue age (days)", 0, 365, 10)
    historical_on_time_rate = st.slider("Historical on-time rate", 0.0, 1.0, 0.7, 0.01)
    team_turnover_rate = st.slider("Team turnover rate", 0.0, 1.0, 0.05, 0.01)
    client_response_lag_hours = st.number_input("Client response lag (hours)", 0, 1000, 18)
    scope_initial = st.number_input("Initial scope items", 1, 10000, 200)
    scope_final = st.number_input("Final scope items", 1, 10000, 220)
    project_type = st.selectbox("Project type", ["Implementation","Upgrade","Migration","CustomDev"])
    schedule_buffer_ratio = st.slider("Schedule buffer ratio", -1.0, 1.0, 0.5, 0.01)
    scope_creep_pct = st.slider("Scope creep %", -0.5, 1.5, 0.1, 0.01)
    issue_resolution_speed = st.number_input("Issue resolution speed", 0.0, 100.0, 2.0, 0.1)
    submitted = st.form_submit_button("Predict")

if model and submitted:
    row = {
        "pm_experience_years": pm_experience_years,
        "team_size": team_size,
        "resource_utilization": resource_utilization,
        "overtime_hours": overtime_hours,
        "num_dependencies": num_dependencies,
        "planned_days": planned_days,
        "days_elapsed": days_elapsed,
        "percent_complete": percent_complete,
        "milestone_delays": milestone_delays,
        "critical_path_length": critical_path_length,
        "budget_spent_pct": budget_spent_pct,
        "change_requests": change_requests,
        "open_issues": open_issues,
        "avg_issue_age_days": avg_issue_age_days,
        "historical_on_time_rate": historical_on_time_rate,
        "team_turnover_rate": team_turnover_rate,
        "client_response_lag_hours": client_response_lag_hours,
        "scope_initial": scope_initial,
        "scope_final": scope_final,
        "project_type": project_type,
        "schedule_buffer_ratio": schedule_buffer_ratio,
        "scope_creep_pct": scope_creep_pct,
        "issue_resolution_speed": issue_resolution_speed
    }
    df = pd.DataFrame([row])
    proba = float(model.predict_proba(df)[0,1])
    pred = int(proba >= 0.5)
    st.metric("Predicted Risk (probability of being late)", f"{proba:.2%}")
    st.write("Prediction:", "🔴 At Risk" if pred==1 else "🟢 On Track")

st.header("Batch Prediction (CSV Upload)")
st.caption("Upload a CSV with the same columns as data/sample_projects_large.csv (without the is_late label).")
uploaded = st.file_uploader("Upload CSV", type=["csv"])
if model and uploaded is not None:
    df_up = pd.read_csv(uploaded)
    if "is_late" in df_up.columns:
        df_up = df_up.drop(columns=["is_late"])
    preds = model.predict_proba(df_up)[:,1]
    out = df_up.copy()
    out["risk_proba"] = preds
    st.dataframe(out.head(30))
    st.download_button("Download predictions", out.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")
