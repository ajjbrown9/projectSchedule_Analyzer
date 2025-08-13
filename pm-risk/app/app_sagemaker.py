# app/app_sagemaker.py
# Purpose:
#   Streamlit UI that calls a **deployed SageMaker real-time endpoint** instead of loading a local model.
#   This lets you test your cloud-hosted model with the same UI flow.
# How it works:
#   - Reads endpoint name from env var SAGEMAKER_ENDPOINT_NAME and region from AWS_REGION
#   - Builds a JSON payload: { "features": { <feature_name>: <value>, ... } }
#   - Calls boto3 sagemaker-runtime.invoke_endpoint(...)
#   - Displays probability and class label using the endpoint's configured threshold (from config.json in the model package)
#
# Usage:
#   export SAGEMAKER_ENDPOINT_NAME=<your-endpoint-name>
#   export AWS_REGION=us-east-1
#   streamlit run app/app_sagemaker.py

import os, json, boto3, streamlit as st, pandas as pd

st.set_page_config(page_title="PM Risk (SageMaker Endpoint UI)", page_icon="☁️", layout="wide")
st.title("☁️ Project Schedule Risk — SageMaker Endpoint UI")

ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT_NAME")
REGION = os.getenv("AWS_REGION", "us-east-1")

if not ENDPOINT_NAME:
    st.error("Missing SAGEMAKER_ENDPOINT_NAME. Set it, then rerun this app.")
    st.stop()

@st.cache_resource
def smr_client():
    return boto3.client("sagemaker-runtime", region_name=REGION)

def invoke_endpoint(payload: dict):
    resp = smr_client().invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(payload).encode("utf-8"),
    )
    return json.loads(resp["Body"].read().decode("utf-8"))

st.sidebar.header("Single Prediction")
with st.sidebar.form("single_pred"):
    fields = {}
    numeric_fields = [
        "pm_experience_years","team_size","resource_utilization","overtime_hours","num_dependencies",
        "planned_days","days_elapsed","percent_complete","milestone_delays","critical_path_length",
        "budget_spent_pct","change_requests","open_issues","avg_issue_age_days","historical_on_time_rate",
        "team_turnover_rate","client_response_lag_hours","scope_initial","scope_final",
        "schedule_buffer_ratio","scope_creep_pct","issue_resolution_speed"
    ]
    for col in numeric_fields:
        fields[col] = st.number_input(col, value=0.0)
    fields["project_type"] = st.selectbox("project_type", ["Implementation","Upgrade","Migration","CustomDev"])
    submitted = st.form_submit_button("Predict via Endpoint")

if submitted:
    payload = {"features": fields}
    try:
        res = invoke_endpoint(payload)
        proba = float(res.get("proba", 0.0))
        pred  = int(res.get("prediction", 0))
        thr   = float(res.get("threshold", 0.5))
        st.metric("Risk probability (from endpoint)", f"{proba:.2%}")
        st.caption(f"Decision threshold (endpoint): {thr:.2f}")
        st.write("Prediction:", "🔴 At Risk" if pred==1 else "🟢 On Track")
    except Exception as e:
        st.error(f"Error invoking endpoint: {e}")

st.header("Batch Prediction (CSV -> Endpoint)")
st.caption("Upload a CSV with the same schema as data/sample_projects_large.csv (without the label).")
uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is not None:
    df_up = pd.read_csv(uploaded)
    if "is_late" in df_up.columns:
        df_up = df_up.drop(columns=["is_late"])
    rows = df_up.to_dict(orient="records")
    results = []
    for row in rows:
        try:
            res = invoke_endpoint({"features": row})
            results.append({**row, "risk_proba": res.get("proba"), "prediction": res.get("prediction")})
        except Exception as e:
            results.append({**row, "error": str(e)})
    st.dataframe(pd.DataFrame(results).head(30))
