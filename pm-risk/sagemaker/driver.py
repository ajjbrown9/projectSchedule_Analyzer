# sagemaker/driver.py
"""
SageMaker driver script for the config-aware project.

What it does:
1) Uploads your local CSV to S3
2) Launches training with SKLearn Estimator (MODE=sagemaker)
3) Deploys a real-time endpoint
4) Shows how to invoke and (optionally) clean up

Run this from the project root or ./sagemaker with AWS credentials configured.
"""

import os, json, time, boto3, sagemaker
from sagemaker.sklearn import SKLearn, SKLearnModel
from sagemaker.inputs import TrainingInput

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
sess = sagemaker.Session()
sm = boto3.client("sagemaker", region_name=AWS_REGION)

# 0) Resolve role
try:
    ROLE_ARN = sagemaker.get_execution_role()  # Works inside Studio/Notebook Instances
except Exception:
    # Fallback: replace with a real role ARN in your account with SageMaker + S3 permissions
    ROLE_ARN = "arn:aws:iam::<your-account-id>:role/data_engineer"

# 1) Upload local CSV to S3
bucket = sess.default_bucket()
prefix = "pm-risk-config-demo"
local_csv = os.path.join(os.path.dirname(__file__), "..", "data", "sample_projects_large.csv")
if not os.path.exists(local_csv):
    raise FileNotFoundError(f"Local CSV not found at {local_csv}")

s3_uri = sess.upload_data(local_csv, bucket=bucket, key_prefix=f"{prefix}/train")
print("Uploaded training data to:", s3_uri)

# 2) Configure Estimator (uses train_configured.py, env MODE=sagemaker)
est = SKLearn(
    entry_point="train_configured.py",
    source_dir=os.path.join(os.path.dirname(__file__), "..", "src"),
    role=ROLE_ARN,
    instance_type="ml.m5.large",
    instance_count=1,
    framework_version="1.2-1",
    py_version="py3",
    environment={"MODE": "sagemaker"},      # <-- tells the code to read SM_* vars
    base_job_name="pm-risk-train-config",
    sagemaker_session=sess
)

train_input = TrainingInput(s3_data=s3_uri, content_type="text/csv")
est.fit({"train": train_input})
print("Model artifact:", est.model_data)

# 3) Register model object for deployment (uses inference_configured.py)
model = SKLearnModel(
    model_data=est.model_data,
    role=ROLE_ARN,
    framework_version="1.2-1",
    entry_point="inference_configured.py",
    source_dir=os.path.join(os.path.dirname(__file__), "..", "src"),
    name=f"pm-risk-config-model-{int(time.time())}",
    sagemaker_session=sess
)

endpoint_name = f"pm-risk-config-endpoint-{int(time.time())}"
predictor = model.deploy(
    instance_type="ml.m5.large",
    initial_instance_count=1,
    endpoint_name=endpoint_name
)
print("✅ Deployed endpoint:", endpoint_name)

# 4) Example invoke (single row)
smr = boto3.client("sagemaker-runtime", region_name=AWS_REGION)
example = {
    "features": {
        "pm_experience_years": 5,
        "team_size": 12,
        "resource_utilization": 0.78,
        "overtime_hours": 6,
        "num_dependencies": 8,
        "planned_days": 180,
        "days_elapsed": 90,
        "percent_complete": 0.42,
        "milestone_delays": 1,
        "critical_path_length": 22,
        "budget_spent_pct": 0.58,
        "change_requests": 2,
        "open_issues": 7,
        "avg_issue_age_days": 11,
        "historical_on_time_rate": 0.73,
        "team_turnover_rate": 0.05,
        "client_response_lag_hours": 10,
        "scope_initial": 250,
        "scope_final": 260,
        "project_type": "Implementation",
        "schedule_buffer_ratio": 0.3,
        "scope_creep_pct": 0.04,
        "issue_resolution_speed": 2.0
    }
}

import json
resp = smr.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="application/json",
    Body=json.dumps(example).encode("utf-8")
)
print("Sample prediction:", resp["Body"].read().decode("utf-8"))

print("\nNext steps:")
print(" - To point the Streamlit UI at this endpoint:")
print("     export SAGEMAKER_ENDPOINT_NAME=" + endpoint_name)
print("     export AWS_REGION=" + AWS_REGION)
print("     streamlit run app/app.py")
print(" - To delete the endpoint when done (avoid charges), run sagemaker/teardown.py.")
