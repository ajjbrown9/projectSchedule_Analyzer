# sagemaker_driver.py
import os
import sagemaker
from sagemaker.sklearn import SKLearn
from sagemaker.inputs import TrainingInput

# ===== USER CONFIGURATION =====
LOCAL_DATA_PATH = "data/sample_projects_large.csv"  # local CSV to upload
S3_BUCKET = None  # if None, will use default SageMaker bucket
PREFIX = "pm-risk-demo"  # S3 prefix
INSTANCE_TYPE = "ml.m5.large"
FRAMEWORK_VERSION = "1.2-1"
PY_VERSION = "py3"
# ==============================

# Initialize SageMaker session and role
sess = sagemaker.Session()
role = sagemaker.get_execution_role()
bucket = S3_BUCKET or sess.default_bucket()

# Upload dataset to S3
print(f"Uploading {LOCAL_DATA_PATH} to s3://{bucket}/{PREFIX}/train ...")
s3_train = sess.upload_data(LOCAL_DATA_PATH, bucket=bucket, key_prefix=f"{PREFIX}/train")

# Create a TrainingInput for the 'train' channel
train_input = TrainingInput(s3_data=s3_train, content_type="text/csv")

# Create Estimator
estimator = SKLearn(
    entry_point="train_configured.py",
    source_dir="src",
    role=role,
    instance_type=INSTANCE_TYPE,
    instance_count=1,
    framework_version=FRAMEWORK_VERSION,
    py_version=PY_VERSION,
    base_job_name="pm-risk-train",
    environment={"MODE": "sagemaker"}  # tell config to use SageMaker mode
)

# Fit the model
print("Starting training job...")
estimator.fit({"train": train_input})

print("Training job complete. Model artifacts uploaded to:")
print(estimator.model_data)
