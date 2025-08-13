# sagemaker/teardown.py
"""
Delete the deployed endpoint (and optionally the endpoint config and model).
Run when you're done testing to avoid ongoing charges.
"""

import os, boto3

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT_NAME")  # or hardcode if preferred

if not ENDPOINT_NAME:
    raise ValueError("Set SAGEMAKER_ENDPOINT_NAME env var to the endpoint you want to delete.")

sm = boto3.client("sagemaker", region_name=AWS_REGION)

print("Deleting endpoint:", ENDPOINT_NAME)
sm.delete_endpoint(EndpointName=ENDPOINT_NAME)

# Optional: if you created names programmatically and want to clean fully, uncomment:
# sm.delete_endpoint_config(EndpointConfigName=ENDPOINT_NAME)
# sm.delete_model(ModelName="<your-model-name>")
print("✅ Deleted endpoint. (EndpointConfig/Model not deleted by default.)")


# Notes:
# - This deletes the endpoint only. If you want to delete the EndpointConfig and Model objects,
#   uncomment the lines at the bottom and set the correct names.
# - Always clean up deployed endpoints after testing to avoid per-hour charges.
