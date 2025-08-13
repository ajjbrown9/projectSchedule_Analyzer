# src/inference.py - full local/SageMaker-compatible handler
import json, os, joblib, numpy as np

def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model.joblib")
    return joblib.load(model_path)

def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        payload = json.loads(request_body)
        feats = payload.get("features")
        if isinstance(feats, dict):
            import pandas as pd
            return pd.DataFrame([feats])
        elif isinstance(feats, list):
            import numpy as np
            return np.array([feats], dtype=float)
        else:
            raise ValueError("features must be a dict (preferred) or list")
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    proba = float(model.predict_proba(input_data)[0, 1])
    pred = int(proba >= 0.5)
    return {"prediction": pred, "proba": proba}

def output_fn(prediction, content_type):
    return json.dumps(prediction)
