# /src/inference_configured.py
# Purpose:
#   Inference handler compatible with SageMaker Hosting (and usable locally).
#   SageMaker looks for these functions:
#     - model_fn(model_dir): load artifacts
#     - input_fn(request_body, request_content_type): parse request
#     - predict_fn(input_data, model): run model
#     - output_fn(prediction, content_type): serialize response
#
# Extras:
#   - Reads decision threshold from /src/config.json (so UI and endpoint agree).
#   - Accepts JSON in the form: {"features": { ... }} or a list of values.

import json, os, joblib, numpy as np, pandas as pd
from config import load_config, threshold

def model_fn(model_dir):
    """Load the fitted pipeline from model_dir (SageMaker provides SM_MODEL_DIR)."""
    model_path = os.path.join(model_dir, 'model.joblib')
    model = joblib.load(model_path)
    # Attach decision threshold for use in predict_fn
    try:
        cfg = load_config()
        model._decision_threshold = threshold(cfg)
    except Exception:
        model._decision_threshold = 0.5
    return model

def input_fn(request_body, request_content_type):
    """Parse the incoming request. We expect application/json with a 'features' key."""
    if request_content_type == 'application/json':
        payload = json.loads(request_body)
        feats = payload.get('features')
        if isinstance(feats, dict):
            return pd.DataFrame([feats])
        elif isinstance(feats, list):
            return np.array([feats], dtype=float)
    raise ValueError('Unsupported content type or payload; expected application/json with {"features": {...}}')

def predict_fn(input_data, model):
    """Run model.predict_proba and apply configured threshold to yield a class label."""
    proba = float(model.predict_proba(input_data)[0, 1])
    thr = getattr(model, '_decision_threshold', 0.5)
    pred = int(proba >= thr)
    return {'prediction': pred, 'proba': proba, 'threshold': thr}

def output_fn(prediction, content_type):
    """Serialize the prediction as JSON string."""
    return json.dumps(prediction)
