# /src/inference_configured.py
import json, os, joblib, numpy as np, pandas as pd
from config import load_config, threshold

def model_fn(model_dir):
    model_path = os.path.join(model_dir, 'model.joblib')
    model = joblib.load(model_path)
    try:
        cfg = load_config()
        model._decision_threshold = threshold(cfg)
    except Exception:
        model._decision_threshold = 0.5
    return model

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        payload = json.loads(request_body)
        feats = payload.get('features')
        if isinstance(feats, dict):
            return pd.DataFrame([feats])
        elif isinstance(feats, list):
            return np.array([feats], dtype=float)
    raise ValueError('Unsupported content type or payload; expected application/json with {"features": {...}}')

def predict_fn(input_data, model):
    proba = float(model.predict_proba(input_data)[0, 1])
    thr = getattr(model, '_decision_threshold', 0.5)
    pred = int(proba >= thr)
    return {'prediction': pred, 'proba': proba, 'threshold': thr}

def output_fn(prediction, content_type):
    return json.dumps(prediction)
