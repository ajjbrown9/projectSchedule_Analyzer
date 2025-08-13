# /src/config.py
# Purpose:
#   Centralized configuration loader for switching the project between
#   LOCAL and SAGEMAKER modes without changing code everywhere.
#   - Reads /src/config.json by default (or APP_CONFIG_PATH env var).
#   - Allows overriding the mode using the MODE env var.
#   - Provides helpers to resolve input/output paths and decision threshold.
#
# Why this matters:
#   In local runs, you read/write from the repo folders.
#   In SageMaker, the platform injects data paths via environment variables:
#     - SM_CHANNEL_TRAIN: where your 'train' channel data is mounted
#     - SM_MODEL_DIR: where you must write trained artifacts
#   We hide those differences behind this module so scripts stay simple.

import json, os
from typing import Any, Dict

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load JSON config from file; apply MODE override if provided via env."""
    path = config_path or os.environ.get("APP_CONFIG_PATH", DEFAULT_CONFIG_PATH)
    with open(path, "r") as f:
        cfg = json.load(f)
    # Optional env override: MODE=sagemaker (useful in SageMaker Estimator.environment)
    env_mode = os.environ.get("MODE")
    if env_mode:
        cfg["mode"] = env_mode.lower()
    if cfg.get("mode") not in ("local", "sagemaker"):
        raise ValueError("config['mode'] must be 'local' or 'sagemaker'.")
    return cfg

def resolve_paths(cfg: Dict[str, Any]):
    """Return (data_path_or_dir, out_dir) based on current mode.
    - LOCAL: read 'local' keys from config.json
    - SAGEMAKER: read env vars defined in 'sagemaker' section (usually SM_CHANNEL_TRAIN / SM_MODEL_DIR)
    """
    mode = cfg["mode"]
    if mode == "local":
        data_path = cfg["local"]["data_path"]
        out_dir   = cfg["local"]["out_dir"]
    else:
        data_path = os.environ.get(cfg["sagemaker"]["train_channel"])
        out_dir   = os.environ.get(cfg["sagemaker"]["model_dir"])
        if not data_path or not out_dir:
            raise EnvironmentError(
                "Missing SageMaker env vars. Ensure your Estimator passes a 'train' channel "
                "and SageMaker has set SM_CHANNEL_TRAIN and SM_MODEL_DIR."
            )
    return data_path, out_dir

def threshold(cfg: Dict[str, Any]) -> float:
    """Decision threshold for converting proba -> class label (default 0.5)."""
    return float(cfg.get("inference", {}).get("threshold", 0.5))
