# /src/config.py
import json, os
from typing import Any, Dict

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

def load_config(config_path: str = None) -> Dict[str, Any]:
    path = config_path or os.environ.get("APP_CONFIG_PATH", DEFAULT_CONFIG_PATH)
    with open(path, "r") as f:
        cfg = json.load(f)
    env_mode = os.environ.get("MODE")
    if env_mode:
        cfg["mode"] = env_mode.lower()
    if cfg.get("mode") not in ("local", "sagemaker"):
        raise ValueError("config['mode'] must be 'local' or 'sagemaker'.")
    return cfg

def resolve_paths(cfg: Dict[str, Any]):
    mode = cfg["mode"]
    if mode == "local":
        data_path = cfg["local"]["data_path"]
        out_dir   = cfg["local"]["out_dir"]
    else:
        data_path = os.environ.get(cfg["sagemaker"]["train_channel"])
        out_dir   = os.environ.get(cfg["sagemaker"]["model_dir"])
        if not data_path or not out_dir:
            raise EnvironmentError("Missing SageMaker env vars for channels/dirs.")
    return data_path, out_dir

def threshold(cfg: Dict[str, Any]) -> float:
    return float(cfg.get("inference", {}).get("threshold", 0.5))
