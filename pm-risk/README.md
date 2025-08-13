# PM Schedule Risk – Config-Aware ML Project

This repo demonstrates an end-to-end **tabular ML** workflow that runs **locally** and on **AWS SageMaker** using the same codebase.
A simple **config file** (`/src/config.json`) switches behavior between environments.

---

## 📦 What’s in here

```
.
├─ app/
│  └─ app.py                      # Streamlit UI (single + batch prediction, shows threshold)
├─ data/
│  └─ sample_projects_large.csv   # 10k synthetic project-management records
├─ model_artifacts/               # Model is saved here after training (local mode)
├─ sagemaker/
│  ├─ driver.py                   # Upload -> Train -> Deploy -> Test (end-to-end)
│  └─ teardown.py                 # Delete endpoint when done
└─ src/
   ├─ config.json                 # mode=local|sagemaker, local paths, threshold
   ├─ config.py                   # resolves paths + threshold from config/env
   ├─ train_configured.py         # trains LR & XGBoost, saves best model
   ├─ inference_configured.py     # SageMaker-compatible inference handler
   ├─ train.py                    # (optional) earlier non-configured trainer
   └─ requirements.txt            # dependencies for local/Streamlit usage
```

---

## 🚀 Quick Start (Local)

```bash
# 1) Setup
pip install -r src/requirements.txt

# 2) Train locally (reads from config.json: local.data_path)
python src/train_configured.py

# 3) Run the UI
streamlit run app/app.py
```
- The trained pipeline is saved to `./model_artifacts/model.joblib`.
- The UI shows the current decision threshold from `config.json` and supports CSV batch scoring.

### Changing the decision threshold
- Edit `/src/config.json` → `inference.threshold` (e.g., 0.60), re-run the app.
- The endpoint (when deployed) will use the same threshold automatically.

---

## ☁️ Quick Start (SageMaker)

You can run this from your laptop (with AWS credentials) or inside **SageMaker Studio**.

```bash
# From the project root
export AWS_REGION=us-east-1
python sagemaker/driver.py
```
What it does:
1) Uploads `data/sample_projects_large.csv` to S3
2) Launches training with SKLearn Estimator using `src/train_configured.py`
   - Sets `MODE=sagemaker` so the script reads SageMaker env vars
   - SageMaker mounts your 'train' channel at /opt/ml/input/data/train
3) Deploys a real-time endpoint using `src/inference_configured.py`
4) Sends a sample prediction request and prints the response
5) Prints the new **endpoint name** for you to reuse

**IMPORTANT:** When finished, delete the endpoint to stop charges:
```bash
export SAGEMAKER_ENDPOINT_NAME=<printed-endpoint-name>
python sagemaker/teardown.py
```

### Using the Streamlit UI with the endpoint
```bash
export SAGEMAKER_ENDPOINT_NAME=<printed-endpoint-name>
export AWS_REGION=us-east-1
pip install -r src/requirements.txt
streamlit run app/app.py
```
> The UI in this repo loads a local model. If you prefer to query the SageMaker endpoint from Streamlit,
> you can swap to an endpoint-calling app (we can add `app/app_sagemaker.py` upon request).

---

## ⚙️ How the config switch works

- `/src/config.json`
  ```json
  {
    "mode": "local",
    "local": { "data_path": "./data/sample_projects_large.csv", "out_dir": "./model_artifacts" },
    "sagemaker": { "train_channel": "SM_CHANNEL_TRAIN", "model_dir": "SM_MODEL_DIR" },
    "inference": { "threshold": 0.5 }
  }
  ```

- `MODE` env var overrides `mode` without editing the file:
  ```bash
  MODE=sagemaker python sagemaker/driver.py
  ```

- In **LOCAL** mode, paths are taken from `local.*`.
- In **SAGEMAKER** mode, paths are read from SageMaker-provided env vars:
  - `SM_CHANNEL_TRAIN` → `/opt/ml/input/data/train`
  - `SM_MODEL_DIR`     → `/opt/ml/model` (SageMaker uploads contents to S3 after training)

---

## 🧪 API Contract (Inference)

**Request**
```json
{
  "features": {
    "team_size": 12,
    "percent_complete": 0.42,
    "...": "..."
  }
}
```

**Response**
```json
{
  "prediction": 1,
  "proba": 0.83,
  "threshold": 0.50
}
```

- Schema must match training feature names/types.
- Decision threshold comes from `config.json` to keep UI and endpoint consistent.

---

## 🧯 Troubleshooting

### Common training issues
- `FileNotFoundError: No CSV files found`  
  Ensure your S3 path is correct and that `.fit({"train": TrainingInput(...)})` is used.
- `KeyError: 'is_late'`  
  Your CSV must include the label column `is_late`.
- Version mismatch (e.g., sklearn/xgboost)  
  Pin consistent versions in `src/requirements.txt` and match SageMaker `framework_version`.

### Endpoint invocation issues
- `415 Unsupported Media Type` → Set `ContentType="application/json"` for `invoke_endpoint`.
- `500/502` on first requests → Check CloudWatch logs; usually a missing dependency or import error in `inference_configured.py`.
- Wrong payload shape → Ensure you send `{"features": {...}}` (dict of feature_name: value).

---

## 💸 Cost & Cleanup

- Training and endpoints are billed per instance-hour. For dev/testing, keep endpoints up only while needed.
- Use `sagemaker/teardown.py` to delete your endpoint.
- Consider **serverless inference** for sporadic traffic.

---

## 🔁 Extending this project

- Add `app/app_sagemaker.py` to call the endpoint directly from Streamlit.
- Add Model Monitor (data capture + drift) and CI/CD with Model Registry.
- Swap XGBoost for LightGBM or CatBoost; add hyperparameter tuning.

---

## 🛠 Tech Stack

- **Python 3.9+**
- **scikit-learn** — preprocessing pipelines, Logistic Regression
- **XGBoost** — gradient boosting for tabular data
- **pandas / numpy** — data wrangling
- **Streamlit** — simple UI for local and SageMaker endpoint predictions
- **boto3** — AWS SDK for Python (SageMaker endpoint calls)
- **AWS SageMaker** — managed training, model hosting, environment variable injection
- **joblib** — serialization of trained pipelines
- **argparse** — CLI argument parsing for flexible script execution

---

## 🌐 Using the Streamlit UI with SageMaker Endpoint

In addition to the local UI (`app/app.py`), you can run `app/app_sagemaker.py` to
send predictions to your deployed SageMaker endpoint.

```bash
export SAGEMAKER_ENDPOINT_NAME=<your-endpoint-name>
export AWS_REGION=us-east-1
pip install -r src/requirements.txt boto3
streamlit run app/app_sagemaker.py
```
This UI:
- Reads feature inputs interactively
- Sends them as JSON to your endpoint
- Displays the model's prediction, probability, and threshold


---

## 🖥️ Streamlit UI Options

You now have **two** UIs:

1) **Local model UI** (loads `./model_artifacts/model.joblib`)
```bash
streamlit run app/app.py
```

2) **SageMaker endpoint UI** (calls your deployed endpoint via boto3)
```bash
export SAGEMAKER_ENDPOINT_NAME=<printed-endpoint-name>
export AWS_REGION=us-east-1
streamlit run app/app_sagemaker.py
```
- The endpoint UI returns the **probability**, **class prediction**, and the **decision threshold** baked into the model’s config.
- Use this when you want to demo the production path without reloading local artifacts.

---

## 🧰 Tech Stack (What & Why)

- **Python 3.10+** — modern standard library & typing improvements.
- **pandas** — fast CSV IO & tabular manipulation for training/inference.
- **scikit-learn** — preprocessing pipeline (`ColumnTransformer`) and logistic regression baseline.
- **XGBoost** — strong performance on tabular classification; used as the primary model.
- **joblib** — reliable model serialization; saves the whole pipeline cleanly.
- **Streamlit** — rapid UI to demo single & batch scoring locally and against a hosted endpoint.
- **AWS SageMaker (SKLearn)** — managed training & real-time inference with minimal boilerplate.
- **boto3** — AWS SDK used by the Streamlit endpoint UI to invoke the deployed model.
- **argparse + JSON config** — clean, testable configuration with local/SageMaker switching via env vars.

Design choices:
- Keep **one codebase** for local & cloud via `src/config.json` + `MODE` env override.
- Use **pipelines** so preprocessing is identical during training and inference.
- Provide **two UIs** to mirror dev vs. prod usage paths.







# How to Set Up and Troubleshoot Your Python Virtual Environment

## 1. Create a New Virtual Environment (Python 3.11.9)
1. Open a terminal in your project folder:
   cd C:\path\to\your\project

2. Create the virtual environment (replace the path with your actual Python 3.11.9 location):
   "C:\Path\To\Python311\python.exe" -m venv .venv

3. Activate the venv:
   .venv\Scripts\activate

4. Confirm Python version:
   python --version
   # Should display: Python 3.11.9

---

## 2. Install Project Dependencies
1. Upgrade core packaging tools:
   pip install --upgrade pip setuptools wheel

2. Install all requirements:
   pip install -r src\requirements.txt

---

## 3. Troubleshooting
- Wrong Python version:
  Delete `.venv` and recreate it with the correct python.exe path for 3.11.9.
  
- Cannot install a package:
  Check Python version and confirm it’s 64-bit.
  
- Pandas or SHAP fails:
  Use the pinned versions in `requirements.txt` — for Python 3.11.9 they should install via wheels without compiling.

---

## 4. Running the Project
- Train model:
  python src\train_configured.py

- Run local UI:
  streamlit run app\app.py

- Run SageMaker endpoint UI:
  streamlit run app\app_sagemaker.py
