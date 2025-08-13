# /src/train_configured.py
import argparse, os, glob, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from config import load_config, resolve_paths

def load_csv(path_or_dir):
    if os.path.isdir(path_or_dir):
        candidates = sorted(glob.glob(os.path.join(path_or_dir, '*.csv')))
        if not candidates:
            raise FileNotFoundError(f'No CSV files found in {path_or_dir}')
        path = candidates[0]
    else:
        path = path_or_dir
    df = pd.read_csv(path)
    y = df['is_late'].values
    X = df.drop(columns=['is_late'])
    return X, y

def build_preprocessor(X):
    cat = X.select_dtypes(include=['object']).columns.tolist()
    num = X.select_dtypes(exclude=['object']).columns.tolist()
    return ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat),
        ('num', StandardScaler(), num)
    ])

def main(args):
    cfg = load_config(args.config)
    data_path, out_dir = resolve_paths(cfg)

    X, y = load_csv(data_path)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pre = build_preprocessor(X)

    lr = Pipeline([('pre', pre), ('clf', LogisticRegression(max_iter=200))])
    lr.fit(X_tr, y_tr)
    f1_lr = f1_score(y_te, lr.predict(X_te))

    xgb = Pipeline([('pre', pre), ('clf', XGBClassifier(
        n_estimators=300, learning_rate=0.08, max_depth=6,
        subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1, eval_metric='logloss'
    ))])
    xgb.fit(X_tr, y_tr)
    f1_xgb = f1_score(y_te, xgb.predict(X_te))

    best = xgb if f1_xgb >= f1_lr else lr

    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(best, os.path.join(out_dir, 'model.joblib'))
    with open(os.path.join(out_dir, 'metrics.txt'), 'w') as f:
        f.write(f'logreg_f1={f1_lr:.4f}\n')
        f.write(f'xgboost_f1={f1_xgb:.4f}\n')
        f.write(f'best={"xgboost" if f1_xgb >= f1_lr else "logreg"}\n')
    print(f"[{cfg['mode']}] Saved model to", os.path.join(out_dir, 'model.joblib'))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None, help='Path to config.json (optional). Defaults to /src/config.json')
    args = ap.parse_args()
    main(args)
