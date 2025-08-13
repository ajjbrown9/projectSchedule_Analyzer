# src/train.py - full training script
import argparse, os, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

def load_data(path):
    df = pd.read_csv(path)
    y = df['is_late'].values
    X = df.drop(columns=['is_late'])
    return X, y

def build_preprocessor(X):
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(exclude=['object']).columns.tolist()
    pre = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', StandardScaler(), num_cols)
    ])
    return pre

def main(args):
    X, y = load_data(args.data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pre = build_preprocessor(X)

    # Baseline
    logreg = Pipeline([('pre', pre), ('clf', LogisticRegression(max_iter=200))])
    logreg.fit(X_train, y_train)
    f1_lr = f1_score(y_test, logreg.predict(X_test))

    # XGBoost
    xgb = Pipeline([('pre', pre), ('clf', XGBClassifier(
        n_estimators=300, learning_rate=0.08, max_depth=6,
        subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1, eval_metric='logloss'
    ))])
    xgb.fit(X_train, y_train)
    f1_xgb = f1_score(y_test, xgb.predict(X_test))

    best = xgb if f1_xgb >= f1_lr else logreg

    os.makedirs(args.out, exist_ok=True)
    joblib.dump(best, os.path.join(args.out, 'model.joblib'))
    with open(os.path.join(args.out, 'metrics.txt'), 'w') as f:
        f.write(f'logreg_f1={f1_lr:.4f}\n')
        f.write(f'xgboost_f1={f1_xgb:.4f}\n')
        f.write(f'best={"xgboost" if f1_xgb >= f1_lr else "logreg"}\n')

    print("Saved best model to", os.path.join(args.out, 'model.joblib'))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    main(args)
