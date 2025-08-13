# src/explainability_demo.py - SHAP demo
import argparse, os, joblib, pandas as pd, shap, matplotlib.pyplot as plt

def main(args):
    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.data)
    X = df.drop(columns=['is_late'])

    model = joblib.load(args.model)
    final_est = getattr(model, 'named_steps', {}).get('clf', model)

    try:
        if final_est.__class__.__name__.lower().startswith('xgb'):
            X_trans = model.named_steps['pre'].fit_transform(X)
            explainer = shap.TreeExplainer(final_est)
            shap_values = explainer.shap_values(X_trans)
            feature_names = list(model.named_steps['pre'].get_feature_names_out())
        else:
            raise Exception("Fallback to Kernel")
    except Exception:
        sample = X.sample(min(400, len(X)), random_state=42)
        f = lambda input_df: model.predict_proba(input_df)[:,1]
        explainer = shap.KernelExplainer(f, sample, link='logit')
        shap_values = explainer.shap_values(sample, nsamples=100)
        X_trans = sample
        feature_names = list(sample.columns)

    shap.summary_plot(shap_values, X_trans, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "shap_summary.png"), bbox_inches="tight")
    plt.close()

    shap.force_plot(getattr(explainer, 'expected_value', 0), shap_values[0], X_trans[0], matplotlib=True, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "shap_force_example.png"), bbox_inches="tight")
    plt.close()

    print("Saved SHAP plots to", args.out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--out', default='./explainability_artifacts')
    args = p.parse_args()
    main(args)
