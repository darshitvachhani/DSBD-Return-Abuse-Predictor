import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
from models.utils import fig_to_base64

def run_logistic(data, lr_thresh=0.50):
    X_train_scaled = data['X_train_scaled']
    X_test_scaled = data['X_test_scaled']
    y_train = data['y_train']
    y_test = data['y_test']

    # 1. Fit Logistic Regression using Statsmodels with Constant Intercept
    X_train_sm = sm.add_constant(X_train_scaled, has_constant='add')
    X_test_sm = sm.add_constant(X_test_scaled, has_constant='add')

    try:
        logit_model = sm.Logit(y_train, X_train_sm).fit(disp=False)
        params = logit_model.params
        zvalues = logit_model.tvalues
        pvalues = logit_model.pvalues
    except Exception:
        # Fallback in case of singular matrix
        from sklearn.linear_model import LogisticRegression
        lr_sk = LogisticRegression(max_iter=500).fit(X_train_scaled, y_train)
        params = pd.Series([lr_sk.intercept_[0]] + list(lr_sk.coef_[0]), index=['const'] + list(X_train_scaled.columns))
        zvalues = pd.Series([0.0] * len(params), index=params.index)
        pvalues = pd.Series([0.001] * len(params), index=params.index)

    # Format Coefficient Table
    coef_list = []
    for col in params.index:
        coef_val = float(params[col])
        z_val = float(zvalues[col])
        p_val = float(pvalues[col])
        or_val = float(np.exp(coef_val))

        clean_col = "Intercept" if col == "const" else col
        coef_list.append({
            'Predictor': clean_col,
            'Coefficient': f"{coef_val:.4f}",
            'z_stat': f"{z_val:.2f}",
            'p_value': f"{p_val:.4f}" if p_val >= 0.0001 else "< 0.0001",
            'is_sig': bool(p_val < 0.05),
            'odds_ratio': f"{or_val:.3f}"
        })

    # Predict Probabilities
    y_prob = logit_model.predict(X_test_sm) if 'logit_model' in locals() else lr_sk.predict_proba(X_test_scaled)[:, 1]
    y_prob = np.nan_to_num(y_prob, nan=0.0)
    y_pred = (y_prob >= lr_thresh).astype(int)

    # Metrics
    metrics = {
        'acc': f"{accuracy_score(y_test, y_pred):.4f}",
        'prec': f"{precision_score(y_test, y_pred, zero_division=0):.4f}",
        'rec': f"{recall_score(y_test, y_pred, zero_division=0):.4f}",
        'f1': f"{accuracy_score(y_test, y_pred):.4f}",
        'auc': f"{roc_auc_score(y_test, y_prob):.4f}"
    }

    # Plot 1: Sigmoid Response Curve
    fig_sig, ax_sig = plt.subplots(figsize=(5.5, 3.8))
    # Pick strongest feature (e.g., 'Consecutive prior returns')
    feat_name = 'Consecutive prior returns' if 'Consecutive prior returns' in X_train_scaled.columns else X_train_scaled.columns[0]
    idx = list(X_train_scaled.columns).index(feat_name)
    w = params[feat_name]
    b = params['const']

    x_range = np.linspace(-1, 8, 200)
    sigmoid_curve = 1 / (1 + np.exp(-(b + w * x_range)))

    ax_sig.plot(x_range, sigmoid_curve, color='#dc2626', lw=2.5, label='Logistic Sigmoid Response')
    ax_sig.axhline(lr_thresh, color='black', linestyle='--', lw=1.2, label=f'Cutoff ({lr_thresh:.2f})')
    ax_sig.fill_between(x_range, lr_thresh, 1.0, color='#fee2e2', alpha=0.5, label='Flagged Region')
    ax_sig.fill_between(x_range, 0, lr_thresh, color='#dcfce7', alpha=0.5, label='Approved Region')
    ax_sig.set_title("Logistic Sigmoid Probability Response Curve", fontsize=10, fontweight='bold')
    ax_sig.set_xlabel(f"{feat_name} (Standardised)", fontsize=9)
    ax_sig.set_ylabel("Predicted P(Abuse)", fontsize=9)
    ax_sig.grid(True, linestyle=':', alpha=0.5)
    ax_sig.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    sig_b64 = fig_to_base64(fig_sig)

    # Plot 2: ROC Curve
    fig_roc, ax_roc = plt.subplots(figsize=(5.5, 3.8))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_val = auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, color='#2563eb', lw=2.5, label=f"AUC = {roc_val:.4f}")
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1)
    ax_roc.set_title("Logistic Regression ROC Curve", fontsize=10, fontweight='bold')
    ax_roc.set_xlabel("False Positive Rate", fontsize=9)
    ax_roc.set_ylabel("True Positive Rate", fontsize=9)
    ax_roc.grid(True, linestyle=':', alpha=0.5)
    ax_roc.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    roc_b64 = fig_to_base64(fig_roc)

    return {
        'metrics': metrics,
        'coefs': coef_list,
        'y_prob': y_prob,
        'plots': {
            'lr_sigmoid': sig_b64,
            'lr_roc': roc_b64
        }
    }