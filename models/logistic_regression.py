import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from models.utils import fig_to_base64, calc_metrics

def run_logistic(data_bundle, lr_thresh=0.50):
    X_tr_sc = data_bundle['X_train_scaled']
    X_te_sc = data_bundle['X_test_scaled']
    y_tr = data_bundle['y_train']
    y_te = data_bundle['y_test']

    # 1. Fit Multivariate Logit
    logit_model = sm.Logit(y_tr, sm.add_constant(X_tr_sc)).fit(disp=False)
    y_prob = logit_model.predict(sm.add_constant(X_te_sc))
    y_pred = (y_prob >= lr_thresh).astype(int)
    metrics = calc_metrics(y_te, y_pred, y_prob)

    # 2. Sigmoid Response Curve
    var_name = 'Consecutive prior returns'
    fig_sig, ax = plt.subplots(figsize=(6.2, 3.4))
    x_pts = np.linspace(X_tr_sc[var_name].min(), X_tr_sc[var_name].max(), 250)
    x_df = pd.DataFrame(0.0, index=np.arange(len(x_pts)), columns=X_tr_sc.columns)
    x_df[var_name] = x_pts
    ax.plot(x_pts, logit_model.predict(sm.add_constant(x_df)), color='#d62728', lw=2.2, label='Estimated Logistic Curve')
    ax.axhline(y=lr_thresh, color='black', linestyle='--', lw=1.2, label=f'Cutoff ({lr_thresh:.2f})')
    ax.fill_between(x_pts, lr_thresh, 1.0, alpha=0.08, color='red', label='Abuse Region')
    ax.fill_between(x_pts, 0.0, lr_thresh, alpha=0.08, color='green', label='Legitimate Region')
    ax.set_xlabel(f"{var_name} (Standardised)")
    ax.set_ylabel("Predicted P(Abuse)")
    ax.set_title("Logistic Sigmoid Probability Response Curve", fontsize=10, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.6)
    sig_b64 = fig_to_base64(fig_sig)

    # 3. Individual ROC Curve
    fig_roc, ax = plt.subplots(figsize=(5.5, 3.2))
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    ax.plot(fpr, tpr, color='#1f77b4', lw=2, label=f"AUC = {roc_auc_score(y_te, y_prob):.4f}")
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Logistic Regression ROC Curve", fontsize=10, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.6)
    roc_b64 = fig_to_base64(fig_roc)

    # 4. Coefficients Table
    coef_df = pd.DataFrame({
        "Predictor": logit_model.params.index,
        "Coefficient": logit_model.params.values.round(4),
        "Std Error": logit_model.bse.values.round(4),
        "z-stat": logit_model.tvalues.values.round(4),
        "p-value": logit_model.pvalues.values.round(4),
        "Odds Ratio": np.exp(logit_model.params.values).round(4)
    })

    return {
        'model': logit_model,
        'y_prob': y_prob,
        'metrics': metrics,
        'coefs': coef_df.to_dict(orient='records'),
        'plots': {
            'sigmoid': sig_b64,
            'roc_lr': roc_b64
        }
    }