import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from models.utils import fig_to_base64

def run_cost_analysis(y_test, y_prob_bag, c_fp=50.0, c_fn=25.0):
    thresholds = np.linspace(0.05, 0.95, 100)
    costs = []
    for t in thresholds:
        p_pred = (y_prob_bag >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, p_pred).ravel()
        costs.append((fp * c_fp) + (fn * c_fn))

    opt_idx = np.argmin(costs)
    opt_t = thresholds[opt_idx]
    min_c = costs[opt_idx]

    fig_c, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(thresholds, costs, color='#d95f02', lw=2.2, label='Expected Total Cost')
    ax.axvline(x=opt_t, color='black', linestyle='--', label=f"Opt Cutoff ({opt_t:.2f})")
    ax.scatter([opt_t], [min_c], color='red', s=60, zorder=5, label=f"Min Cost (£{min_c:,.0f})")
    ax.set_xlabel("Cutoff Threshold")
    ax.set_ylabel("Expected Cost (£)")
    ax.legend(loc='upper center', fontsize=8)
    cost_b64 = fig_to_base64(fig_c)

    return {
        'opt_thresh': f"{opt_t:.2f}",
        'min_cost': f"£{min_c:,.2f}",
        'plots': {'cost_curve': cost_b64}
    }

def generate_combined_roc(y_test, probs_dict):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    colors = {'Logistic Reg.': '#1f77b4', 'CART Tree': '#ff7f0e', 'Random Forest': '#2ca02c', 'Bagging': '#9467bd'}
    for name, prob in probs_dict.items():
        fpr, tpr, _ = roc_curve(y_test, prob)
        ax.plot(fpr, tpr, color=colors[name], lw=2, label=f"{name} (AUC = {roc_auc_score(y_test, prob):.4f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Master Combined ROC", fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    return fig_to_base64(fig)
