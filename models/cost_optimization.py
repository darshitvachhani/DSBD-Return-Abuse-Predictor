import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from models.utils import fig_to_base64

def run_cost_analysis(y_true, y_probs, c_fp=50.0, c_fn=25.0):
    """
    Computes expected total business loss across threshold range [0.05, 0.95]:
    Expected Cost(t) = C_FP * FP(t) + C_FN * FN(t)
    """
    thresholds = np.linspace(0.05, 0.95, 100)
    costs = []
    fp_counts = []
    fn_counts = []

    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        total_cost = (c_fp * fp) + (c_fn * fn)
        costs.append(total_cost)
        fp_counts.append(fp)
        fn_counts.append(fn)

    costs = np.array(costs)
    opt_idx = np.argmin(costs)
    opt_thresh = round(float(thresholds[opt_idx]), 2)
    min_cost = float(costs[opt_idx])

    # Generate Cost Curve Plot
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(thresholds, costs, color='#d95f02', lw=2.5, label='Expected Total Loss (£)')
    ax.axvline(opt_thresh, color='black', linestyle='--', lw=1.5, label=f'Optimal Cutoff (t* = {opt_thresh})')
    ax.scatter([opt_thresh], [min_cost], color='red', s=90, zorder=5, label=f'Min Loss (£{min_cost:,.2f})')

    ax.set_title("Empirical Cost-Sensitive Threshold Curve", fontsize=10, fontweight='bold')
    ax.set_xlabel("Classification Cutoff Threshold (t)", fontsize=9)
    ax.set_ylabel("Expected Aggregate Cost (£)", fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()

    return {
        'plots': {
            'cost_curve': fig_to_base64(fig)
        },
        'opt_thresh': opt_thresh,
        'min_cost': f"£{min_cost:,.2f}",
        'c_fp': c_fp,
        'c_fn': c_fn,
        'opt_fp': fp_counts[opt_idx],
        'opt_fn': fn_counts[opt_idx]
    }

def generate_combined_roc(y_true, model_probs_dict):
    """Generates the multi-model combined ROC benchmark chart."""
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']

    for (name, probs), col in zip(model_probs_dict.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=col, label=f"{name} (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_title("Multi-Model ROC Comparison Curve", fontsize=10, fontweight='bold')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()

    return fig_to_base64(fig)