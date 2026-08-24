import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.04, dpi=160)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def calc_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "acc": f"{accuracy_score(y_true, y_pred):.4f}",
        "prec": f"{precision_score(y_true, y_pred, zero_division=0):.4f}",
        "rec": f"{recall_score(y_true, y_pred, zero_division=0):.4f}",
        "spec": f"{tn / (tn + fp):.4f}" if (tn + fp) > 0 else "0.0000",
        "f1": f"{f1_score(y_true, y_pred, zero_division=0):.4f}",
        "auc": f"{roc_auc_score(y_true, y_prob):.4f}",
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }

def generate_custom_confusion_matrix(cm, model_title="Model"):
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    total = tn + fp + fn + tp

    # Compact figure bounds with zero dead space
    fig, ax = plt.subplots(figsize=(4.0, 3.8))
    ax.set_xlim(-0.55, 2.05)
    ax.set_ylim(-0.05, 2.45)
    ax.set_aspect('equal')
    ax.axis('off')

    bg_correct = '#fef08a'   # Soft pastel yellow for TP/TN
    bg_error = '#fee2e2'     # Soft pastel blush for FP/FN

    cells = [
        (0, 1, bg_correct, "TP", tp, (tp / total) * 100 if total > 0 else 0),
        (1, 1, bg_error,   "FP", fp, (fp / total) * 100 if total > 0 else 0),
        (0, 0, bg_error,   "FN", fn, (fn / total) * 100 if total > 0 else 0),
        (1, 0, bg_correct, "TN", tn, (tn / total) * 100 if total > 0 else 0),
    ]

    for x, y, bg, code, count, pct in cells:
        rect = Rectangle((x, y), 1, 1, facecolor=bg, edgecolor='#1e293b', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.5, y + 0.64, code, ha='center', va='center',
                fontsize=13, fontweight='bold', color='#0f172a')
        ax.text(x + 0.5, y + 0.40, f"{count:,}", ha='center', va='center',
                fontsize=11, fontweight='bold', color='#1e293b')
        ax.text(x + 0.5, y + 0.20, f"({pct:.1f}%)", ha='center', va='center',
                fontsize=8.0, color='#475569')

    # Top Labels: Actual Values
    ax.text(1.0, 2.32, "Actual Values", ha='center', va='center',
            fontsize=11, fontweight='bold', color='#0f172a')
    ax.text(0.5, 2.12, "Positive (1)", ha='center', va='center',
            fontsize=8.5, fontweight='bold', color='#334155')
    ax.text(1.5, 2.12, "Negative (0)", ha='center', va='center',
            fontsize=8.5, fontweight='bold', color='#334155')

    # Left Labels: Predicted Values
    ax.text(-0.45, 1.0, "Predicted Values", ha='center', va='center',
            rotation=90, fontsize=11, fontweight='bold', color='#0f172a')
    ax.text(-0.15, 1.5, "Positive (1)", ha='center', va='center',
            rotation=90, fontsize=8.5, fontweight='bold', color='#334155')
    ax.text(-0.15, 0.5, "Negative (0)", ha='center', va='center',
            rotation=90, fontsize=8.5, fontweight='bold', color='#334155')

    return fig_to_base64(fig)