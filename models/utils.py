import io
import base64
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)
    return img_b64

def calc_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "acc": f"{accuracy_score(y_true, y_pred):.4f}",
        "prec": f"{precision_score(y_true, y_pred, zero_division=0):.4f}",
        "rec": f"{recall_score(y_true, y_pred, zero_division=0):.4f}",
        "spec": f"{tn / (tn + fp):.4f}",
        "f1": f"{f1_score(y_true, y_pred, zero_division=0):.4f}",
        "auc": f"{roc_auc_score(y_true, y_prob):.4f}",
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)
    }
