import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from models.utils import fig_to_base64, generate_custom_confusion_matrix

def run_random_forest(data, n_trees=100, depth=8, rf_thresh=0.50):
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']

    rf = RandomForestClassifier(
        n_estimators=int(n_trees),
        max_depth=int(depth),
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    y_prob = rf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= float(rf_thresh)).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc_val = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    # 1. Representative Tree Visualization
    fig_tree, ax_tree = plt.subplots(figsize=(10, 4.2))
    plot_tree(
        rf.estimators_[0],
        feature_names=X_train.columns.tolist(),
        class_names=['Legit', 'Abuse'],
        filled=True,
        rounded=True,
        fontsize=6,
        ax=ax_tree,
        max_depth=3,
        proportion=True
    )
    ax_tree.set_title("Random Forest: Representative Tree (Top Splits)", fontsize=9, fontweight='bold')
    plt.tight_layout()
    plot_tree_b64 = fig_to_base64(fig_tree)

    # 2. Confusion Matrix Plot
    plot_cm_b64 = generate_custom_confusion_matrix(cm, "Random Forest")

    # 3. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig_roc, ax_roc = plt.subplots(figsize=(5.2, 3.8))
    ax_roc.plot(fpr, tpr, color='#16a34a', lw=2, label=f'Random Forest (AUC = {auc_val:.4f})')
    ax_roc.plot([0, 1], [0, 1], color='#64748b', linestyle='--', lw=1)
    ax_roc.set_xlabel('1 - Specificity (False Positive Rate)', fontsize=8.5)
    ax_roc.set_ylabel('Sensitivity (True Positive Rate)', fontsize=8.5)
    ax_roc.set_title('Random Forest ROC Curve', fontsize=9.5, fontweight='bold')
    ax_roc.legend(loc='lower right', fontsize=8)
    ax_roc.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plot_roc_b64 = fig_to_base64(fig_roc)

    return {
        'model': rf,
        'y_prob': y_prob,
        'metrics': {
            'acc': f"{acc:.4f}",
            'prec': f"{prec:.4f}",
            'rec': f"{rec:.4f}",
            'f1': f"{f1:.4f}",
            'auc': f"{auc_val:.4f}",
            'tn': int(cm[0, 0]),
            'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]),
            'tp': int(cm[1, 1])
        },
        'plots': {
            'rf_tree': plot_tree_b64,
            'rf_cm': plot_cm_b64,
            'rf_roc': plot_roc_b64
        }
    }