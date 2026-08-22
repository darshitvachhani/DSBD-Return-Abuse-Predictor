import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import roc_curve, roc_auc_score
from models.utils import fig_to_base64, calc_metrics

def run_random_forest(data_bundle, n_trees=100, depth=8, rf_thresh=0.50, seed=42):
    X_tr = data_bundle['X_train']
    X_te = data_bundle['X_test']
    y_tr = data_bundle['y_train']
    y_te = data_bundle['y_test']

    # 1. Fit Random Forest Classifier
    rf = RandomForestClassifier(
        n_estimators=int(n_trees),
        max_depth=int(depth),
        min_samples_leaf=10,
        random_state=seed,
        n_jobs=-1
    )
    rf.fit(X_tr, y_tr)
    y_prob = rf.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= rf_thresh).astype(int)
    metrics = calc_metrics(y_te, y_pred, y_prob)

    # 2. Representative Tree Visualization (Depth 3 Truncated Subtree)
    fig_t, ax_t = plt.subplots(figsize=(10, 4.4))
    plot_tree(
        rf.estimators_[0],
        feature_names=X_tr.columns.tolist(),
        class_names=['Legit', 'Abuse'],
        max_depth=3,
        filled=True,
        rounded=True,
        precision=2,
        fontsize=8,
        impurity=False,
        proportion=True,
        ax=ax_t
    )
    ax_t.set_title("Random Forest: Representative Tree (Top Splits)", fontsize=10, fontweight='bold')
    tree_b64 = fig_to_base64(fig_t)

    # 3. Gini Feature Importance Chart (Primary Discussion Chart)
    fig_i, ax_i = plt.subplots(figsize=(6, 3.2))
    pd.Series(rf.feature_importances_, index=X_tr.columns).sort_values().tail(8).plot(
        kind='barh', color='#2ca02c', edgecolor='black', ax=ax_i
    )
    ax_i.set_title("Random Forest: Top 8 Predictors by Importance", fontsize=10, fontweight='bold')
    ax_i.set_xlabel("Mean Decrease in Impurity (Gini)")
    ax_i.grid(axis='x', linestyle=':', alpha=0.6)
    imp_b64 = fig_to_base64(fig_i)

    # 4. Random Forest ROC Curve
    fig_r, ax_r = plt.subplots(figsize=(5.5, 3.2))
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    ax_r.plot(fpr, tpr, color='#2ca02c', lw=2, label=f"AUC = {roc_auc_score(y_te, y_prob):.4f}")
    ax_r.plot([0, 1], [0, 1], 'k--', lw=1)
    ax_r.set_xlabel("False Positive Rate")
    ax_r.set_ylabel("True Positive Rate")
    ax_r.set_title("Random Forest ROC Curve", fontsize=10, fontweight='bold')
    ax_r.legend(loc='lower right', fontsize=8)
    ax_r.grid(True, linestyle=':', alpha=0.6)
    roc_b64 = fig_to_base64(fig_r)

    return {
        'model': rf,
        'y_prob': y_prob,
        'metrics': metrics,
        'plots': {
            'rf_tree': tree_b64,
            'rf_imp': imp_b64,
            'roc_rf': roc_b64
        }
    }