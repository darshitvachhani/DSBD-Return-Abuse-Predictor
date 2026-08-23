import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import roc_curve, roc_auc_score
from models.utils import fig_to_base64, calc_metrics

def run_bagging(data_bundle, n_trees=60, depth=8, bag_thresh=0.50, seed=42):
    X_tr = data_bundle['X_train']
    X_te = data_bundle['X_test']
    y_tr = data_bundle['y_train']
    y_te = data_bundle['y_test']

    bag = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=int(depth), min_samples_leaf=5, random_state=seed),
        n_estimators=int(n_trees), random_state=seed, n_jobs=-1
    )
    bag.fit(X_tr, y_tr)
    y_prob = bag.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= bag_thresh).astype(int)
    metrics = calc_metrics(y_te, y_pred, y_prob)

    # 1. Representative Bootstrap Tree (First Estimator Subtree)
    fig_t, ax_t = plt.subplots(figsize=(10, 6.4))
    plot_tree(bag.estimators_[0], feature_names=X_tr.columns.tolist(), class_names=['Legit', 'Abuse'],
              max_depth=3, filled=True, rounded=True, precision=2, fontsize=8, impurity=False, proportion=True, ax=ax_t)
    ax_t.set_title("Bagging: Representative Bootstrap Tree (Top Splits)", fontsize=10, fontweight='bold')
    tree_b64 = fig_to_base64(fig_t)

    # 2. Ensemble Mean Feature Importance
    bag_imp = np.mean([tree.feature_importances_ for tree in bag.estimators_], axis=0)
    fig_i, ax_i = plt.subplots(figsize=(6, 3.2))
    pd.Series(bag_imp, index=X_tr.columns).sort_values().tail(8).plot(
        kind='barh', color='#9467bd', edgecolor='black', ax=ax_i
    )
    ax_i.set_title("Bagging: Mean Feature Importance (All Trees)", fontsize=10, fontweight='bold')
    ax_i.set_xlabel("Mean Gini Importance")
    ax_i.grid(axis='x', linestyle=':', alpha=0.6)
    imp_b64 = fig_to_base64(fig_i)

    # 3. Bagging ROC Curve
    fig_r, ax_r = plt.subplots(figsize=(5.5, 3.2))
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    ax_r.plot(fpr, tpr, color='#9467bd', lw=2, label=f"AUC = {roc_auc_score(y_te, y_prob):.4f}")
    ax_r.plot([0, 1], [0, 1], 'k--', lw=1)
    ax_r.set_xlabel("False Positive Rate")
    ax_r.set_ylabel("True Positive Rate")
    ax_r.set_title("Bagging Classifier ROC Curve", fontsize=10, fontweight='bold')
    ax_r.legend(loc='lower right', fontsize=8)
    ax_r.grid(True, linestyle=':', alpha=0.6)
    roc_b64 = fig_to_base64(fig_r)

    return {
        'model': bag,
        'y_prob': y_prob,
        'metrics': metrics,
        'plots': {
            'bag_tree': tree_b64,
            'bag_imp': imp_b64,
            'roc_bag': roc_b64
        }
    }