import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import roc_curve, roc_auc_score
from models.utils import fig_to_base64, calc_metrics

def run_cart(data_bundle, max_depth=3, cart_thresh=0.50, seed=42):
    X_tr = data_bundle['X_train']
    X_te = data_bundle['X_test']
    y_tr = data_bundle['y_train']
    y_te = data_bundle['y_test']

    cart = DecisionTreeClassifier(max_depth=int(max_depth), min_samples_leaf=20, random_state=seed)
    cart.fit(X_tr, y_tr)
    y_prob = cart.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= cart_thresh).astype(int)
    metrics = calc_metrics(y_te, y_pred, y_prob)

    # 1. Full Decision Tree Diagram
    fig_t, ax = plt.subplots(figsize=(10, 4.4))
    plot_tree(cart, feature_names=X_tr.columns.tolist(), class_names=['Legit', 'Abuse'],
              filled=True, rounded=True, precision=2, fontsize=8, impurity=False, proportion=True, ax=ax)
    ax.set_title(f"Pruned Classification Tree (Max Depth = {max_depth})", fontsize=10, fontweight='bold')
    tree_b64 = fig_to_base64(fig_t)

    # 2. CART Gini Feature Importance Plot
    fig_i, ax_i = plt.subplots(figsize=(6, 3.2))
    pd.Series(cart.feature_importances_, index=X_tr.columns).sort_values().tail(8).plot(
        kind='barh', color='#ff7f0e', edgecolor='black', ax=ax_i
    )
    ax_i.set_title("CART: Top Gini Feature Importances", fontsize=10, fontweight='bold')
    ax_i.set_xlabel("Gini Importance Score")
    ax_i.grid(axis='x', linestyle=':', alpha=0.6)
    cart_imp_b64 = fig_to_base64(fig_i)

    # 3. CART ROC Curve
    fig_r, ax_r = plt.subplots(figsize=(5.5, 3.2))
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    ax_r.plot(fpr, tpr, color='#ff7f0e', lw=2, label=f"AUC = {roc_auc_score(y_te, y_prob):.4f}")
    ax_r.plot([0, 1], [0, 1], 'k--', lw=1)
    ax_r.set_xlabel("False Positive Rate")
    ax_r.set_ylabel("True Positive Rate")
    ax_r.set_title("CART Decision Tree ROC Curve", fontsize=10, fontweight='bold')
    ax_r.legend(loc='lower right', fontsize=8)
    ax_r.grid(True, linestyle=':', alpha=0.6)
    roc_b64 = fig_to_base64(fig_r)

    return {
        'model': cart,
        'y_prob': y_prob,
        'metrics': metrics,
        'plots': {
            'cart_tree': tree_b64,
            'cart_imp': cart_imp_b64,
            'roc_cart': roc_b64
        }
    }