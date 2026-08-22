import os
import zipfile

# 1. Directory Tree
directories = ["models", "templates", "templates/sections"]
for d in directories:
  os.makedirs(d, exist_ok=True)

files = {}

# ==========================================
# 1. models/__init__.py & utils.py
# ==========================================
files["models/__init__.py"] = ""

files["models/utils.py"] = """import io
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
"""

# ==========================================
# 2. models/data_pipeline.py
# ==========================================
files["models/data_pipeline.py"] = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from models.utils import fig_to_base64

NUM_COLS = [
    'Order Value', 'Price', 'Days between delivery and return',
    'Customer liftime return rate', 'Discount in percentage',
    'Account age (in days)', 'Consecutive prior returns'
]
CAT_COLS = ['Payment method', 'Catergory Mix', 'Device signals']
TARGET_COL = 'is_abuse'
TEXT_COL = 'Free text return reason'
CUST_ID_COL = 'Customer ID'

def load_and_preprocess(csv_path="Return Abuse data filtered 1.csv"):
    df = pd.read_csv(csv_path)
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    for col in NUM_COLS:
        df[col] = df[col].fillna(df[col].median())
    for col in CAT_COLS:
        df[col] = df[col].fillna(df[col].mode()[0])
    df[TEXT_COL] = df[TEXT_COL].fillna("None")
    return df

def prepare_partition(df, test_size=0.20, seed=42):
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df[TARGET_COL]
    )

    tfidf = TfidfVectorizer(max_features=4, stop_words='english')
    X_tr_t = tfidf.fit_transform(train_df[TEXT_COL]).toarray()
    X_te_t = tfidf.transform(test_df[TEXT_COL]).toarray()
    tfidf_cols = [f"tfidf_{w}" for w in tfidf.get_feature_names_out()[:3]]

    X_tr_cat = pd.get_dummies(train_df[CAT_COLS], drop_first=True, dtype=int)
    X_te_cat = pd.get_dummies(test_df[CAT_COLS], drop_first=True, dtype=int).reindex(
        columns=X_tr_cat.columns, fill_value=0
    )

    X_train = pd.concat([
        train_df[NUM_COLS].reset_index(drop=True),
        X_tr_cat.reset_index(drop=True),
        pd.DataFrame(X_tr_t[:, :3], columns=tfidf_cols)
    ], axis=1)

    X_test = pd.concat([
        test_df[NUM_COLS].reset_index(drop=True),
        X_te_cat.reset_index(drop=True),
        pd.DataFrame(X_te_t[:, :3], columns=tfidf_cols)
    ], axis=1)

    y_train = train_df[TARGET_COL].values
    y_test = test_df[TARGET_COL].values

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.2))
    ax1.pie([len(train_df), len(test_df)], labels=['Train Set', 'Test Set'], autopct='%1.1f%%', colors=['#2b5c8f', '#d95f02'], startangle=140)
    ax1.set_title(f"Partitioning ({int(test_size*100)}% Test)", fontsize=10, fontweight='bold')
    ax2.pie([np.sum(y_train == 0), np.sum(y_train == 1)], labels=['Legitimate (0)', 'Abuse (1)'], autopct='%1.1f%%', colors=['#2ca02c', '#d62728'], startangle=140)
    ax2.set_title("Training Class Balance", fontsize=10, fontweight='bold')
    pie_b64 = fig_to_base64(fig)

    return {
        'train_df': train_df, 'test_df': test_df,
        'X_train': X_train, 'X_test': X_test,
        'X_train_scaled': X_train_scaled, 'X_test_scaled': X_test_scaled,
        'y_train': y_train, 'y_test': y_test,
        'scaler': scaler, 'tfidf': tfidf,
        'cat_cols': X_tr_cat.columns, 'tfidf_cols': tfidf_cols,
        'pie_plot': pie_b64
    }
"""

# ==========================================
# 3. models/logistic_regression.py
# ==========================================
files["models/logistic_regression.py"] = """import re
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from sklearn.metrics import roc_curve, roc_auc_score
from models.utils import fig_to_base64, calc_metrics

def run_logistic(data_bundle, lr_thresh=0.50, text_col='Free text return reason'):
    X_tr_sc = data_bundle['X_train_scaled']
    X_te_sc = data_bundle['X_test_scaled']
    y_tr = data_bundle['y_train']
    y_te = data_bundle['y_test']

    logit_model = sm.Logit(y_tr, sm.add_constant(X_tr_sc)).fit(disp=False)
    y_prob = logit_model.predict(sm.add_constant(X_te_sc))
    y_pred = (y_prob >= lr_thresh).astype(int)
    metrics = calc_metrics(y_te, y_pred, y_prob)

    all_text = " ".join(data_bundle['train_df'][text_col].dropna().astype(str).tolist())
    wc = WordCloud(width=750, height=350, background_color='white', colormap='magma', stopwords=STOPWORDS, max_words=60).generate(re.sub(r'[^\\w\\s]', ' ', all_text).lower())
    fig_wc, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    wc_b64 = fig_to_base64(fig_wc)

    var_name = 'Consecutive prior returns'
    fig_sig, ax = plt.subplots(figsize=(6.2, 3.4))
    x_pts = np.linspace(X_tr_sc[var_name].min(), X_tr_sc[var_name].max(), 200)
    x_df = pd.DataFrame(0.0, index=np.arange(len(x_pts)), columns=X_tr_sc.columns)
    x_df[var_name] = x_pts
    ax.plot(x_pts, logit_model.predict(sm.add_constant(x_df)), color='#d62728', lw=2.2, label='Estimated Logistic Curve')
    ax.axhline(y=lr_thresh, color='black', linestyle='--', lw=1.2, label=f'Cutoff ({lr_thresh:.2f})')
    ax.set_xlabel(f"{var_name} (Standardised)")
    ax.set_ylabel("Predicted P(Abuse)")
    ax.legend(loc='best', fontsize=8)
    sig_b64 = fig_to_base64(fig_sig)

    fig_roc, ax = plt.subplots(figsize=(5.5, 3.2))
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    ax.plot(fpr, tpr, color='#1f77b4', lw=2, label=f"AUC = {roc_auc_score(y_te, y_prob):.4f}")
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.legend(loc='lower right')
    roc_b64 = fig_to_base64(fig_roc)

    coef_df = pd.DataFrame({
        "Predictor": logit_model.params.index,
        "Coefficient": logit_model.params.values.round(4),
        "Std Error": logit_model.bse.values.round(4),
        "z-stat": logit_model.tvalues.values.round(4),
        "p-value": logit_model.pvalues.values.round(4),
        "Odds Ratio": np.exp(logit_model.params.values).round(4)
    })

    return {
        'model': logit_model, 'y_prob': y_prob, 'metrics': metrics,
        'coefs': coef_df.to_dict(orient='records'),
        'plots': {'wordcloud': wc_b64, 'sigmoid': sig_b64, 'roc_lr': roc_b64}
    }
"""

# ==========================================
# 4. models/cart_tree.py
# ==========================================
files["models/cart_tree.py"] = """import matplotlib.pyplot as plt
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

    fig_t, ax = plt.subplots(figsize=(10, 4.4))
    plot_tree(cart, feature_names=X_tr.columns.tolist(), class_names=['Legit', 'Abuse'],
              filled=True, rounded=True, precision=2, fontsize=8, impurity=False, proportion=True, ax=ax)
    tree_b64 = fig_to_base64(fig_t)

    fig_r, ax = plt.subplots(figsize=(5.5, 3.2))
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    ax.plot(fpr, tpr, color='#ff7f0e', lw=2, label=f"AUC = {roc_auc_score(y_te, y_prob):.4f}")
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.legend(loc='lower right')
    roc_b64 = fig_to_base64(fig_r)

    return {
        'model': cart, 'y_prob': y_prob, 'metrics': metrics,
        'plots': {'cart_tree': tree_b64, 'roc_cart': roc_b64}
    }
"""

# ==========================================
# 5. models/random_forest.py
# ==========================================
files["models/random_forest.py"] = """import pandas as pd
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

    rf = RandomForestClassifier(n_estimators=int(n_trees), max_depth=int(depth), min_samples_leaf=10, random_state=seed, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    y_prob = rf.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= rf_thresh).astype(int)
    metrics = calc_metrics(y_te, y_pred, y_prob)

    fig_t, ax = plt.subplots(figsize=(10, 4.4))
    plot_tree(rf.estimators_[0], feature_names=X_tr.columns.tolist(), class_names=['Legit', 'Abuse'],
              max_depth=3, filled=True, rounded=True, precision=2, fontsize=8, impurity=False, proportion=True, ax=ax)
    tree_b64 = fig_to_base64(fig_t)

    fig_i, ax = plt.subplots(figsize=(6, 3.2))
    pd.Series(rf.feature_importances_, index=X_tr.columns).sort_values().tail(8).plot(kind='barh', color='#2ca02c', ax=ax)
    ax.set_title("Random Forest: Top 8 Predictors", fontsize=10, fontweight='bold')
    imp_b64 = fig_to_base64(fig_i)

    fig_r, ax = plt.subplots(figsize=(5.5, 3.2))
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    ax.plot(fpr, tpr, color='#2ca02c', lw=2, label=f"AUC = {roc_auc_score(y_te, y_prob):.4f}")
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.legend(loc='lower right')
    roc_b64 = fig_to_base64(fig_r)

    return {
        'model': rf, 'y_prob': y_prob, 'metrics': metrics,
        'plots': {'rf_tree': tree_b64, 'rf_imp': imp_b64, 'roc_rf': roc_b64}
    }
"""

# ==========================================
# 6. models/bagging.py
# ==========================================
files["models/bagging.py"] = """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
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

    bag_imp = np.mean([tree.feature_importances_ for tree in bag.estimators_], axis=0)
    fig_i, ax = plt.subplots(figsize=(6, 3.2))
    pd.Series(bag_imp, index=X_tr.columns).sort_values().tail(8).plot(kind='barh', color='#9467bd', ax=ax)
    ax.set_title("Bagging: Mean Feature Importance", fontsize=10, fontweight='bold')
    imp_b64 = fig_to_base64(fig_i)

    fig_r, ax = plt.subplots(figsize=(5.5, 3.2))
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    ax.plot(fpr, tpr, color='#9467bd', lw=2, label=f"AUC = {roc_auc_score(y_te, y_prob):.4f}")
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.legend(loc='lower right')
    roc_b64 = fig_to_base64(fig_r)

    return {
        'model': bag, 'y_prob': y_prob, 'metrics': metrics,
        'plots': {'bag_imp': imp_b64, 'roc_bag': roc_b64}
    }
"""

# ==========================================
# 7. models/clustering.py
# ==========================================
files["models/clustering.py"] = """import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from models.utils import fig_to_base64

def run_clustering(df, num_cols, target_col='is_abuse', cust_id_col='Customer ID', n_clusters=3, seed=42):
    cust_df = df.groupby(cust_id_col).agg({**{c: 'mean' for c in num_cols}, target_col: 'mean'}).reset_index()
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(cust_df[num_cols])

    inertias = [KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X_sc).inertia_ for k in range(1, 8)]
    fig_e, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(range(1, 8), inertias, 'o-', color='#1f77b4', lw=2)
    ax.set_xlabel("Clusters (k)")
    ax.set_ylabel("Inertia")
    elbow_b64 = fig_to_base64(fig_e)

    km = KMeans(n_clusters=int(n_clusters), random_state=seed, n_init=10)
    cust_df['Cluster'] = "Cluster " + km.fit_predict(X_sc).astype(str)
    pca_c = PCA(n_components=2, random_state=seed).fit_transform(X_sc)
    cust_df['PCA1'], cust_df['PCA2'] = pca_c[:, 0], pca_c[:, 1]

    fig_p, ax = plt.subplots(figsize=(6.2, 3.6))
    sns.scatterplot(data=cust_df, x='PCA1', y='PCA2', hue='Cluster', palette='tab10', alpha=0.75, s=40, ax=ax)
    pca_b64 = fig_to_base64(fig_p)

    tbl = cust_df.groupby('Cluster').agg({
        cust_id_col: 'count',
        target_col: lambda x: f"{x.mean()*100:.2f}%",
        'Consecutive prior returns': 'mean',
        'Customer liftime return rate': 'mean',
        'Order Value': 'mean'
    }).reset_index()
    tbl.columns = ['Segment', 'Customer Count', 'Abuse Rate', 'Avg Prior Returns', 'Lifetime Return Rate', 'Avg Order Value (£)']

    return {
        'table': tbl.to_dict(orient='records'),
        'plots': {'elbow': elbow_b64, 'pca_cluster': pca_b64}
    }
"""

# ==========================================
# 8. models/cost_optimization.py
# ==========================================
files["models/cost_optimization.py"] = """import numpy as np
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
"""

# ==========================================
# 9. HTML Partials (templates/sections/)
# ==========================================
files["templates/sections/simulator.html"] = """<section id="sec-simulator" class="section-card nav-anchor border-primary">
    <div class="section-header bg-primary text-white d-flex justify-content-between align-items-center">
        <span><i class="bi bi-cpu-fill me-2 text-warning"></i> Interactive Return Abuse Risk Evaluator</span>
        <span class="badge bg-light text-primary">Live Predictor</span>
    </div>
    <div class="p-4">
        <p class="text-muted small mb-3">Manually enter transaction, behavioral, and reason values below to instantly evaluate return risk via the live ensemble engine.</p>
        <div class="row g-3">
            <div class="col-md-3"><label class="form-label small fw-bold">Order Value (£)</label><input type="number" id="sim_order_val" class="form-control form-control-sm" value="85.0" step="1"></div>
            <div class="col-md-3"><label class="form-label small fw-bold">Item Price (£)</label><input type="number" id="sim_price" class="form-control form-control-sm" value="45.0" step="1"></div>
            <div class="col-md-3"><label class="form-label small fw-bold">Days to Return</label><input type="number" id="sim_days_ret" class="form-control form-control-sm" value="21" min="1" max="30"></div>
            <div class="col-md-3"><label class="form-label small fw-bold">Prior Returns</label><input type="number" id="sim_prior_rets" class="form-control form-control-sm" value="4" min="0" max="50"></div>
            <div class="col-md-3"><label class="form-label small fw-bold">Lifetime Rate (0-1)</label><input type="number" id="sim_ret_rate" class="form-control form-control-sm" value="0.45" step="0.05" min="0" max="1"></div>
            <div class="col-md-3"><label class="form-label small fw-bold">Account Age (Days)</label><input type="number" id="sim_acc_age" class="form-control form-control-sm" value="120"></div>
            <div class="col-md-3"><label class="form-label small fw-bold">Discount (%)</label><input type="number" id="sim_discount" class="form-control form-control-sm" value="5" min="0" max="50"></div>
            <div class="col-md-3"><label class="form-label small fw-bold">Payment Method</label><select id="sim_pay_method" class="form-select form-select-sm"><option>Credit Card</option><option>Debit Card</option><option>PayPal</option><option>Bank Transfer</option></select></div>
            <div class="col-md-6"><label class="form-label small fw-bold">Free Text Return Reason</label><input type="text" id="sim_reason_text" class="form-control form-control-sm" value="Item did not suit me"></div>
            <div class="col-md-6 d-flex align-items-end"><button class="btn btn-primary btn-sm w-100 fw-bold" onclick="evaluateSingleReturn()"><i class="bi bi-shield-check me-1"></i> Evaluate Return Abuse Probability</button></div>
        </div>
        <div id="sim_result_box" class="alert alert-warning mt-3 p-3 d-flex justify-content-between align-items-center">
            <div>
                <h6 class="fw-bold mb-1" id="sim_pred_title">Risk Score: 78.4% — Abusive Return (High Risk)</h6>
                <span class="small" id="sim_pred_action"><strong>Action:</strong> Manual Warehouse Inspection Required Before Refund Authorization</span>
            </div>
            <span class="badge bg-danger fs-6" id="sim_badge">High Risk</span>
        </div>
    </div>
</section>
"""

files["templates/sections/business_context.html"] = """<section id="sec-intro" class="section-card nav-anchor">
    <div class="section-header"><i class="bi bi-info-circle text-primary me-2"></i> 1. Business Problem & Strategic Framework</div>
    <div class="p-4">
        <div class="row g-4">
            <div class="col-md-6">
                <h6 class="fw-bold text-dark">The Challenge: Abuse vs. Churn Risk</h6>
                <p class="small text-muted">Returns generate massive operational friction. Blanket return friction alienates high-value legitimate customers. Machine learning creates dynamic, risk-tiered policy interventions.</p>
            </div>
            <div class="col-md-6">
                <div class="p-3 bg-light rounded border">
                    <h6 class="fw-bold text-dark mb-2">Core Analytical Objectives</h6>
                    <ul class="small text-muted mb-0 ps-3">
                        <li>Supervised Classification (Logistic, CART, Random Forest, Bagging).</li>
                        <li>Unstructured Text Feature Extraction via TF-IDF Vectorization.</li>
                        <li>Customer Segmentation via K-Means Clustering and 2D PCA.</li>
                        <li>Threshold Calibration under Asymmetric Business Misclassification Costs.</li>
                    </ul>
                </div>
            </div>
        </div>
    </div>
</section>
"""

files["templates/sections/data_split.html"] = """<section id="sec-split" class="section-card nav-anchor">
    <div class="section-header"><i class="bi bi-pie-chart text-primary me-2"></i> 2. Data Understanding & Partitioning</div>
    <div class="p-3">
        <div class="row g-3">
            <div class="col-lg-6">
                <div class="code-box">
<pre><code class="language-python"># Leakage-Free Train-Test Partitioning Pipeline
train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df['is_abuse'])
tfidf = TfidfVectorizer(max_features=4, stop_words='english')
X_tr_t = tfidf.fit_transform(train_df['Free text return reason'])
X_te_t = tfidf.transform(test_df['Free text return reason'])
</code></pre>
                </div>
            </div>
            <div class="col-lg-6">
                <div class="output-box">
                    <div class="control-panel">
                        <label class="form-label small fw-bold d-flex justify-content-between mb-1">
                            <span>Test Partition Size:</span> <span id="lbl_test_size" class="badge bg-primary">20%</span>
                        </label>
                        <input type="range" class="form-range" id="input_test_size" min="0.10" max="0.40" step="0.05" value="0.20" oninput="document.getElementById('lbl_test_size').innerText = Math.round(this.value*100)+'%'; triggerUpdate();">
                    </div>
                    <div class="text-center"><img id="img_split_pie" src="data:image/png;base64,{{ data.plots.split_pie }}" class="img-fluid rounded" alt="Pie Chart"></div>
                </div>
            </div>
        </div>
    </div>
</section>
"""

files["templates/sections/logistic.html"] = """<section id="sec-lr" class="section-card nav-anchor">
    <div class="section-header"><i class="bi bi-activity text-primary me-2"></i> 3. Text Analytics & Logistic Regression</div>
    <div class="p-3">
        <div class="row g-3">
            <div class="col-lg-6">
                <div class="code-box mb-2">
<pre><code class="language-python"># Multivariate Logistic Regression
logit_model = sm.Logit(y_train, sm.add_constant(X_train_scaled)).fit(disp=False)
y_prob_lr = logit_model.predict(sm.add_constant(X_test_scaled))
y_pred_lr = (y_prob_lr >= lr_thresh).astype(int)
</code></pre>
                </div>
                <div class="table-responsive output-box" style="max-height: 200px; overflow-y: auto;">
                    <table class="table table-sm table-striped" style="font-size: 0.72rem;">
                        <thead><tr><th>Predictor</th><th>Estimate</th><th>z-stat</th><th>p-value</th><th>Odds Ratio</th></tr></thead>
                        <tbody id="tbl_lr_coefs">
                            {% for r in data.lr_coefs %}
                            <tr><td>{{ r.Predictor }}</td><td>{{ r.Coefficient }}</td><td>{{ r['z-stat'] }}</td><td>{{ r['p-value'] }}</td><td>{{ r['Odds Ratio'] }}</td></tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
            <div class="col-lg-6">
                <div class="output-box">
                    <div class="control-panel">
                        <label class="form-label small fw-bold d-flex justify-content-between mb-1">
                            <span>Logistic Decision Cutoff:</span> <span id="lbl_lr_thresh" class="badge bg-primary">0.50</span>
                        </label>
                        <input type="range" class="form-range" id="input_lr_thresh" min="0.10" max="0.90" step="0.05" value="0.50" oninput="document.getElementById('lbl_lr_thresh').innerText = parseFloat(this.value).toFixed(2); triggerUpdate();">
                    </div>
                    <div class="d-flex justify-content-between mb-2">
                        <span class="badge bg-primary metric-badge">Accuracy: <span id="m_lr_acc">{{ data.metrics.lr.acc }}</span></span>
                        <span class="badge bg-success metric-badge">Precision: <span id="m_lr_prec">{{ data.metrics.lr.prec }}</span></span>
                        <span class="badge bg-warning text-dark metric-badge">Recall: <span id="m_lr_rec">{{ data.metrics.lr.rec }}</span></span>
                        <span class="badge bg-info text-dark metric-badge">AUC: <span id="m_lr_auc">{{ data.metrics.lr.auc }}</span></span>
                    </div>
                    <div class="text-center">
                        <img id="img_sigmoid" src="data:image/png;base64,{{ data.plots.sigmoid }}" class="img-fluid rounded mb-2" alt="Sigmoid">
                        <img id="img_wordcloud" src="data:image/png;base64,{{ data.plots.wordcloud }}" class="img-fluid rounded" alt="Wordcloud">
                    </div>
                </div>
            </div>
        </div>
    </div>
</section>
"""

files["templates/sections/cart.html"] = """<section id="sec-cart" class="section-card nav-anchor">
    <div class="section-header"><i class="bi bi-diagram-3 text-primary me-2"></i> 4. Classification Tree (CART)</div>
    <div class="p-3">
        <div class="row g-3">
            <div class="col-lg-6">
                <div class="code-box">
<pre><code class="language-python"># Pruned Decision Tree (CART)
cart = DecisionTreeClassifier(max_depth=cart_depth, min_samples_leaf=20, random_state=42)
cart.fit(X_train, y_train)
y_prob_cart = cart.predict_proba(X_test)[:, 1]
y_pred_cart = (y_prob_cart >= cart_thresh).astype(int)
</code></pre>
                </div>
                <div class="text-center mt-2"><img id="img_roc_cart" src="data:image/png;base64,{{ data.plots.roc_cart }}" class="img-fluid rounded output-box" alt="CART ROC"></div>
            </div>
            <div class="col-lg-6">
                <div class="output-box">
                    <div class="control-panel">
                        <div class="row g-2">
                            <div class="col-6">
                                <label class="form-label small fw-bold d-flex justify-content-between mb-1"><span>Max Depth:</span> <span id="lbl_cart_depth" class="badge bg-primary">3</span></label>
                                <input type="range" class="form-range" id="input_cart_depth" min="2" max="6" step="1" value="3" oninput="document.getElementById('lbl_cart_depth').innerText = this.value; triggerUpdate();">
                            </div>
                            <div class="col-6">
                                <label class="form-label small fw-bold d-flex justify-content-between mb-1"><span>Cutoff:</span> <span id="lbl_cart_thresh" class="badge bg-primary">0.50</span></label>
                                <input type="range" class="form-range" id="input_cart_thresh" min="0.10" max="0.90" step="0.05" value="0.50" oninput="document.getElementById('lbl_cart_thresh').innerText = parseFloat(this.value).toFixed(2); triggerUpdate();">
                            </div>
                        </div>
                    </div>
                    <div class="d-flex justify-content-between mb-2">
                        <span class="badge bg-primary metric-badge">Acc: <span id="m_cart_acc">{{ data.metrics.cart.acc }}</span></span>
                        <span class="badge bg-success metric-badge">Prec: <span id="m_cart_prec">{{ data.metrics.cart.prec }}</span></span>
                        <span class="badge bg-warning text-dark metric-badge">Rec: <span id="m_cart_rec">{{ data.metrics.cart.rec }}</span></span>
                        <span class="badge bg-info text-dark metric-badge">AUC: <span id="m_cart_auc">{{ data.metrics.cart.auc }}</span></span>
                    </div>
                    <div class="text-center"><img id="img_cart_tree" src="data:image/png;base64,{{ data.plots.cart_tree }}" class="img-fluid rounded" alt="CART Tree"></div>
                </div>
            </div>
        </div>
    </div>
</section>
"""

files["templates/sections/random_forest.html"] = """<section id="sec-rf" class="section-card nav-anchor">
    <div class="section-header"><i class="bi bi-tree text-primary me-2"></i> 5. Random Forest Ensemble</div>
    <div class="p-3">
        <div class="row g-3">
            <div class="col-lg-6">
                <div class="code-box">
<pre><code class="language-python"># Random Forest Ensemble
rf = RandomForestClassifier(n_estimators=rf_trees, max_depth=8, min_samples_leaf=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_prob_rf = rf.predict_proba(X_test)[:, 1]
y_pred_rf = (y_prob_rf >= rf_thresh).astype(int)
</code></pre>
                </div>
                <div class="text-center mt-2"><img id="img_rf_imp" src="data:image/png;base64,{{ data.plots.rf_imp }}" class="img-fluid rounded output-box" alt="RF Importance"></div>
            </div>
            <div class="col-lg-6">
                <div class="output-box">
                    <div class="control-panel">
                        <div class="row g-2">
                            <div class="col-6">
                                <label class="form-label small fw-bold d-flex justify-content-between mb-1"><span>Trees:</span> <span id="lbl_rf_trees" class="badge bg-primary">100</span></label>
                                <input type="range" class="form-range" id="input_rf_trees" min="25" max="200" step="25" value="100" oninput="document.getElementById('lbl_rf_trees').innerText = this.value; triggerUpdate();">
                            </div>
                            <div class="col-6">
                                <label class="form-label small fw-bold d-flex justify-content-between mb-1"><span>Cutoff:</span> <span id="lbl_rf_thresh" class="badge bg-primary">0.50</span></label>
                                <input type="range" class="form-range" id="input_rf_thresh" min="0.10" max="0.90" step="0.05" value="0.50" oninput="document.getElementById('lbl_rf_thresh').innerText = parseFloat(this.value).toFixed(2); triggerUpdate();">
                            </div>
                        </div>
                    </div>
                    <div class="d-flex justify-content-between mb-2">
                        <span class="badge bg-primary metric-badge">Acc: <span id="m_rf_acc">{{ data.metrics.rf.acc }}</span></span>
                        <span class="badge bg-success metric-badge">Prec: <span id="m_rf_prec">{{ data.metrics.rf.prec }}</span></span>
                        <span class="badge bg-warning text-dark metric-badge">Rec: <span id="m_rf_rec">{{ data.metrics.rf.rec }}</span></span>
                        <span class="badge bg-info text-dark metric-badge">AUC: <span id="m_rf_auc">{{ data.metrics.rf.auc }}</span></span>
                    </div>
                    <div class="text-center"><img id="img_rf_tree" src="data:image/png;base64,{{ data.plots.rf_tree }}" class="img-fluid rounded" alt="RF Tree"></div>
                </div>
            </div>
        </div>
    </div>
</section>
"""

files["templates/sections/bagging.html"] = """<section id="sec-bag" class="section-card nav-anchor">
    <div class="section-header"><i class="bi bi-collection text-primary me-2"></i> 6. Bagging Classifier</div>
    <div class="p-3">
        <div class="row g-3">
            <div class="col-lg-6">
                <div class="code-box">
<pre><code class="language-python"># Bootstrap Aggregated Classifier
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=8, min_samples_leaf=5),
    n_estimators=bag_trees, random_state=42, n_jobs=-1
)
bagging.fit(X_train, y_train)
y_prob_bag = bagging.predict_proba(X_test)[:, 1]
y_pred_bag = (y_prob_bag >= bag_thresh).astype(int)
</code></pre>
                </div>
                <div class="text-center mt-2"><img id="img_roc_bag" src="data:image/png;base64,{{ data.plots.roc_bag }}" class="img-fluid rounded output-box" alt="Bagging ROC"></div>
            </div>
            <div class="col-lg-6">
                <div class="output-box">
                    <div class="control-panel">
                        <div class="row g-2">
                            <div class="col-6">
                                <label class="form-label small fw-bold d-flex justify-content-between mb-1"><span>Estimators:</span> <span id="lbl_bag_trees" class="badge bg-primary">60</span></label>
                                <input type="range" class="form-range" id="input_bag_trees" min="25" max="150" step="25" value="60" oninput="document.getElementById('lbl_bag_trees').innerText = this.value; triggerUpdate();">
                            </div>
                            <div class="col-6">
                                <label class="form-label small fw-bold d-flex justify-content-between mb-1"><span>Cutoff:</span> <span id="lbl_bag_thresh" class="badge bg-primary">0.50</span></label>
                                <input type="range" class="form-range" id="input_bag_thresh" min="0.10" max="0.90" step="0.05" value="0.50" oninput="document.getElementById('lbl_bag_thresh').innerText = parseFloat(this.value).toFixed(2); triggerUpdate();">
                            </div>
                        </div>
                    </div>
                    <div class="d-flex justify-content-between mb-2">
                        <span class="badge bg-primary metric-badge">Acc: <span id="m_bag_acc">{{ data.metrics.bag.acc }}</span></span>
                        <span class="badge bg-success metric-badge">Prec: <span id="m_bag_prec">{{ data.metrics.bag.prec }}</span></span>
                        <span class="badge bg-warning text-dark metric-badge">Rec: <span id="m_bag_rec">{{ data.metrics.bag.rec }}</span></span>
                        <span class="badge bg-info text-dark metric-badge">AUC: <span id="m_bag_auc">{{ data.metrics.bag.auc }}</span></span>
                    </div>
                    <div class="text-center"><img id="img_bag_imp" src="data:image/png;base64,{{ data.plots.bag_imp }}" class="img-fluid rounded" alt="Bagging Importance"></div>
                </div>
            </div>
        </div>
    </div>
</section>
"""

files["templates/sections/benchmark.html"] = """<section id="sec-comp" class="section-card nav-anchor">
    <div class="section-header"><i class="bi bi-table text-primary me-2"></i> 7. Master Model Benchmark & Combined ROC</div>
    <div class="p-3">
        <div class="row g-3">
            <div class="col-lg-6">
                <div class="output-box h-100">
                    <h6 class="fw-bold mb-2">Comparison Table (Test Set)</h6>
                    <div class="table-responsive">
                        <table class="table table-sm table-bordered align-middle" style="font-size: 0.78rem;">
                            <thead class="table-light"><tr><th>Model</th><th>Acc</th><th>Prec</th><th>Rec</th><th>F1</th><th>AUC</th><th>FP / FN</th></tr></thead>
                            <tbody>
                                <tr><td>Logistic Reg.</td><td id="t_lr_acc">{{ data.metrics.lr.acc }}</td><td id="t_lr_prec">{{ data.metrics.lr.prec }}</td><td id="t_lr_rec">{{ data.metrics.lr.rec }}</td><td id="t_lr_f1">{{ data.metrics.lr.f1 }}</td><td id="t_lr_auc">{{ data.metrics.lr.auc }}</td><td><span class="badge bg-secondary" id="t_lr_fp">{{ data.metrics.lr.fp }}</span> / <span class="badge bg-danger" id="t_lr_fn">{{ data.metrics.lr.fn }}</span></td></tr>
                                <tr><td>CART Tree</td><td id="t_cart_acc">{{ data.metrics.cart.acc }}</td><td id="t_cart_prec">{{ data.metrics.cart.prec }}</td><td id="t_cart_rec">{{ data.metrics.cart.rec }}</td><td id="t_cart_f1">{{ data.metrics.cart.f1 }}</td><td id="t_cart_auc">{{ data.metrics.cart.auc }}</td><td><span class="badge bg-secondary" id="t_cart_fp">{{ data.metrics.cart.fp }}</span> / <span class="badge bg-danger" id="t_cart_fn">{{ data.metrics.cart.fn }}</span></td></tr>
                                <tr><td>Random Forest</td><td id="t_rf_acc">{{ data.metrics.rf.acc }}</td><td id="t_rf_prec">{{ data.metrics.rf.prec }}</td><td id="t_rf_rec">{{ data.metrics.rf.rec }}</td><td id="t_rf_f1">{{ data.metrics.rf.f1 }}</td><td id="t_rf_auc">{{ data.metrics.rf.auc }}</td><td><span class="badge bg-secondary" id="t_rf_fp">{{ data.metrics.rf.fp }}</span> / <span class="badge bg-danger" id="t_rf_fn">{{ data.metrics.rf.fn }}</span></td></tr>
                                <tr class="table-primary fw-bold"><td>Bagging</td><td id="t_bag_acc">{{ data.metrics.bag.acc }}</td><td id="t_bag_prec">{{ data.metrics.bag.prec }}</td><td id="t_bag_rec">{{ data.metrics.bag.rec }}</td><td id="t_bag_f1">{{ data.metrics.bag.f1 }}</td><td id="t_bag_auc">{{ data.metrics.bag.auc }}</td><td><span class="badge bg-secondary" id="t_bag_fp">{{ data.metrics.bag.fp }}</span> / <span class="badge bg-danger" id="t_bag_fn">{{ data.metrics.bag.fn }}</span></td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
            <div class="col-lg-6">
                <div class="output-box text-center h-100"><img id="img_combined_roc" src="data:image/png;base64,{{ data.plots.combined_roc }}" class="img-fluid rounded" alt="Combined ROC"></div>
            </div>
        </div>
    </div>
</section>
"""

files["templates/sections/clustering.html"] = """<section id="sec-clust" class="section-card nav-anchor">
    <div class="section-header"><i class="bi bi-people text-primary me-2"></i> 8. Customer Behavioural Clustering</div>
    <div class="p-3">
        <div class="row g-3">
            <div class="col-lg-6">
                <div class="output-box">
                    <div class="control-panel">
                        <label class="form-label small fw-bold d-flex justify-content-between mb-1"><span>Clusters (k):</span> <span id="lbl_n_clusters" class="badge bg-primary">3</span></label>
                        <input type="range" class="form-range" id="input_n_clusters" min="2" max="5" step="1" value="3" oninput="document.getElementById('lbl_n_clusters').innerText = this.value; triggerUpdate();">
                    </div>
                    <div class="text-center"><img id="img_elbow" src="data:image/png;base64,{{ data.plots.elbow }}" class="img-fluid rounded mb-2" alt="Elbow"></div>
                </div>
            </div>
            <div class="col-lg-6">
                <div class="output-box">
                    <div class="text-center mb-2"><img id="img_pca_cluster" src="data:image/png;base64,{{ data.plots.pca_cluster }}" class="img-fluid rounded" alt="PCA"></div>
                    <div class="table-responsive">
                        <table class="table table-sm table-striped" style="font-size: 0.72rem;">
                            <thead><tr><th>Segment</th><th>Count</th><th>Abuse %</th><th>Prior Returns</th><th>Return Rate</th></tr></thead>
                            <tbody id="tbl_clusters">
                                {% for c in data.clusters %}
                                <tr><td>{{ c.Segment }}</td><td>{{ c['Customer Count'] }}</td><td><strong>{{ c['Abuse Rate'] }}</strong></td><td>{{ "%.2f"|format(c['Avg Prior Returns']) }}</td><td>{{ "%.2f"|format(c['Lifetime Return Rate']) }}</td></tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>
</section>
"""

files["templates/sections/cost.html"] = """<section id="sec-cost" class="section-card nav-anchor">
    <div class="section-header"><i class="bi bi-cash-stack text-primary me-2"></i> 9. Cost-Sensitive Threshold Optimisation</div>
    <div class="p-3">
        <div class="row g-3">
            <div class="col-lg-6">
                <div class="output-box">
                    <div class="control-panel">
                        <div class="row g-2 mb-2">
                            <div class="col-6">
                                <label class="form-label small fw-bold d-flex justify-content-between mb-1"><span>Cost FP:</span> <span id="lbl_c_fp" class="badge bg-primary">£50</span></label>
                                <input type="range" class="form-range" id="input_c_fp" min="10" max="150" step="10" value="50" oninput="document.getElementById('lbl_c_fp').innerText = '£'+this.value; triggerUpdate();">
                            </div>
                            <div class="col-6">
                                <label class="form-label small fw-bold d-flex justify-content-between mb-1"><span>Cost FN:</span> <span id="lbl_c_fn" class="badge bg-primary">£25</span></label>
                                <input type="range" class="form-range" id="input_c_fn" min="10" max="100" step="5" value="25" oninput="document.getElementById('lbl_c_fn').innerText = '£'+this.value; triggerUpdate();">
                            </div>
                        </div>
                    </div>
                    <div class="alert alert-warning py-2 mb-0" style="font-size: 0.8rem;">
                        <strong>Optimal Cutoff ($t^*$):</strong> <span id="opt_thresh_val" class="fw-bold">{{ data.opt_thresh }}</span> | <strong>Min Expected Cost:</strong> <span id="min_cost_val" class="fw-bold">{{ data.min_cost }}</span>
                    </div>
                </div>
            </div>
            <div class="col-lg-6">
                <div class="output-box text-center"><img id="img_cost_curve" src="data:image/png;base64,{{ data.plots.cost_curve }}" class="img-fluid rounded" alt="Cost Curve"></div>
            </div>
        </div>
    </div>
</section>
"""

# ==========================================
# 10. templates/index.html (Master Shell)
# ==========================================
files["templates/index.html"] = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Return Abuse Analytics & Machine Learning Platform</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.0/font/bootstrap-icons.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/styles/atom-one-dark.min.css">
    <style>
        body { background-color: #f1f5f9; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: #1e293b; }
        .sidebar { background: #0f172a; min-height: 100vh; color: #fff; z-index: 1000; width: 240px; }
        .sidebar .nav-link { color: #94a3b8; font-size: 0.84rem; font-weight: 500; border-radius: 6px; margin-bottom: 4px; padding: 7px 10px; }
        .sidebar .nav-link:hover, .sidebar .nav-link.active { color: #fff; background: #334155; }
        .main-container { margin-left: 240px; padding: 25px 35px; }
        .section-card { background: #ffffff; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.06); margin-bottom: 35px; border: 1px solid #e2e8f0; overflow: hidden; }
        .section-header { background: #f8fafc; padding: 12px 20px; border-bottom: 1px solid #e2e8f0; font-weight: 700; color: #0f172a; }
        .code-box { background: #282c34; border-radius: 8px; padding: 12px; font-size: 0.76rem; max-height: 380px; overflow-y: auto; }
        .output-box { background: #ffffff; border-radius: 8px; border: 1px solid #e2e8f0; padding: 14px; }
        .control-panel { background: #eff6ff; border: 2px solid #3b82f6; border-radius: 8px; padding: 14px; margin-bottom: 14px; }
        .metric-badge { font-size: 0.8rem; font-weight: 600; padding: 6px 9px; border-radius: 6px; }
        .nav-anchor { scroll-margin-top: 20px; }
    </style>
</head>
<body>
<div class="container-fluid p-0">
    <nav class="sidebar py-3 px-3 position-fixed">
        <div class="px-2 mb-3">
            <h6 class="text-white fw-bold mb-0"><i class="bi bi-shield-check text-primary me-1"></i> FraudGuard AI</h6>
            <small class="text-muted" style="font-size: 0.72rem;">Return Abuse Analytics</small>
        </div>
        <ul class="nav flex-column">
            <li class="nav-item"><a class="nav-link active" href="#sec-simulator"><i class="bi bi-cpu-fill text-warning me-2"></i> Live Return Tester</a></li>
            <li class="nav-item"><a class="nav-link" href="#sec-intro"><i class="bi bi-info-circle me-2"></i> 1. Business Context</a></li>
            <li class="nav-item"><a class="nav-link" href="#sec-split"><i class="bi bi-pie-chart me-2"></i> 2. Data Partitioning</a></li>
            <li class="nav-item"><a class="nav-link" href="#sec-lr"><i class="bi bi-activity me-2"></i> 3. Logistic Regression</a></li>
            <li class="nav-item"><a class="nav-link" href="#sec-cart"><i class="bi bi-diagram-3 me-2"></i> 4. CART Decision Tree</a></li>
            <li class="nav-item"><a class="nav-link" href="#sec-rf"><i class="bi bi-tree me-2"></i> 5. Random Forest</a></li>
            <li class="nav-item"><a class="nav-link" href="#sec-bag"><i class="bi bi-collection me-2"></i> 6. Bagging Classifier</a></li>
            <li class="nav-item"><a class="nav-link" href="#sec-comp"><i class="bi bi-table me-2"></i> 7. Model Benchmark</a></li>
            <li class="nav-item"><a class="nav-link" href="#sec-clust"><i class="bi bi-people me-2"></i> 8. Customer Segments</a></li>
            <li class="nav-item"><a class="nav-link" href="#sec-cost"><i class="bi bi-cash-stack me-2"></i> 9. Cost Optimization</a></li>
        </ul>
    </nav>
    <main class="main-container">
        <div class="d-flex justify-content-between align-items-center pb-3 mb-4 border-bottom">
            <div>
                <h3 class="fw-bold mb-1">Return Abuse Detection & Behavioral Analytics</h3>
                <p class="text-muted small mb-0">Modular Localhost Workbench (Jinja2 Includes + Isolated Python Modules)</p>
            </div>
            <button class="btn btn-primary btn-sm fw-bold px-3" onclick="triggerUpdate()"><i class="bi bi-arrow-repeat me-1"></i> Sync & Recompute Models</button>
        </div>

        {% include "sections/simulator.html" %}
        {% include "sections/business_context.html" %}
        {% include "sections/data_split.html" %}
        {% include "sections/logistic.html" %}
        {% include "sections/cart.html" %}
        {% include "sections/random_forest.html" %}
        {% include "sections/bagging.html" %}
        {% include "sections/benchmark.html" %}
        {% include "sections/clustering.html" %}
        {% include "sections/cost.html" %}
    </main>
</div>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/highlight.min.js"></script>
<script>
    hljs.highlightAll();

    function evaluateSingleReturn() {
        const payload = {
            order_val: parseFloat(document.getElementById('sim_order_val').value),
            price: parseFloat(document.getElementById('sim_price').value),
            days_ret: parseFloat(document.getElementById('sim_days_ret').value),
            prior_rets: parseFloat(document.getElementById('sim_prior_rets').value),
            ret_rate: parseFloat(document.getElementById('sim_ret_rate').value),
            acc_age: parseFloat(document.getElementById('sim_acc_age').value),
            discount: parseFloat(document.getElementById('sim_discount').value),
            pay_method: document.getElementById('sim_pay_method').value,
            reason_text: document.getElementById('sim_reason_text').value
        };

        fetch('/api/predict_single', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
        .then(res => res.json())
        .then(data => {
            const resBox = document.getElementById('sim_result_box');
            resBox.className = `alert alert-${data.badge} mt-3 p-3 d-flex justify-content-between align-items-center`;
            document.getElementById('sim_pred_title').innerText = `Risk Score: ${data.prob} — ${data.pred_class}`;
            document.getElementById('sim_pred_action').innerHTML = `<strong>Action:</strong> ${data.action}`;
            document.getElementById('sim_badge').className = `badge bg-${data.badge} fs-6`;
            document.getElementById('sim_badge').innerText = data.prob_raw > 0.65 ? 'High Risk' : (data.prob_raw > 0.35 ? 'Medium Risk' : 'Low Risk');
        })
        .catch(err => console.error("Error evaluating return:", err));
    }

    function triggerUpdate() {
        const payload = {
            test_size: parseFloat(document.getElementById('input_test_size').value),
            lr_thresh: parseFloat(document.getElementById('input_lr_thresh').value),
            cart_depth: parseInt(document.getElementById('input_cart_depth').value),
            cart_thresh: parseFloat(document.getElementById('input_cart_thresh').value),
            rf_trees: parseInt(document.getElementById('input_rf_trees').value),
            rf_depth: 8,
            rf_thresh: parseFloat(document.getElementById('input_rf_thresh').value),
            bag_trees: parseInt(document.getElementById('input_bag_trees').value),
            bag_thresh: parseFloat(document.getElementById('input_bag_thresh').value),
            n_clusters: parseInt(document.getElementById('input_n_clusters').value),
            c_fp: parseFloat(document.getElementById('input_c_fp').value),
            c_fn: parseFloat(document.getElementById('input_c_fn').value)
        };

        fetch('/api/update', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
        .then(res => res.json())
        .then(data => {
            for (let [key, b64] of Object.entries(data.plots)) {
                let elem = document.getElementById('img_' + key);
                if (elem) elem.src = 'data:image/png;base64,' + b64;
            }

            for (let [m_key, m_val] of Object.entries(data.metrics)) {
                ['acc', 'prec', 'rec', 'auc'].forEach(metric => {
                    let badge = document.getElementById(`m_${m_key}_${metric}`);
                    if (badge) badge.innerText = m_val[metric];
                });
                ['acc', 'prec', 'rec', 'f1', 'auc', 'fp', 'fn'].forEach(metric => {
                    let cell = document.getElementById(`t_${m_key}_${metric}`);
                    if (cell) cell.innerText = m_val[metric];
                });
            }

            document.getElementById('opt_thresh_val').innerText = data.opt_thresh;
            document.getElementById('min_cost_val').innerText = data.min_cost;

            let tbodyLr = document.getElementById('tbl_lr_coefs');
            if (tbodyLr && data.lr_coefs) {
                tbodyLr.innerHTML = '';
                data.lr_coefs.forEach(r => {
                    tbodyLr.innerHTML += `<tr><td>${r.Predictor}</td><td>${r.Coefficient}</td><td>${r['z-stat']}</td><td>${r['p-value']}</td><td>${r['Odds Ratio']}</td></tr>`;
                });
            }

            let tbodyClust = document.getElementById('tbl_clusters');
            if (tbodyClust && data.clusters) {
                tbodyClust.innerHTML = '';
                data.clusters.forEach(c => {
                    tbodyClust.innerHTML += `<tr><td>${c.Segment}</td><td>${c['Customer Count']}</td><td><strong>${c['Abuse Rate']}</strong></td><td>${parseFloat(c['Avg Prior Returns']).toFixed(2)}</td><td>${parseFloat(c['Lifetime Return Rate']).toFixed(2)}</td></tr>`;
                });
            }
        })
        .catch(err => console.error("Error updating dashboard:", err));
    }
</script>
</body>
</html>
"""

# ==========================================
# 11. Main app.py & requirements.txt
# ==========================================
files["requirements.txt"] = """flask
pandas
numpy
matplotlib
seaborn
scikit-learn
statsmodels
wordcloud
gunicorn
"""

files["app.py"] = """import warnings
import matplotlib
from flask import Flask, render_template, request, jsonify
import pandas as pd

matplotlib.use('Agg')
warnings.filterwarnings('ignore')

from models.data_pipeline import load_and_preprocess, prepare_partition, NUM_COLS
from models.logistic_regression import run_logistic
from models.cart_tree import run_cart
from models.random_forest import run_random_forest
from models.bagging import run_bagging
from models.clustering import run_clustering
from models.cost_optimization import run_cost_analysis, generate_combined_roc

app = Flask(__name__)
RAW_DF = load_and_preprocess()

def orchestrate_pipeline(params=None):
    if params is None:
        params = {
            'test_size': 0.20, 'lr_thresh': 0.50, 'cart_depth': 3, 'cart_thresh': 0.50,
            'rf_trees': 100, 'rf_depth': 8, 'rf_thresh': 0.50,
            'bag_trees': 60, 'bag_thresh': 0.50, 'n_clusters': 3,
            'c_fp': 50.0, 'c_fn': 25.0
        }

    data = prepare_partition(RAW_DF, test_size=params['test_size'])
    lr_res = run_logistic(data, lr_thresh=params['lr_thresh'])
    cart_res = run_cart(data, max_depth=params['cart_depth'], cart_thresh=params['cart_thresh'])
    rf_res = run_random_forest(data, n_trees=params['rf_trees'], depth=params['rf_depth'], rf_thresh=params['rf_thresh'])
    bag_res = run_bagging(data, n_trees=params['bag_trees'], depth=params['rf_depth'], bag_thresh=params['bag_thresh'])
    clust_res = run_clustering(RAW_DF, NUM_COLS, n_clusters=params['n_clusters'])
    cost_res = run_cost_analysis(data['y_test'], bag_res['y_prob'], c_fp=params['c_fp'], c_fn=params['c_fn'])

    combined_roc = generate_combined_roc(data['y_test'], {
        'Logistic Reg.': lr_res['y_prob'], 'CART Tree': cart_res['y_prob'],
        'Random Forest': rf_res['y_prob'], 'Bagging': bag_res['y_prob']
    })

    plots = {
        'split_pie': data['pie_plot'],
        'combined_roc': combined_roc,
        **lr_res['plots'], **cart_res['plots'],
        **rf_res['plots'], **bag_res['plots'],
        **clust_res['plots'], **cost_res['plots']
    }

    metrics = {
        'lr': lr_res['metrics'], 'cart': cart_res['metrics'],
        'rf': rf_res['metrics'], 'bag': bag_res['metrics']
    }

    return {
        'metrics': metrics,
        'plots': plots,
        'lr_coefs': lr_res['coefs'],
        'clusters': clust_res['table'],
        'opt_thresh': cost_res['opt_thresh'],
        'min_cost': cost_res['min_cost'],
        'models': {
            'bag': bag_res['model'],
            'tfidf': data['tfidf'],
            'cat_cols': data['cat_cols'],
            'tfidf_cols': data['tfidf_cols']
        }
    }

GLOBAL_PIPELINE = orchestrate_pipeline()

@app.route('/')
def index():
    return render_template('index.html', data=GLOBAL_PIPELINE)

@app.route('/api/predict_single', methods=['POST'])
def predict_single():
    req = request.json or {}
    try:
        order_val = float(req.get('order_val', 85.0))
        price = float(req.get('price', 45.0))
        days_ret = float(req.get('days_ret', 21.0))
        ret_rate = float(req.get('ret_rate', 0.45))
        discount = float(req.get('discount', 5.0))
        acc_age = float(req.get('acc_age', 120.0))
        prior_rets = float(req.get('prior_rets', 4.0))
        pay_method = req.get('pay_method', 'Credit Card')
        reason_text = req.get('reason_text', 'Item did not suit me')

        bag_model = GLOBAL_PIPELINE['models']['bag']
        tfidf = GLOBAL_PIPELINE['models']['tfidf']
        cat_cols_ref = GLOBAL_PIPELINE['models']['cat_cols']
        tfidf_cols = GLOBAL_PIPELINE['models']['tfidf_cols']

        input_num = pd.DataFrame([{
            'Order Value': order_val, 'Price': price,
            'Days between delivery and return': days_ret,
            'Customer liftime return rate': ret_rate,
            'Discount in percentage': discount,
            'Account age (in days)': acc_age,
            'Consecutive prior returns': prior_rets
        }])

        input_cat = pd.get_dummies(
            pd.DataFrame([{'Payment method': pay_method, 'Catergory Mix': 'Gifts & Household', 'Device signals': 'Desktop'}]),
            drop_first=True, dtype=int
        ).reindex(columns=cat_cols_ref, fill_value=0)

        input_text = pd.DataFrame(tfidf.transform([reason_text]).toarray()[:, :3], columns=tfidf_cols)
        input_sample = pd.concat([input_num, input_cat, input_text], axis=1)

        prob = bag_model.predict_proba(input_sample)[0, 1]

        if prob < 0.35:
            pred_class = "Legitimate Return (Low Risk)"
            badge = "success"
            action = "Instant Return Approval & Full Refund (Zero Friction)"
        elif prob < 0.65:
            pred_class = "Moderate Risk (Flagged for Review)"
            badge = "warning"
            action = "Apply Return Shipping Fee (£3.99) & Standard Verification"
        else:
            pred_class = "Abusive Return (High Risk)"
            badge = "danger"
            action = "Manual Warehouse Inspection Required Before Refund Authorization"

        return jsonify({
            'prob': f"{prob*100:.1f}%", 'prob_raw': prob,
            'pred_class': pred_class, 'badge': badge, 'action': action
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/update', methods=['POST'])
def update_dashboard():
    global GLOBAL_PIPELINE
    req = request.json or {}
    params = {
        'test_size': float(req.get('test_size', 0.20)),
        'lr_thresh': float(req.get('lr_thresh', 0.50)),
        'cart_depth': int(req.get('cart_depth', 3)),
        'cart_thresh': float(req.get('cart_thresh', 0.50)),
        'rf_trees': int(req.get('rf_trees', 100)),
        'rf_depth': 8,
        'rf_thresh': float(req.get('rf_thresh', 0.50)),
        'bag_trees': int(req.get('bag_trees', 60)),
        'bag_thresh': float(req.get('bag_thresh', 0.50)),
        'n_clusters': int(req.get('n_clusters', 3)),
        'c_fp': float(req.get('c_fp', 50.0)),
        'c_fn': float(req.get('c_fn', 25.0))
    }
    updated_data = orchestrate_pipeline(params)
    GLOBAL_PIPELINE = updated_data

    clean_data = {
        'metrics': updated_data['metrics'],
        'plots': updated_data['plots'],
        'lr_coefs': updated_data['lr_coefs'],
        'clusters': updated_data['clusters'],
        'opt_thresh': updated_data['opt_thresh'],
        'min_cost': updated_data['min_cost']
    }
    return jsonify(clean_data)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
"""

# Write all files to disk
for path, content in files.items():
  with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("All modular files created successfully.")

# Create the ZIP bundle
zip_name = "return_abuse_dashboard.zip"
with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zipf:
  for path in files.keys():
    zipf.write(path)
  if os.path.exists("Return Abuse data filtered 1.csv"):
    zipf.write("Return Abuse data filtered 1.csv")

print(f"Bundled complete project into {zip_name}!")