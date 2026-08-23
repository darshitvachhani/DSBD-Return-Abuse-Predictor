import warnings
import matplotlib
from flask import Flask, render_template, request, jsonify
import pandas as pd

matplotlib.use('Agg')
warnings.filterwarnings('ignore')

from models.data_pipeline import load_and_preprocess, prepare_partition, NUM_COLS
from models.text_analytics import run_text_analytics
from models.logistic_regression import run_logistic
from models.cart_tree import run_cart
from models.random_forest import run_random_forest
from models.bagging import run_bagging
from models.clustering import run_clustering
from models.cost_optimization import run_cost_analysis, generate_combined_roc
from models.implications import generate_model_implications
from models.business_impact import generate_policy_matrix_plot, calculate_business_savings

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

    # 1. Partitioning & Preprocessing
    data = prepare_partition(RAW_DF, test_size=params['test_size'])
    
    # 2. Text Analytics & Classifiers
    text_res = run_text_analytics(data['train_df'])
    lr_res = run_logistic(data, lr_thresh=params['lr_thresh'])
    cart_res = run_cart(data, max_depth=params['cart_depth'], cart_thresh=params['cart_thresh'])
    rf_res = run_random_forest(data, n_trees=params['rf_trees'], depth=params['rf_depth'], rf_thresh=params['rf_thresh'])
    bag_res = run_bagging(data, n_trees=params['bag_trees'], depth=params['rf_depth'], bag_thresh=params['bag_thresh'])
    
    # 3. Dual Clustering (Customer & Product)
    clust_res = run_clustering(RAW_DF, NUM_COLS, n_clusters=params['n_clusters'])
    
    # 4. Cost Optimization & Multi-Model ROC
    cost_res = run_cost_analysis(data['y_test'], bag_res['y_prob'], c_fp=params['c_fp'], c_fn=params['c_fn'])
    combined_roc = generate_combined_roc(data['y_test'], {
        'Logistic Reg.': lr_res['y_prob'], 'CART Tree': cart_res['y_prob'],
        'Random Forest': rf_res['y_prob'], 'Bagging': bag_res['y_prob']
    })

    # 5. Implications & Business Impact Logic
    metrics = {
        'lr': lr_res['metrics'], 'cart': cart_res['metrics'],
        'rf': rf_res['metrics'], 'bag': bag_res['metrics']
    }
    implications_data = generate_model_implications(metrics, cost_res['opt_thresh'])
    policy_matrix_plot = generate_policy_matrix_plot()
    biz_impact_data = calculate_business_savings(RAW_DF, best_auc=float(bag_res['metrics']['auc']))

    plots = {
        'split_pie': data['pie_plot'],
        'combined_roc': combined_roc,
        'policy_matrix': policy_matrix_plot,
        **text_res['plots'],
        **lr_res['plots'], **cart_res['plots'],
        **rf_res['plots'], **bag_res['plots'],
        **clust_res['plots'], **cost_res['plots']
    }

    return {
        'metrics': metrics,
        'plots': plots,
        'lr_coefs': lr_res['coefs'],
        'cust_clusters': clust_res.get('cust_table', clust_res.get('table', [])),
        'prod_clusters': clust_res.get('prod_table', []),
        'opt_thresh': cost_res['opt_thresh'],
        'min_cost': cost_res['min_cost'],
        'implications': implications_data,
        'biz_impact': biz_impact_data,
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
        'cust_clusters': updated_data['cust_clusters'],
        'prod_clusters': updated_data['prod_clusters'],
        'opt_thresh': updated_data['opt_thresh'],
        'min_cost': updated_data['min_cost']
    }
    return jsonify(clean_data)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)