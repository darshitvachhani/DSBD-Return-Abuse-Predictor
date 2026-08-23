import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from models.utils import fig_to_base64

def plot_decision_boundaries(X_2d, labels, kmeans, x_label, y_label, title, palette):
    fig, ax = plt.subplots(figsize=(6.2, 4.0))

    # Pad around actual points
    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))

    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.25, cmap=palette)
    ax.contour(xx, yy, Z, colors='k', linewidths=0.7, linestyles='--')

    ax.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=labels, cmap=palette,
        edgecolor='k', s=25, alpha=0.75
    )

    centroids = kmeans.cluster_centers_
    ax.scatter(
        centroids[:, 0], centroids[:, 1],
        marker='X', s=160, c='red', edgecolor='black',
        label='Centroids', zorder=10
    )

    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel(f"{x_label} (Standardised)", fontsize=9)
    ax.set_ylabel(f"{y_label} (Standardised)", fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()

    return fig_to_base64(fig)

def run_clustering(raw_df, num_cols, n_clusters=3):
    # ==========================================
    # 1. CUSTOMER BEHAVIOR SEGMENTATION
    # ==========================================
    cust_df = raw_df[['Consecutive prior returns', 'Customer liftime return rate', 'is_abuse']].dropna().copy()
    # Robust percentile cap to avoid single-customer distortion on plots
    p99_prior = cust_df['Consecutive prior returns'].quantile(0.99)
    cust_df['plot_prior'] = cust_df['Consecutive prior returns'].clip(upper=p99_prior)
    
    scaler_c = StandardScaler()
    X_cust_scaled = scaler_c.fit_transform(cust_df[['plot_prior', 'Customer liftime return rate']])

    kmeans_cust = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cust_df['Cluster'] = kmeans_cust.fit_predict(X_cust_scaled)

    cust_summary = []
    for c_id in sorted(cust_df['Cluster'].unique()):
        sub = cust_df[cust_df['Cluster'] == c_id]
        abuse_rate = sub['is_abuse'].mean()
        avg_prior = sub['Consecutive prior returns'].mean()
        avg_rate = sub['Customer liftime return rate'].mean()

        if avg_prior > 10:
            seg_name = f"Cluster {c_id}: Serial Habitual Returners"
        elif avg_rate > 0.60:
            seg_name = f"Cluster {c_id}: High-Velocity Chronic Abusers"
        elif abuse_rate > 0.25:
            seg_name = f"Cluster {c_id}: Moderate / Indecisive Shoppers"
        else:
            seg_name = f"Cluster {c_id}: Low-Risk Loyal Customers"

        cust_summary.append({
            'Segment': seg_name,
            'Customer Count': int(len(sub)),
            'Abuse Rate': f"{abuse_rate * 100:.1f}%",
            'Avg Prior Returns': f"{avg_prior:.2f}",
            'Lifetime Return Rate': f"{avg_rate:.2f}"
        })

    plot_cust = plot_decision_boundaries(
        X_cust_scaled, cust_df['Cluster'].values, kmeans_cust,
        "Prior Returns", "Lifetime Return Rate",
        f"Customer Behavioral Clusters (K={n_clusters})", "viridis"
    )

    # ==========================================
    # 2. ORDER / BASKET DYNAMICS SEGMENTATION
    # ==========================================
    order_df = raw_df[['Order Value', 'Days between delivery and return', 'is_abuse']].dropna().copy()
    p99_val = order_df['Order Value'].quantile(0.99)
    order_df['plot_val'] = order_df['Order Value'].clip(upper=p99_val)

    scaler_o = StandardScaler()
    X_order_scaled = scaler_o.fit_transform(order_df[['plot_val', 'Days between delivery and return']])

    kmeans_order = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    order_df['Cluster'] = kmeans_order.fit_predict(X_order_scaled)

    order_summary = []
    for c_id in sorted(order_df['Cluster'].unique()):
        sub = order_df[order_df['Cluster'] == c_id]
        abuse_rate = sub['is_abuse'].mean()
        avg_basket = sub['Order Value'].mean()
        avg_days = sub['Days between delivery and return'].mean()

        if avg_days > 18 and avg_basket > 60:
            seg_name = f"Cluster {c_id}: Late Wardrobing Window"
        elif avg_days > 18:
            seg_name = f"Cluster {c_id}: Slow Standard Returns"
        elif avg_basket < 45:
            seg_name = f"Cluster {c_id}: Rapid Low-Ticket Exchanges"
        else:
            seg_name = f"Cluster {c_id}: Standard Basket Returns"

        order_summary.append({
            'Segment': seg_name,
            'Product Count': int(len(sub)),
            'Abuse Rate': f"{abuse_rate * 100:.1f}%",
            'Avg Order Value': f"£{avg_basket:.2f}",
            'Avg Return Window (Days)': f"{avg_days:.1f} days"
        })

    plot_order = plot_decision_boundaries(
        X_order_scaled, order_df['Cluster'].values, kmeans_order,
        "Order Value (£)", "Days to Return",
        f"Order Dynamics Clusters (K={n_clusters})", "plasma"
    )

    # ==========================================
    # 3. PRODUCT CATEGORY & MERCHANDISING CLUSTERING
    # ==========================================
    cat_df = raw_df[['Price', 'Discount in percentage', 'is_abuse']].dropna().copy()
    p99_price = cat_df['Price'].quantile(0.99)
    cat_df['plot_price'] = cat_df['Price'].clip(upper=p99_price)

    scaler_cat = StandardScaler()
    X_cat_scaled = scaler_cat.fit_transform(cat_df[['plot_price', 'Discount in percentage']])

    kmeans_cat = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cat_df['Cluster'] = kmeans_cat.fit_predict(X_cat_scaled)

    cat_summary = []
    for c_id in sorted(cat_df['Cluster'].unique()):
        sub = cat_df[cat_df['Cluster'] == c_id]
        abuse_rate = sub['is_abuse'].mean()
        avg_p = sub['Price'].mean()
        avg_d = sub['Discount in percentage'].mean()

        if avg_p > 50:
            seg_name = f"Cluster {c_id}: Premium High-Value SKUs"
        elif avg_d > 20:
            seg_name = f"Cluster {c_id}: High-Discount Clearance Lines"
        else:
            seg_name = f"Cluster {c_id}: Core Standard Merchandise"

        cat_summary.append({
            'Segment': seg_name,
            'Item Count': int(len(sub)),
            'Abuse Rate': f"{abuse_rate * 100:.1f}%",
            'Avg Price': f"£{avg_p:.2f}",
            'Avg Discount': f"{avg_d:.1f}%"
        })

    plot_cat = plot_decision_boundaries(
        X_cat_scaled, cat_df['Cluster'].values, kmeans_cat,
        "Unit Price (£)", "Discount (%)",
        f"Product Category Risk Clusters (K={n_clusters})", "cividis"
    )

    return {
        'plots': {
            'cust_cluster': plot_cust,
            'prod_cluster': plot_order,
            'cat_cluster': plot_cat
        },
        'cust_table': cust_summary,
        'prod_table': order_summary,
        'cat_table': cat_summary,
        'table': cust_summary
    }