import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from models.utils import fig_to_base64


def plot_decision_boundaries(X_2d, labels, kmeans, x_label, y_label, title, palette):
    """Generates a 2D decision boundary plot with clear cluster demarcation zones."""
    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    # Meshgrid resolution for boundaries
    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot filled decision regions
    ax.contourf(xx, yy, Z, alpha=0.25, cmap=palette)
    ax.contour(xx, yy, Z, colors='k', linewidths=0.7, linestyles='--')

    # Scatter points
    scatter = ax.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=labels, cmap=palette,
        edgecolor='k', s=25, alpha=0.75
    )

    # Centroids
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
    cust_features = ['Consecutive prior returns', 'Customer liftime return rate']
    X_cust = raw_df[cust_features].dropna()
    scaler_c = StandardScaler()
    X_cust_scaled = scaler_c.fit_transform(X_cust)

    kmeans_cust = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cust_labels = kmeans_cust.fit_predict(X_cust_scaled)

    df_cust = raw_df.loc[X_cust.index].copy()
    df_cust['Cluster'] = cust_labels

    cust_summary = []
    for c_id in sorted(df_cust['Cluster'].unique()):
        sub = df_cust[df_cust['Cluster'] == c_id]
        abuse_rate = sub['is_abuse'].mean()
        
        if abuse_rate > 0.45:
            seg_name = f"Cluster {c_id}: Chronic Return Abusers (High Risk)"
        elif abuse_rate > 0.20:
            seg_name = f"Cluster {c_id}: Moderate / Indecisive Shoppers"
        else:
            seg_name = f"Cluster {c_id}: Loyal Low-Return Core"

        cust_summary.append({
            'Segment': seg_name,
            'Customer Count': int(len(sub)),
            'Abuse Rate': f"{abuse_rate * 100:.1f}%",
            'Avg Prior Returns': f"{sub['Consecutive prior returns'].mean():.2f}",
            'Lifetime Return Rate': f"{sub['Customer liftime return rate'].mean():.2f}"
        })

    plot_cust = plot_decision_boundaries(
        X_cust_scaled, cust_labels, kmeans_cust,
        "Prior Returns", "Lifetime Return Rate",
        f"Customer Behavioral Clusters (K={n_clusters})", "viridis"
    )

    # ==========================================
    # 2. PRODUCT / ORDER DYNAMICS SEGMENTATION
    # ==========================================
    prod_features = ['Order Value', 'Days between delivery and return']
    X_prod = raw_df[prod_features].dropna()
    scaler_p = StandardScaler()
    X_prod_scaled = scaler_p.fit_transform(X_prod)

    kmeans_prod = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    prod_labels = kmeans_prod.fit_predict(X_prod_scaled)

    df_prod = raw_df.loc[X_prod.index].copy()
    df_prod['Cluster'] = prod_labels

    prod_summary = []
    for c_id in sorted(df_prod['Cluster'].unique()):
        sub = df_prod[df_prod['Cluster'] == c_id]
        abuse_rate = sub['is_abuse'].mean()

        if sub['Order Value'].mean() > raw_df['Order Value'].mean() and sub['Days between delivery and return'].mean() > 20:
            seg_name = f"Cluster {c_id}: High-Ticket Late Returns (Wardrobing Risk)"
        elif sub['Order Value'].mean() < raw_df['Order Value'].mean():
            seg_name = f"Cluster {c_id}: Low-Ticket Immediate Fit Issues"
        else:
            seg_name = f"Cluster {c_id}: Standard Retail Apparel Orders"

        prod_summary.append({
            'Segment': seg_name,
            'Product Count': int(len(sub)),
            'Abuse Rate': f"{abuse_rate * 100:.1f}%",
            'Avg Order Value': f"£{sub['Order Value'].mean():.2f}",
            'Avg Return Window (Days)': f"{sub['Days between delivery and return'].mean():.1f} days"
        })

    plot_prod = plot_decision_boundaries(
        X_prod_scaled, prod_labels, kmeans_prod,
        "Order Value (£)", "Days to Return",
        f"Product / Order Dynamics Clusters (K={n_clusters})", "plasma"
    )

    return {
        'plots': {
            'cust_cluster': plot_cust,
            'prod_cluster': plot_prod
        },
        'cust_table': cust_summary,
        'prod_table': prod_summary,
        'table': cust_summary
    }