import pandas as pd
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
