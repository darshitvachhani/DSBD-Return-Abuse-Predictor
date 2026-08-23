import pandas as pd
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

    # Large Single Pie Chart
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.pie(
        [len(train_df), len(test_df)],
        labels=['Train Set', 'Test Set'],
        autopct='%1.1f%%',
        colors=['#2b5c8f', '#d95f02'],
        startangle=140,
        textprops={'fontsize': 13, 'weight': 'bold'},
        radius=1.15
    )
    ax.set_title(f"Partitioning ({int(test_size*100)}% Test)", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
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