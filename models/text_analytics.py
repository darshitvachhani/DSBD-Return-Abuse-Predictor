import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from models.utils import fig_to_base64

def run_text_analytics(train_df, text_col='Free text return reason'):
    corpus_series = train_df[text_col].dropna().astype(str)
    all_text = " ".join(corpus_series.tolist())
    cleaned_text = re.sub(r'[^\w\s]', ' ', all_text).lower()

    # 1. Circular Word Cloud (Left Plot)
    x, y = np.ogrid[:600, :600]
    circle_mask = (x - 300) ** 2 + (y - 300) ** 2 > 290 ** 2
    circle_mask = 255 * circle_mask.astype(np.uint8)

    wc = WordCloud(
        width=600, height=600,
        background_color='white',
        colormap='magma',
        stopwords=STOPWORDS,
        mask=circle_mask,
        contour_width=2,
        contour_color='#8b5cf6',
        max_words=60,
        random_state=42
    ).generate(cleaned_text)

    fig_wc, ax_wc = plt.subplots(figsize=(5.5, 4.0))
    ax_wc.imshow(wc, interpolation='bilinear')
    ax_wc.axis('off')
    ax_wc.set_title("Return Reason Word Cloud (Corpus Overview)", fontsize=10, fontweight='bold')
    plt.tight_layout()
    wc_b64 = fig_to_base64(fig_wc)

    # 2. Combined TF-IDF Weight vs. Raw Count (Right Plot)
    tfidf_vec = TfidfVectorizer(max_features=8, stop_words='english')
    tfidf_matrix = tfidf_vec.fit_transform(corpus_series)
    tfidf_scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
    feature_names = np.array(tfidf_vec.get_feature_names_out())

    cnt_vec = CountVectorizer(vocabulary=feature_names)
    cnt_matrix = cnt_vec.fit_transform(corpus_series)
    raw_counts = np.asarray(cnt_matrix.sum(axis=0)).ravel()

    df_tfidf = pd.DataFrame({
        'Feature': feature_names,
        'Raw_Count': raw_counts,
        'TFIDF_Weight': tfidf_scores
    }).sort_values('TFIDF_Weight', ascending=True)

    fig_dual, ax1 = plt.subplots(figsize=(5.5, 4.0))
    y_pos = np.arange(len(df_tfidf))
    bar_height = 0.38

    # Primary axis: TF-IDF Weights (Purple)
    bars1 = ax1.barh(y_pos - bar_height/2, df_tfidf['TFIDF_Weight'], height=bar_height, color='#7c3aed', label='Mean TF-IDF Weight')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df_tfidf['Feature'], fontsize=9, fontweight='bold')
    ax1.set_xlabel('Learned TF-IDF Weight (Signal Quality)', color='#7c3aed', fontsize=8, fontweight='bold')
    ax1.grid(axis='x', linestyle=':', alpha=0.5)

    # Secondary top axis: Raw Counts (Slate Gray)
    ax2 = ax1.twiny()
    bars2 = ax2.barh(y_pos + bar_height/2, df_tfidf['Raw_Count'], height=bar_height, color='#94a3b8', alpha=0.6, label='Raw Occurrence Count')
    ax2.set_xlabel('Raw Corpus Word Count (Volume)', color='#475569', fontsize=8, fontweight='bold')

    # Combined Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=7.5)

    ax1.set_title("TF-IDF Attenuation vs. Raw Count", fontsize=10, fontweight='bold', pad=12)
    plt.tight_layout()
    tfidf_b64 = fig_to_base64(fig_dual)

    return {
        'plots': {
            'wordcloud': wc_b64,
            'tfidf_comparison': tfidf_b64,
            'text_freq': tfidf_b64
        }
    }