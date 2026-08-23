import numpy as np
import re
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import STOPWORDS, WordCloud
from models.utils import fig_to_base64

def run_text_analytics(train_df, text_col='Free text return reason'):
    all_text = " ".join(train_df[text_col].dropna().astype(str).tolist())
    cleaned_text = re.sub(r'[^\w\s]', ' ', all_text).lower()
    
    # Circular mask
    x, y = np.ogrid[:600, :600]
    circle_mask = (x - 300) ** 2 + (y - 300) ** 2 > 290 ** 2
    circle_mask = 255 * circle_mask.astype(np.uint8)

    # 1. Circular Word Cloud Plot
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
    ax_wc.set_title("Return Reason Word Cloud (Training Partition)", fontsize=10, fontweight='bold')
    plt.tight_layout()
    wc_b64 = fig_to_base64(fig_wc)

    # 2. Term Frequency Plot (Matching Dimensions)
    words = [w for w in cleaned_text.split() if w not in STOPWORDS and len(w) > 2]
    freq_series = pd.Series(words).value_counts().head(8).sort_values(ascending=True)
    
    fig_freq, ax_freq = plt.subplots(figsize=(5.5, 4.0))
    freq_series.plot(kind='barh', color='#8b5cf6', edgecolor='black', ax=ax_freq)
    ax_freq.set_title("Top 8 Return Reason Keywords", fontsize=10, fontweight='bold')
    ax_freq.set_xlabel("Term Frequency in Corpus")
    ax_freq.grid(axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()
    freq_b64 = fig_to_base64(fig_freq)

    return {
        'plots': {
            'wordcloud': wc_b64,
            'text_freq': freq_b64
        }
    }