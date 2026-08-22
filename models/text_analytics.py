import re
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import STOPWORDS, WordCloud
from models.utils import fig_to_base64

def run_text_analytics(train_df, text_col='Free text return reason'):
    # 1. Generate Training Corpus Word Cloud
    all_text = " ".join(train_df[text_col].dropna().astype(str).tolist())
    cleaned_text = re.sub(r'[^\w\s]', ' ', all_text).lower()
    
    wc = WordCloud(
        width=750, height=350,
        background_color='white',
        colormap='magma',
        stopwords=STOPWORDS,
        max_words=60,
        random_state=42
    ).generate(cleaned_text)
    
    fig_wc, ax_wc = plt.subplots(figsize=(6.5, 3.2))
    ax_wc.imshow(wc, interpolation='bilinear')
    ax_wc.axis('off')
    ax_wc.set_title("Return Reason Word Cloud (Training Partition)", fontsize=10, fontweight='bold')
    wc_b64 = fig_to_base64(fig_wc)

    # 2. Extract Top Reason Term Frequencies
    words = [w for w in cleaned_text.split() if w not in STOPWORDS and len(w) > 2]
    freq_series = pd.Series(words).value_counts().head(8).sort_values(ascending=True)
    
    fig_freq, ax_freq = plt.subplots(figsize=(6, 3.2))
    freq_series.plot(kind='barh', color='#8b5cf6', edgecolor='black', ax=ax_freq)
    ax_freq.set_title("Top 8 Return Reason Keywords", fontsize=10, fontweight='bold')
    ax_freq.set_xlabel("Term Frequency in Corpus")
    ax_freq.grid(axis='x', linestyle=':', alpha=0.6)
    freq_b64 = fig_to_base64(fig_freq)

    return {
        'plots': {
            'wordcloud': wc_b64,
            'text_freq': freq_b64
        }
    }