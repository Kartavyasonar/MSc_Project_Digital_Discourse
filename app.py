# --- From script: 11_dashboard_streamlit.py ---
import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="UK Immigration Discourse Analysis",
    page_icon="🇬🇧",
    layout="wide"
)

# --- Caching Data Loading ---
@st.cache_data
def load_data():
    """
    Loads the dashboard data from CSV files. Uses caching for performance.
    """
    reddit_data_path = 'data/processed/reddit_dashboard_data.csv'
    laws_data_path = 'data/processed/laws_dashboard_data.csv'

    if not os.path.exists(reddit_data_path):
        st.error(f"Data file not found at {reddit_data_path}. Please run the full data processing pipeline first.")
        return None, None
    
    df_reddit = pd.read_csv(reddit_data_path)
    
    df_laws = None
    if os.path.exists(laws_data_path):
        df_laws = pd.read_csv(laws_data_path)
    
    return df_reddit, df_laws

# --- Main App ---
def main():
    st.title("🇬🇧 Analysis of Digital Immigration Discourse on Reddit")
    st.markdown("An interactive dashboard to explore themes, emotions, and related legislation in UK immigration discussions.")

    df, df_laws = load_data()

    if df is not None:
        # --- Sidebar for Filtering ---
        st.sidebar.header("Filter by Topic")
        topic_list = sorted(df['Final_Topic_Label'].unique())
        selected_topic = st.sidebar.selectbox("Select a topic to explore:", topic_list)
        
        # --- Filtered Data ---
        df_filtered = df[df['Final_Topic_Label'] == selected_topic]
        
        # --- Main Panel ---
        st.header(f"Analysis for Topic: {selected_topic}")
        st.markdown(f"**Total Posts:** {len(df_filtered)}")

        col1, col2 = st.columns(2)

        # --- Word Cloud ---
        with col1:
            st.subheader("Key Terms Word Cloud")
            try:
                text = " ".join(review for review in df_filtered['text_cleaned'])
                wordcloud = WordCloud(background_color="white", colormap="viridis", width=800, height=400).generate(text)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
            except ValueError:
                st.warning("Not enough text to generate a word cloud for this topic.")

        # --- Emotion Distribution ---
        with col2:
            st.subheader("Emotion Distribution")
            emotion_counts = df_filtered['emotion_label'].value_counts().head(10)
            st.bar_chart(emotion_counts)
            
        # --- Global Heatmap ---
        st.header("Global View: Topic-Emotion Heatmap")
        crosstab = pd.crosstab(df['Final_Topic_Label'], df['emotion_label'])
        fig_heatmap, ax_heatmap = plt.subplots(figsize=(16, 10))
        sns.heatmap(crosstab, cmap='YlGnBu', annot=False, ax=ax_heatmap)
        ax_heatmap.set_title('Heatmap of Topic and Emotion Correlations', fontsize=16)
        st.pyplot(fig_heatmap)

        # --- Sample Posts ---
        st.header("Sample Posts from this Topic")
        st.dataframe(df_filtered[['title', 'selftext', 'subreddit', 'emotion_label']].head(5))
        
        # --- Legislation Links ---
        if df_laws is not None:
            st.header("UK Legislation Related to This Topic")
            linked_laws = df_laws[df_laws['Topic'] == selected_topic]
            if not linked_laws.empty:
                st.table(linked_laws[['Legislation_Title', 'Link', 'Date']])
            else:
                st.info("No specific legislation was strongly matched to this topic.")

if __name__ == '__main__':
    main()