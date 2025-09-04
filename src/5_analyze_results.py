import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

# --- Directory Setup ---
os.makedirs('reports/figures', exist_ok=True)


def generate_visualizations(input_path='data/processed/reddit_dashboard_data.csv'):
    """
    Generates and saves all key visualizations for the report.
    """
    print("--- Generating visualizations ---")
    if not os.path.exists(input_path):
        print(f"Error: Dashboard data not found at {input_path}. Please run the full pipeline first.")
        return

    df = pd.read_csv(input_path)

    # Figure 1: Topic Distribution
    plt.figure(figsize=(12, 8))
    sns.countplot(y='Final_Topic_Label', data=df, order=df['Final_Topic_Label'].value_counts().index, palette='viridis')
    plt.title('Figure 5.1: Distribution of Posts by Topic', fontsize=16)
    plt.xlabel('Number of Posts', fontsize=12)
    plt.ylabel('Topic', fontsize=12)
    plt.tight_layout()
    plt.savefig('reports/figures/topic_distribution.png')
    print("Saved topic distribution plot.")

    # Figure 2: Emotion Distribution
    plt.figure(figsize=(12, 8))
    sns.countplot(y='emotion_label', data=df, order=df['emotion_label'].value_counts().index[:10], palette='plasma')
    plt.title('Figure 5.3: Overall Distribution of Top 10 Emotions', fontsize=16)
    plt.xlabel('Number of Posts', fontsize=12)
    plt.ylabel('Emotion', fontsize=12)
    plt.tight_layout()
    plt.savefig('reports/figures/emotion_distribution.png')
    print("Saved emotion distribution plot.")

    # Figure 3: Topic-Emotion Heatmap
    crosstab = pd.crosstab(df['Final_Topic_Label'], df['emotion_label'])
    plt.figure(figsize=(16, 10))
    sns.heatmap(crosstab, cmap='YlGnBu', annot=False)
    plt.title('Figure 5.4: Heatmap of Topic and Emotion Correlations', fontsize=16)
    plt.xlabel('Emotion', fontsize=12)
    plt.ylabel('Topic', fontsize=12)
    plt.tight_layout()
    plt.savefig('reports/figures/topic_emotion_heatmap.png')
    print("Saved topic-emotion heatmap.")
    
    print("--- All visualizations generated and saved to reports/figures/ ---")


def generate_summary_tables(input_path='data/processed/reddit_dashboard_data.csv'):
    """
    Generates summary tables (as DataFrames) for the report.
    """
    print("--- Generating summary tables ---")
    if not os.path.exists(input_path):
        print(f"Error: Dashboard data not found at {input_path}. Please run the full pipeline first.")
        return

    df = pd.read_csv(input_path)

    # Table 1: Topic Distribution
    topic_counts = df['Final_Topic_Label'].value_counts().reset_index()
    topic_counts.columns = ['Topic', 'Post Count']
    topic_counts['Percentage of Corpus'] = (topic_counts['Post Count'] / len(df) * 100).round(1).astype(str) + '%'
    print("\n--- Table 5.1: Distribution of Posts Across Final Identified Topics ---")
    print(topic_counts)
    topic_counts.to_csv('reports/table_topic_distribution.csv', index=False)

    # Table 2: Emotion Distribution
    emotion_counts = df['emotion_label'].value_counts().reset_index().head(10)
    emotion_counts.columns = ['Emotion', 'Post Count']
    emotion_counts['Percentage of Corpus'] = (emotion_counts['Post Count'] / len(df) * 100).round(1).astype(str) + '%'
    print("\n--- Table 5.2 (from report): Top 10 Emotions Detected in the Corpus ---")
    print(emotion_counts)
    emotion_counts.to_csv('reports/table_emotion_distribution.csv', index=False)
    
    print("\n--- Summary tables generated and saved to reports/ ---")


if __name__ == '__main__':
    generate_visualizations()
    generate_summary_tables()