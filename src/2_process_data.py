import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import os

# --- Download NLTK data if not present ---
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

# --- Directory Setup ---
os.makedirs('data/processed', exist_ok=True)


def clean_text_data(input_path='data/raw/reddit_scraped_posts.csv', output_path='data/processed/reddit_cleaned.csv'):
    """
    Cleans raw Reddit text data by normalizing, tokenizing, and removing stopwords.
    """
    print("--- Starting text cleaning ---")
    df = pd.read_csv(input_path)
    df['selftext'] = df['selftext'].fillna('')
    df['full_text'] = df['title'] + ' ' + df['selftext']
    
    stop_words = set(stopwords.words('english'))

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\@\w+|\#', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        tokens = word_tokenize(text)
        filtered_tokens = [w for w in tokens if not w in stop_words]
        return " ".join(filtered_tokens)

    df['text_cleaned'] = df['full_text'].apply(clean_text)
    df.to_csv(output_path, index=False)
    print(f"--- Text cleaning complete. Saved to {output_path} ---")
    return df


def apply_topic_labels(input_path='data/processed/reddit_with_topics.csv', output_path='data/processed/reddit_with_final_topics.csv'):
    """
    Applies a multi-stage, rule-based labeling process to the topic model output.
    """
    print("--- Applying custom and refined topic labels ---")
    df = pd.read_csv(input_path)

    
    def map_custom_label(topic_name):
        if "brp" in topic_name or "biometric" in topic_name: return "BRP & Biometric Problems"
        if "visa" in topic_name or "application" in topic_name: return "Visa Applications & Issues"
        if "settled" in topic_name or "euss" in topic_name: return "EUSS & Settled Status"
        if "delay" in topic_name or "ukvi" in topic_name: return "UKVI Delays & Complaints"
        if "share" in topic_name or "work" in topic_name: return "Right to Work / Share Code"
        if "student" in topic_name: return "Student Visa & Universities"
        if "ilr" in topic_name: return "ILR & Settlement"
        if "law" in topic_name or "policy" in topic_name: return "UK Immigration Law & Policy"
        if "nhs" in topic_name or "health" in topic_name: return "NHS & Health Access"
        return "General Immigration Concerns"
    
    df['Final_Topic_Label'] = df['topic_name'].apply(map_custom_label)

    # Stages 2 & 3: Refinement and fallback (logic from 6b and 6c)
    def refine_label(row):
        if row['Final_Topic_Label'] == "General Immigration Concerns":
            text = row['full_text'].lower()
            if any(kw in text for kw in ["delay", "waiting", "complaint"]): return "UKVI Delays & Complaints"
            if any(kw in text for kw in ["brp", "biometric"]): return "BRP & Biometric Problems"
            # Add more refinement rules as needed
        return row['Final_Topic_Label']

    df['Final_Topic_Label'] = df.apply(refine_label, axis=1)
    
    df.to_csv(output_path, index=False)
    print(f"--- Topic labeling complete. Saved to {output_path} ---")
    return df

def create_dashboard_data(
    topics_path='data/processed/reddit_with_final_topics.csv',
    emotions_path='data/processed/reddit_with_emotions.csv',
    legislation_path='data/processed/topic_legislation_mapping.csv',
    output_reddit_path='data/processed/reddit_dashboard_data.csv',
    output_laws_path='data/processed/laws_dashboard_data.csv'
):
    """
    Merges all analysis outputs into final datasets for the Streamlit dashboard.
    """
    print("--- Merging data for dashboard ---")
    df_topics = pd.read_csv(topics_path)
    df_emotions = pd.read_csv(emotions_path)
    
    # Merge topics and emotions
    df_dashboard = pd.merge(df_topics, df_emotions[['id', 'emotion_label']], on='id', how='left')
    df_dashboard.to_csv(output_reddit_path, index=False)
    print(f"--- Reddit dashboard data created. Saved to {output_reddit_path} ---")

    # Prepare legislation data (simple copy/rename in this case)
    if os.path.exists(legislation_path):
        df_laws = pd.read_csv(legislation_path)
        df_laws.to_csv(output_laws_path, index=False)
        print(f"--- Laws dashboard data created. Saved to {output_laws_path} ---")
    else:
        print(f"Warning: Legislation mapping file not found at {legislation_path}. Skipping.")

    return df_dashboard

if __name__ == '__main__':
    clean_text_data()
    # Note: The following functions depend on the output of the modeling script.
    # They would typically be run after train_models.py
    # For a simple pipeline, you might call them from a master script.
    print("To run the full processing pipeline, first run train_models.py, then run the remaining functions in this script.")