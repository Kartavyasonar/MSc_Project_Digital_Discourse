import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import os

REDDIT_DATA_PATH = "data/processed/reddit_dashboard_data.csv"
LEGISLATION_PATH = "data/processed/legislation_cleaned.csv"
OUTPUT_PATH = "data/processed/topic_legislation_mapping.csv"

def clean_text(text):
    if pd.isna(text):
        return ""
    return str(text).lower().strip()

def match_legislation_to_topic(topic, laws_df, threshold=75):
    matches = []
    topic_cleaned = clean_text(topic)

    for _, row in laws_df.iterrows():
        combined_text = f"{row['keyword']} {row['title']}"
        score = fuzz.partial_ratio(topic_cleaned, clean_text(combined_text))
        if score >= threshold:
            matches.append({
                "topic": topic,
                "law_keyword": row['keyword'],
                "law_title": row['title'],
                "law_link": row['link'],
                "law_date": row['date'],
                "match_score": score
            })
    return matches

def link_legislation_to_topics():
    reddit_df = pd.read_csv(REDDIT_DATA_PATH)
    laws_df = pd.read_csv(LEGISLATION_PATH)

    unique_topics = reddit_df["Final_Topic_Label"].dropna().unique()
    all_matches = []

    for topic in unique_topics:
        topic_matches = match_legislation_to_topic(topic, laws_df)
        if topic_matches:
            all_matches.extend(topic_matches)
        else:
            all_matches.append({
                "topic": topic,
                "law_keyword": "No match",
                "law_title": "No match",
                "law_link": "",
                "law_date": "",
                "match_score": 0
            })

    df_out = pd.DataFrame(all_matches)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved legislation-topic links to {OUTPUT_PATH}")

if __name__ == "__main__":
    link_legislation_to_topics()
