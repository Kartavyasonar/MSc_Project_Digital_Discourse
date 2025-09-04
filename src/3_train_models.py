import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from fuzzywuzzy import fuzz
import os

# --- Directory Setup ---
os.makedirs('data/processed', exist_ok=True)


def run_topic_modeling(input_path='data/processed/reddit_cleaned.csv', output_path='data/processed/reddit_with_topics.csv'):
    """
    Performs topic modeling on the cleaned text data using BERTopic.
    """
    print("--- Starting topic modeling ---")
    df = pd.read_csv(input_path)
    docs = df['text_cleaned'].tolist()

    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    topic_model = BERTopic(embedding_model=embedding_model, min_topic_size=10, verbose=True)
    
    topics, _ = topic_model.fit_transform(docs)
    
    df['topic_id'] = topics
    topic_info = topic_model.get_topic_info()
    df = pd.merge(df, topic_info[['Topic', 'Name']], left_on='topic_id', right_on='Topic', how='left')
    df = df.rename(columns={'Name': 'topic_name'})
    
    df.to_csv(output_path, index=False)
    print(f"--- Topic modeling complete. Saved results to {output_path} ---")
    return df


def run_emotion_detection(input_path='data/processed/reddit_cleaned.csv', output_path='data/processed/reddit_with_emotions.csv'):
    """
    Runs emotion detection using the fine-grained GoEmotions model.
    """
    print("--- Starting emotion detection ---")
    df = pd.read_csv(input_path)
    
    # Using the more detailed GoEmotions model as the primary choice
    model_name = "monologg/bert-base-cased-goemotions-original"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    labels = model.config.id2label
    
    emotions = []
    for text in df['full_text']:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs).logits
        
        # Apply sigmoid to get probabilities and find the most likely emotion
        probabilities = torch.sigmoid(logits)
        top_emotion_idx = probabilities.argmax().item()
        emotions.append(labels[top_emotion_idx])
        
    df['emotion_label'] = emotions
    df.to_csv(output_path, index=False)
    print(f"--- Emotion detection complete. Saved results to {output_path} ---")
    return df


def link_legislation_to_topics(
    topics_path='data/processed/reddit_with_final_topics.csv',
    legislation_path='data/raw/uk_legislation.csv',
    output_path='data/processed/topic_legislation_mapping.csv'
):
    """
    Links identified topics to relevant legislation using fuzzy string matching.
    """
    print("--- Linking topics to legislation ---")
    if not os.path.exists(legislation_path):
        print(f"Error: Legislation file not found at {legislation_path}. Skipping.")
        return

    df_topics = pd.read_csv(topics_path)
    df_laws = pd.read_csv(legislation_path)
    
    unique_topics = df_topics['Final_Topic_Label'].unique()
    
    mappings = []
    for topic in unique_topics:
        best_match_score = 0
        best_match_law = None
        for _, law in df_laws.iterrows():
            score = fuzz.partial_ratio(topic.lower(), str(law['title']).lower())
            if score > best_match_score:
                best_match_score = score
                best_match_law = law
        
        if best_match_law is not None and best_match_score > 75: # Confidence threshold
            mappings.append({
                'Topic': topic,
                'Legislation_Title': best_match_law['title'],
                'Link': best_match_law['link'],
                'Date': best_match_law['date'],
                'Match_Score': best_match_score
            })
            
    df_mapping = pd.DataFrame(mappings)
    df_mapping.to_csv(output_path, index=False)
    print(f"--- Topic-legislation mapping complete. Saved to {output_path} ---")
    return df_mapping

if __name__ == '__main__':
    run_topic_modeling()
    run_emotion_detection()
    # Note: The linking script depends on the output of process_data.py
    # which should be run after this script. A master script would handle this ordering.
    print("Run process_data.py after this script, then call link_legislation_to_topics() from this script again or a master script.")