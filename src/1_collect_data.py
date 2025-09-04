import praw
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import os

# --- Configuration ---
# Ensure you have a praw.ini file set up or use environment variables
# For simplicity, this script assumes praw.ini
CLIENT_ID = "YOUR_CLIENT_ID"
CLIENT_SECRET = "YOUR_CLIENT_SECRET"
USER_AGENT = "MScProject/0.1 by YourUsername"

# --- Directory Setup ---
os.makedirs('data/raw', exist_ok=True)

def scrape_reddit_data():
    """
    Scrapes posts from specified subreddits based on keywords.
    """
    print("--- Starting Reddit Data Scraping ---")
    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT,
    )

    subreddits_to_scrape = [
        'ukvisa', 'spousevisauk', 'immigration', 'visas', 'immigrationUK',
        'unitedkingdom', 'europe', 'britishproblems', 'legaladviceuk',
        'the3million', 'migrants', 'openrightsgroup', 'AskUK',
        'ukpolitics', 'worldnews', 'immigrationlaw'
    ]

    keywords = [
        'share code', 'digital immigration', 'online immigration status', 'evisa',
        'digital BRP', 'immigration app', 'BRP replacement', 'settled status',
        'EUSS', 'UK visa', 'spouse visa', 'student visa', 'tier 2 visa',
        'ILR', 'home office error', 'vfs delay', 'UKVI portal problem',
        'biometric delay', 'email from UKVI', 'right to work UK',
        'renting with share code', 'NHS and immigration', 'check immigration status',
        'immigration bill', 'UK immigration law', 'european settlement scheme',
        'rwanda policy'
    ]

    scraped_posts = []
    for sub_name in subreddits_to_scrape:
        print(f"Scraping subreddit: r/{sub_name}")
        subreddit = reddit.subreddit(sub_name)
        for post in subreddit.new(limit=1000):
            post_text = post.title + " " + post.selftext
            if any(keyword.lower() in post_text.lower() for keyword in keywords):
                scraped_posts.append({
                    'id': post.id,
                    'title': post.title,
                    'selftext': post.selftext,
                    'created_utc': post.created_utc,
                    'author': str(post.author),
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'subreddit': str(post.subreddit),
                    'url': post.url,
                    'keyword_matched': [kw for kw in keywords if kw.lower() in post_text.lower()]
                })
        time.sleep(2) # Respect API rate limits

    df = pd.DataFrame(scraped_posts)
    output_path = 'data/raw/reddit_scraped_posts.csv'
    df.to_csv(output_path, index=False)
    print(f"--- Reddit scraping complete. Saved {len(df)} posts to {output_path} ---")
    return df

def scrape_legislation_data():
    """
    Scrapes UK legislation and guidance from government websites.
    """
    print("--- Starting Legislation Data Scraping ---")
    
    # Part 1: GOV.UK API for guidance and policy
    gov_uk_results = []
    search_keywords = ["eVisa", "immigration act 2014", "biometric residence permit", "right to rent"]
    for keyword in search_keywords:
        url = f"https://www.gov.uk/api/search.json?q={keyword}&count=100"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            for item in data.get('results', []):
                gov_uk_results.append({
                    'title': item.get('title'),
                    'link': "https://www.gov.uk" + item.get('link'),
                    'summary': item.get('description'),
                    'date': item.get('public_timestamp', '').split('T')[0],
                    'source': 'GOV.UK'
                })
        except requests.exceptions.RequestException as e:
            print(f"Error fetching from GOV.UK for keyword '{keyword}': {e}")
        time.sleep(1)

    # Part 2: legislation.gov.uk for formal acts
    legislation_gov_uk_results = []
    base_url = "https://www.legislation.gov.uk/ukpga/2014/22/contents" # Example: Immigration Act 2014
    try:
        response = requests.get(base_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.find('h1', class_='title').text.strip()
        legislation_gov_uk_results.append({
            'title': title,
            'link': base_url,
            'summary': "Primary UK legislation concerning immigration.",
            'date': '2014-05-14', # Manually added for this example
            'source': 'legislation.gov.uk'
        })
    except requests.exceptions.RequestException as e:
        print(f"Error fetching from legislation.gov.uk: {e}")

    df_gov = pd.DataFrame(gov_uk_results)
    df_leg = pd.DataFrame(legislation_gov_uk_results)
    df_combined = pd.concat([df_gov, df_leg]).drop_duplicates(subset=['title']).reset_index(drop=True)
    
    output_path = 'data/raw/uk_legislation.csv'
    df_combined.to_csv(output_path, index=False)
    print(f"--- Legislation scraping complete. Saved {len(df_combined)} documents to {output_path} ---")
    return df_combined

if __name__ == '__main__':
    scrape_reddit_data()
    scrape_legislation_data()