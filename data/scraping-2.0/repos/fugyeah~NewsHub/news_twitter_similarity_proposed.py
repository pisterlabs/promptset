import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import configparser
from typing import Dict, List, Tuple
import tweepy
import openai

# Config parser
config = configparser.ConfigParser()
config.read('config.ini')

# Twitter API config
API_KEY = config.get('TWITTER', 'API_KEY')
API_SECRET_KEY = config.get('TWITTER', 'API_SECRET_KEY')
ACCESS_TOKEN = config.get('TWITTER', 'ACCESS_TOKEN')
ACCESS_TOKEN_SECRET = config.get('TWITTER', 'ACCESS_TOKEN_SECRET')

# OpenAI API config
OPENAI_API_KEY = config.get('OPENAI', 'OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY


# Model config
MODEL_NAME = config.get('MODEL', 'MODEL_NAME')

# Thresholds
SIMILARITY_THRESHOLD = config.getfloat('TWITTER_THRESHOLDS', 'SIMILARITY_THRESHOLD')
TOP_N_ARTICLES = config.getint('TWITTER_THRESHOLDS', 'TOP_N_ARTICLES')


def load_data(filepath: str) -> List[Tuple]:
    with open(filepath, 'rb') as f:
        summaries = pickle.load(f)
    print(f"Loaded {len(summaries)} summaries from {filepath}")
    return summaries


def generate_embeddings(summaries: List[Tuple], model: SentenceTransformer) -> np.ndarray:
    corpus = [summary[0] + ' ' + summary[2][:500] for summary in summaries]
    embeddings = model.encode(corpus, convert_to_tensor=True)
    embeddings_np = embeddings.cpu().numpy()
    normalized_embeddings = embeddings_np / np.linalg.norm(embeddings_np, axis=1, keepdims=True)
    return normalized_embeddings


def generate_similarity_matrix(normalized_embeddings: np.ndarray) -> np.ndarray:
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
    np.fill_diagonal(similarity_matrix, -1)
    return similarity_matrix


def get_top_articles(similarity_matrix: np.ndarray, summaries: List[Tuple], threshold: float, top_n: int) -> List[Tuple]:
    row_indices, col_indices = np.where(similarity_matrix > threshold)
    if len(row_indices) == 0 or len(col_indices) == 0:
        raise Exception("No pair of articles have similarity above the threshold")
    
    indices = np.argsort(similarity_matrix[row_indices, col_indices])[::-1]
    top_indices = indices[:top_n]
    top_articles = [(summaries[row_indices[i]], summaries[col_indices[i]]) for i in top_indices]
    return top_articles


def generate_top_articles_by_category(top_articles: List[Tuple]) -> Dict[str, Tuple]:
    top_articles_by_category = {}
    for article1, _ in top_articles:
        _, category, _, _, _, _ = article1
        if category not in top_articles_by_category:
            top_articles_by_category[category] = article1
        if len(top_articles_by_category) >= 5:
            break
    return top_articles_by_category


def generate_engaging_tweet(headline: str, summary: str, url: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a professional news agent, you take news headlines and convert them to tweets to be published ASAP. Transform the following information into an engaging tweet and link to NewsPlanetAi.com: THE ENTIRE TWEET MUST BE LESS THAN 200 CHARACTERS"
        },
        {
            "role": "user",
            "content": f"Please summarize and turn this article into a tweet, that MUST be less than 200 characters long, including the hashtags:\nHeadline: {headline}\nSummary: {summary}\nURL: NewsPlanetAi.com"
        }
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0.8,
        max_tokens=60
    )
    tweet = response['choices'][0]['message']['content']
    if tweet.startswith('"'):
        tweet = tweet.strip('"')
    return tweet


def post_tweet(tweet: str):
    confirmation = input("Do you want to tweet this? (yes/no): ")
    if confirmation.lower() != "yes":
        print("Tweet not posted.")
        return
    client = tweepy.Client(consumer_key=API_KEY, consumer_secret=API_SECRET_KEY, access_token=ACCESS_TOKEN, access_token_secret=ACCESS_TOKEN_SECRET)
    client.create_tweet(text=tweet)
    print("Tweet posted successfully")


def main():
    # Load and preprocess data
    print("Loading and preprocessing data")
    summaries = load_data('cache/summaries.p')

    # Load model
    print("Loading model")
    model = SentenceTransformer(MODEL_NAME)

    # Generate embeddings
    print("Generating embeddings")
    normalized_embeddings = generate_embeddings(summaries, model)

    # Generate similarity matrix
    print("Generating similarity matrix")
    similarity_matrix = generate_similarity_matrix(normalized_embeddings)

    # Get top articles
    print("Getting top articles")
    top_articles = get_top_articles(similarity_matrix, summaries, SIMILARITY_THRESHOLD, TOP_N_ARTICLES)

    # Get top articles by category
    print("Getting top articles by category")
    top_articles_by_category = generate_top_articles_by_category(top_articles)

    # Print articles
    print("Printing articles")
    for idx, article in enumerate(top_articles_by_category.values()):
        headline, category, summary, url, _, _ = article
        print(f"Article {idx + 1}: {headline} ({url})\n")

    # Request article choice
    article_num = int(input("Enter the number of the article you want to choose: ")) - 1
    articles_list = list(top_articles_by_category.values())
    chosen_article = articles_list[article_num]

    # Generate tweet data
    headline, _, summary, url, _, _ = chosen_article
    tweet = generate_engaging_tweet(headline, summary, url)

    # Post tweet
    print(f"Prepared tweet: \n{tweet}")
    post_tweet(tweet)


if __name__ == "__main__":
    main()

