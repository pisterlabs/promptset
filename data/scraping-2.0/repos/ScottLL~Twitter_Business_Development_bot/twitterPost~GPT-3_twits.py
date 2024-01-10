import os
import requests
import random
import tweepy
from PIL import Image
from io import BytesIO
import time
import openai

CRYPTOPANIC_API_KEY = os.getenv('CRYPTOPANIC_API_KEY')

# Authenticate to Twitter
auth = tweepy.OAuthHandler(os.getenv('CONSUMER_KEY'), os.getenv('CONSUMER_SECRET'))
auth.set_access_token(os.getenv('TOKEN'), os.getenv('TOKEN_SECRET'))
api = tweepy.API(auth)

# Authenticate to OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

global_api_rate_delay = .2  # All API methods are rate limited per IP at 5req/sec.


def make_url(filter=None, kind=None, region=None, page=None):
    """Handle of URL variables for API POST."""
    url = 'https://cryptopanic.com/api/v1/posts/?auth_token={}'.format(CRYPTOPANIC_API_KEY)

    if kind is not None and kind in ['news', 'media']:
        url += "&kind={}".format(kind)

    filters = ['rising', 'hot', 'bullish', 'bearish', 'important', 'saved', 'lol']
    if filter is not None and filter in filters:
        url += "&filter={}".format(filter)

    regions = ['en', 'de', 'es', 'fr', 'it', 'pt', 'ru']  # (English), (Deutsch), (Español), (Français), (Italiano), (Português), (Русский)--> Respectively
    if region is not None and region in regions:
        url += "&region={}".format(region)

    if page is not None:
        url += "&page={}".format(page)

    return url


def get_page_json(url=None):
    """
    Get First Page.

    Returns Json.

    """
    time.sleep(global_api_rate_delay)
    if not url:
        url = "https://cryptopanic.com/api/v1/posts/?auth_token={}".format(CRYPTOPANIC_API_KEY)
    page = requests.get(url)
    data = page.json()
    return data


def get_news():
    """Fetch news from CryptoPanic API."""
    # Get top headlines from CryptoPanic
    url = make_url(kind='news', filter='hot', region='en')
    data = get_page_json(url)
    articles = data['results']
    if not articles:
        return None, None, None
    selected_article = random.choice(articles)

    # Get the summary, original URL and image of the selected article
    summary = selected_article['title']
    metadata = selected_article.get('metadata', None)
    if metadata is None:
        news_url = selected_article['url']
    else:
        news_url = metadata.get('original_url', selected_article['url'])
    image_url = selected_article.get('image', None)
    if image_url:
        # Download the image and convert to a PIL Image object
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
    else:
        img = None

    return summary, news_url, img

def generate_tweet():
    """Generate tweet text and get news."""
    # Get summary, URL and image from CryptoPanic API
    summary, news_url, img = get_news()
    if not summary:
        return None, None, None
    prompt = f"What's the latest news related to {summary.strip()} in the crypto world? give me a summary of the news in 150 characters or less, and add hashtags before the keywords at the begining of the sentense you generate. no space between sentense. no space between hashtags and sentense."
    message_log = [{"role": "user","content": prompt}]
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = message_log,
        max_tokens=300,
        stop=None,
        temperature=0.7,
    )
    for choice in response.choices:
        if "text" in choice:
            return choice.text
    
    tweet_text = response.choices[0].message.content
    return tweet_text, news_url, img

def post_tweet():
    """Post tweet with image and URL."""
    # Generate tweet text, news URL and image
    tweet_text, news_url, img = generate_tweet()

    # Post tweet
    if img is not None:
        # Save image locally
        img_path = "image.jpg"
        img.save(img_path)

        # Post tweet with image
        try:
            api.update_with_media(
                filename=img_path,
                status=f"{tweet_text[:230]} {news_url}"
            )
        except tweepy.TweepError as e:
            print(e)
            return

        # Remove image file
        os.remove(img_path)
    else:
        # Post tweet without image
        try:
            api.update_status(f"{tweet_text[:280 - len(news_url) - 1]} {news_url}")
        except tweepy.TweepError as e:
            print(e)
            return

import time

if __name__ == "__main__":
    post_tweet()
        # time.sleep(1800)  # wait for 30 minutes before posting again
