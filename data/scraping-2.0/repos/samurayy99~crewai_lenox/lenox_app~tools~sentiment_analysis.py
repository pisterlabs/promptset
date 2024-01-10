import os
import praw
import tweepy
import json
from nltk.sentiment import SentimentIntensityAnalyzer
from langchain.tools import tool

class RedditSentimentAnalysis:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT')
        )
        self.sia = SentimentIntensityAnalyzer()

    @tool("Analyze Reddit Sentiment")
    def analyze_reddit_sentiment(self, keywords, subreddits, limit=100):
        try:
            posts = []
            for subreddit in subreddits:
                for post in self.reddit.subreddit(subreddit).hot(limit=limit):
                    if any(keyword.lower() in post.title.lower() for keyword in keywords):
                        sentiment_score = self.sia.polarity_scores(post.title)
                        posts.append({
                            'title': post.title,
                            'sentiment': sentiment_score['compound'],
                            'url': post.url
                        })
            return {'status': 'success', 'data': posts}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

class TwitterSentimentAnalysis:
    def __init__(self):
        auth = tweepy.OAuthHandler(os.getenv('TWITTER_CONSUMER_KEY'), os.getenv('TWITTER_CONSUMER_SECRET'))
        auth.set_access_token(os.getenv('TWITTER_ACCESS_TOKEN'), os.getenv('TWITTER_ACCESS_TOKEN_SECRET'))
        self.api = tweepy.API(auth, wait_on_rate_limit=True)
        self.sia = SentimentIntensityAnalyzer()

    @tool("Analyze Twitter Sentiment")
    def analyze_twitter_sentiment(self, keywords, limit=100):
        try:
            tweets = []
            for keyword in keywords:
                for tweet in tweepy.Cursor(self.api.search, q=keyword, tweet_mode='extended', lang='en').items(limit):
                    sentiment_score = self.sia.polarity_scores(tweet.full_text)
                    tweets.append({
                        'text': tweet.full_text,
                        'sentiment': sentiment_score['compound'],
                        'url': f"https://twitter.com/user/status/{tweet.id}"
                    })
            return {'status': 'success', 'data': tweets}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
