# src/sentiment_analyzer.py

import openai
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import requests
from .api_client import OpenAIApiClient

class SentimentAnalyzer:
    def __init__(self):
        self.api_client = OpenAIApiClient()
        self.sia = SentimentIntensityAnalyzer()

    def analyze_text(self, text):
        # Preprocess the text
        preprocessed_text = self.api_client.preprocess_text(text)

        # Get sentiment scores
        sentiment_scores = self.sia.polarity_scores(preprocessed_text)

        return sentiment_scores

    def analyze_url(self, url):
        # Fetch the content of the URL
        response = requests.get(url)
        content = response.text

        # Analyze the sentiment of the content
        sentiment_scores = self.analyze_text(content)

        return sentiment_scores
