```python
# SentimentAnalyzer.py

import requests
import json
import openai

class SentimentAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = self.api_key

    def analyze_sentiment(self, text):
        response = openai.Completion.create(
          engine="text-davinci-002",
          prompt=text,
          temperature=0.5,
          max_tokens=60
        )
        sentiment = self.get_sentiment(response.choices[0].text.strip())
        return sentiment

    def get_sentiment(self, response):
        if 'positive' in response:
            return 'Positive'
        elif 'negative' in response:
            return 'Negative'
        else:
            return 'Neutral'

    def batch_analyze(self, texts):
        sentiments = []
        for text in texts:
            sentiment = self.analyze_sentiment(text)
            sentiments.append(sentiment)
        return sentiments

if __name__ == "__main__":
    api_key = 'your-api-key'
    analyzer = SentimentAnalyzer(api_key)
    text = "I love this product!"
    print(analyzer.analyze_sentiment(text))
```
