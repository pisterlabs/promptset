#Build as of 6.5.23
import nltk
import openai
import logging
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from typing import Any, Dict, List, Union

class ChatCompletionResponse:
    choices: List[Dict[str, Any]]

    def __init__(self, **kwargs):
        self.choices = kwargs.get('choices', [])

class SentimentAnalysis:
    def __init__(self, openai_api_key: str):
        nltk.download('vader_lexicon')
        self.vader = SentimentIntensityAnalyzer()
        openai.api_key = openai_api_key
        # Set up logging
        logging.basicConfig(filename='sentiment_analysis.log', level=logging.ERROR)

    def analyze_vader(self, text: str) -> float:
        try:
            scores = self.vader.polarity_scores(text)
            return scores['compound']
        except Exception as e:
            logging.error(f"Error analyzing text with Vader: {str(e)}")
            return 0  # Return neutral sentiment score in case of error

    def analyze_openai(self, text: str) -> str:
        try:
            response = openai.ChatCompletion.create(
                model="text-curie-0001", 
                messages=[
                    {"role": "system", "content": "This is a sentiment analysis task. Please categorize the following news article's sentiment as 'Good', 'Bad', or 'Unknown'."},
                    {"role": "user", "content": text}
                ]
            )
            if 'choices' in response and len(response.choices) > 0: 
                return response.choices[0].message['content'] 
            else:
                logging.error("OpenAI API returned no choices or the list itself was empty.")
                return 'Unknown'
        except Exception as e:
            logging.error(f"Error analyzing text with OpenAI: {str(e)}")
            return 'Unknown'  # Return 'Unknown' sentiment in case of error
