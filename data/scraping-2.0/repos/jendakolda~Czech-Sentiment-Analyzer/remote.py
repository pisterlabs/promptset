import openai
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


class TranslatorSentimentAnalyzer:
    def __init__(self, openai_api_key):
        openai.api_key = openai_api_key
        nltk.download('vader_lexicon')
        self.sia = SentimentIntensityAnalyzer()

    @staticmethod
    def translate_to_english(text):
        response = openai.Completion.create(
            model="gpt-4",  # "gpt-4.0-turbo",
            prompt=f"Translate the following Czech text to English: \"{text}\"",
            max_tokens=150
        )
        return response.choices[0].text.strip()

    def analyze_sentiment(self, text):
        return self.sia.polarity_scores(text)

    def evaluate_and_report(self, text):
        # Translate the text
        translated_text = self.translate_to_english(text)

        # Get sentiment scores
        sentiment_scores = self.analyze_sentiment(translated_text)

        # Construct a meaningful report
        report = {
            "translated_text": translated_text,
            "sentiment_analysis": {
                "positive": f"{sentiment_scores['pos'] * 100:.2f}%",
                "negative": f"{sentiment_scores['neg'] * 100:.2f}%",
                "neutral": f"{sentiment_scores['neu'] * 100:.2f}%",
                "overall_sentiment": "Positive" if sentiment_scores['compound'] > 0.05 else (
                    "Negative" if sentiment_scores['compound'] < -0.05 else "Neutral")
            }
        }

        return report


if __name__ == "__main__":
    # Read API key from api_key.txt
    with open('api_key.txt', 'r') as f:
        api_key = f.readline().strip()

    analyzer = TranslatorSentimentAnalyzer(api_key)

    czech_text = input("Enter the Czech text to analyze: ").strip()
    result = analyzer.evaluate_and_report(czech_text)

    print("\nTranslated Text:")
    print(result["translated_text"])

    print("\nSentiment Analysis:")
    for key, value in result["sentiment_analysis"].items():
        print(f"{key.capitalize()}: {value}")

