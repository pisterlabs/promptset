import openai
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)

class SentimentAnalyzer:
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key

    def analyze_sentiment(self, text):
        if not text:
            logging.error("No text provided for sentiment analysis")
            return "Error: No text"

        openai.api_key = self.openai_api_key
        try:
            response = openai.Completion.create(
                engine="gpt-3.5-turbo-0613",  # Updated model
                prompt=f"Analyze the sentiment of the following text:\n\n\"{text}\"\n\nSentiment:",
                max_tokens=1,
                n=1,
                stop=None,
                temperature=0.7,
            )
            sentiment = response.choices[0].text.strip()
            return sentiment
        except Exception as e:
            logging.error(f"Error in sentiment analysis: {e}")
            return "Error: Exception"

# Example usage (if needed for testing or demonstration)
if __name__ == "__main__":
    analyzer = SentimentAnalyzer("your-openai-api-key")
    sentiment = analyzer.analyze_sentiment("This is a test text.")
    print(f"Sentiment: {sentiment}")
