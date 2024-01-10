#import nltk
import openai
import requests
import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from .models import ExpertOpinion


class GoldMarketAnalyzer:
    def __init__(self):
        self.api_key = '1LIUUC0LBOCFOJ9P'
        self.news_api_key = '74821d4399404e1c98b5daa4d4e8c0b4'
        self.openai_api_key = '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'
        openai.api_key = self.openai_api_key

    def get_gold_price(self):
        symbol = 'XAUUSD'
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={self.api_key}'

        try:
            response = requests.get(url)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
            data = response.json()
            latest_date = max(data['Time Series (Daily)'].keys())
            latest_price = float(data['Time Series (Daily)'][latest_date]['4. close'])
            return latest_price
        except requests.RequestException as e:
            print(f"Error fetching gold prices: {e}")
            return None

    def analyze_price_movement(self, current_price, short_term_sma, long_term_sma, macd, signal_line):
        if short_term_sma > long_term_sma and macd > signal_line:
            return "↑ Upward (Potential Buy)"
        elif short_term_sma < long_term_sma and macd < signal_line:
            return "↓ Downward (Potential Sell)"
        else:
            return "→ No clear trend"

    def calculate_sma(self, prices, window):
        return prices.rolling(window=window).mean()

    def calculate_macd(self, prices, short_window, long_window, signal_window):
        short_ema = prices.iloc[:, 0].ewm(span=short_window, adjust=False).mean()
        long_ema = prices.iloc[:, 0].ewm(span=long_window, adjust=False).mean()
        prices['MACD'] = short_ema - long_ema
        prices['Signal_Line'] = prices['MACD'].ewm(span=signal_window, adjust=False).mean()
        return prices

    def simulate_next_week_price_trend(self, current_price, weekly_price_change, historical_prices):
        trend_slope = (historical_prices.iloc[-1] - historical_prices.iloc[0]) / len(historical_prices)
        next_week_price = current_price + trend_slope + weekly_price_change
        return next_week_price

    def expert_input(self):
        if __name__ == "__main__":
            # If running as a script, take input from the console
            user_input = input(
                "Enter the expert's opinion on the movement of gold prices (choose 'Upward', 'Downward', or 'No Change'): "
            )
        else:
            # If running through Django, set a default or retrieve from another source
            user_input = "No Change"  # Set a default value or retrieve from another source

        user_input_lower = user_input.lower()
        if 'upward' in user_input_lower:
            return "↑ Upward"
        elif 'downward' in user_input_lower:
            return "↓ Downward"
        elif 'no change' in user_input_lower:
            return "→ No Change"
        else:
            return "→ No Change"

    def get_gold_news(self):
        news_url = f'https://newsapi.org/v2/everything?q=gold&apiKey={self.news_api_key}'
        response = requests.get(news_url)
        news_data = response.json()
        return news_data['articles']

    def analyze_news_sentiment(self, articles):
        sid = SentimentIntensityAnalyzer()
        sentiments = [sid.polarity_scores(article['title'])['compound'] for article in articles]
        avg_sentiment = sum(sentiments) / len(sentiments)
        if avg_sentiment >= 0.05:
            return "Positive News Sentiment"
        elif avg_sentiment <= -0.05:
            return "Negative News Sentiment"
        else:
            return "Neutral News Sentiment"

    # Download the sentiment analysis lexicon
    #nltk.download('vader_lexicon')

    def gpt3_assist_analysis(self, news_sentiment, expert_opinion, overall_direction, historical_prices):
        # Construct the prompt and include historical prices in the conversation
        prompt_text = "Based on the received news data, provide me with a forecast for the next week price movement of the XAUUSD pair, indicating either 'price growth' or 'price down.' The response to 'price growth' or 'price down' occurs by comparing the current quote with the expected value for tomorrow, with a deviation of $10 USD per 1 troy ounce of XAUUSD. In case of no change or if the mentioned deviation is less than $10 USD per 1 troy ounce of XAUUSD, specify 'Unchanged.'"

        # Include historical prices in the conversation
        conversation = [
            {"role": "system", "content": "You are a financial analyst providing insights on gold prices."},
            {"role": "user", "content": prompt_text},
            {"role": "user", "content": f"Historical Prices: {', '.join(map(str, historical_prices))}"},
            {"role": "user", "content": f"News Sentiment: {news_sentiment}"},
            {"role": "user", "content": f"Expert Opinion: {expert_opinion}"},
            {"role": "user", "content": f"Overall Direction: {overall_direction}"}
        ]

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",  # Update to the desired model
                messages=conversation
            )
            insight = response['choices'][0]['message']['content']
            return insight
        except Exception as e:
            print(f"Error in GPT-3 API request: {e}")
            return "GPT-3 insight unavailable."

    def get_historical_prices(self):
        # Replace this with your actual logic to fetch historical prices
        current_gold_price = self.get_gold_price()
        historical_prices = np.random.uniform(low=current_gold_price - 5, high=current_gold_price + 5, size=100)
        return historical_prices

    def run_analysis(self):
        # Retrieve the latest expert opinion
        latest_expert_opinion = ExpertOpinion.objects.last()

        expert_opinion_text = latest_expert_opinion.opinion if latest_expert_opinion else "No expert opinion available"

        gold_news = self.get_gold_news()
        news_sentiment = self.analyze_news_sentiment(gold_news)
        current_gold_price = self.get_gold_price()
        historical_prices = np.random.uniform(low=current_gold_price - 5, high=current_gold_price + 5, size=100)
        historical_prices = pd.Series(historical_prices)
        prices_df = pd.DataFrame({'Price': historical_prices})
        short_term_sma = self.calculate_sma(historical_prices, window=10)
        long_term_sma = self.calculate_sma(historical_prices, window=50)
        macd_window = 12
        signal_window = 9
        prices_df = self.calculate_macd(prices_df, short_window=macd_window, long_window=26, signal_window=signal_window)
        weekly_short_term_sma = self.calculate_sma(historical_prices, window=5)
        weekly_long_term_sma = self.calculate_sma(historical_prices, window=25)
        expert_opinion = self.expert_input()
        prediction = self.analyze_price_movement(
            current_gold_price, short_term_sma.iloc[-1], long_term_sma.iloc[-1],
            prices_df['MACD'].iloc[-1], prices_df['Signal_Line'].iloc[-1]
        )
        next_day_price = current_gold_price + np.random.uniform(-2, 2)
        next_day_prediction = self.analyze_price_movement(
            next_day_price, short_term_sma.iloc[-1], long_term_sma.iloc[-1],
            prices_df['MACD'].iloc[-1], prices_df['Signal_Line'].iloc[-1]
        )
        weekly_price_change = weekly_short_term_sma.iloc[-1] - weekly_long_term_sma.iloc[-1]
        next_week_price_trend = self.simulate_next_week_price_trend(current_gold_price, weekly_price_change, historical_prices)
        next_week_prediction_trend = self.analyze_price_movement(
            next_week_price_trend, weekly_short_term_sma.iloc[-1], weekly_long_term_sma.iloc[-1],
            prices_df['MACD'].iloc[-1], prices_df['Signal_Line'].iloc[-1]
        )
        weight_expert = 0.5
        weight_news = 0.2
        weight_sma = 0.3
        weighted_conclusion = (
            weight_expert * (1 if expert_opinion == "↑ Upward" else (-1 if expert_opinion == "↓ Downward" else 0)) +
            weight_news * (1 if news_sentiment == "Positive News Sentiment" else (-1 if news_sentiment == "Negative News Sentiment" else 0)) +
            weight_sma * (1 if prediction.startswith("↑") else (-1 if prediction.startswith("↓") else 0))
        )
        overall_direction = "↑ Upward" if weighted_conclusion > 0 else "↓ Downward" if weighted_conclusion < 0 else "→ No Change"
        historical_prices = self.get_historical_prices()
        gpt_response = self.gpt3_assist_analysis(news_sentiment, expert_opinion, overall_direction, historical_prices)

        # You can return or print the results as needed
        return {
            'current_gold_price': current_gold_price,
            'prediction': prediction,
            'next_day_prediction': next_day_prediction,
            'next_week_prediction_trend': next_week_prediction_trend,
            'overall_direction': overall_direction,
            'gpt_response': gpt_response,
            'expert_opinion': expert_opinion_text,
            'short_term_sma': short_term_sma.iloc[-1],
            'long_term_sma': long_term_sma.iloc[-1],
            'macd': prices_df['MACD'].iloc[-1],
            'signal_line': prices_df['Signal_Line'].iloc[-1],
            'weekly_short_term_sma': weekly_short_term_sma.iloc[-1],
            'weekly_long_term_sma': weekly_long_term_sma.iloc[-1],
            'next_week_prediction_price': next_week_price_trend,
            'weighted_conclusion': weighted_conclusion,
        }


# Example usage
if __name__ == "__main__":
    analyzer = GoldMarketAnalyzer()
    analysis_results = analyzer.run_analysis()
    print(analysis_results)


