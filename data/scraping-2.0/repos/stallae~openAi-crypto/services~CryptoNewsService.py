import requests
import openai
from datetime import datetime, timedelta
from app.models import News, HistoricalPrices, Trends


class CryptoNewsService:
    def __init__(self):
        self.api_key = "YOUR_GOOGLE_NEWS_API_KEY"
        self.base_url = "https://newsapi.org/v2"
        self.crypto_keywords = {
            'bitcoin': ['bitcoin', 'btc'],
            'ethereum': ['ethereum', 'eth'],
            # Adicione outras criptomoedas e suas palavras-chave aqui
        }

        self.openai_api_key = "YOUR_OPENAI_API_KEY"
        openai.api_key = self.openai_api_key

    def populate_database_if_empty(self, db):
        session = db.Session()
        news_count = session.query(News).count()
        if news_count == 0:
            news_data = self.fetch_news_from_last_hours(90)
            self.insert_news_into_database(news_data, db)

    def fetch_news_from_last_hours(self,hours):
        query = "cryptocurrency"
        to_date = datetime.utcnow()
        from_date = to_date - timedelta(hours=hours)
        from_date_str = from_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        to_date_str = to_date.strftime("%Y-%m-%dT%H:%M:%SZ")

        url = f"{self.base_url}/everything?q={query}&from={from_date_str}&to={to_date_str}&sortBy=popularity&apiKey={self.api_key}"
        response = requests.get(url)
        response_json = response.json()

        if response.status_code != 200:
            raise Exception(f"Google News API request failed with status code {response.status_code}")

        return response_json["articles"]

    def insert_news_into_database(self, news_data, db):
        session = db.Session()

        for article in news_data:
            title = article["title"]
            description = article["description"]
            published_at = article["publishedAt"]
            url = article["url"]
            cryptocurrency = self.get_related_cryptocurrency(title, description)

            if cryptocurrency:
                news = News(title=title, description=description, published_at=published_at, url=url, cryptocurrency=cryptocurrency)
                session.add(news)

        session.commit()

    def get_related_cryptocurrency(self, title, description):
        text = f"{title} {description}".lower()

        for crypto, keywords in self.crypto_keywords.items():
            if any(keyword.lower() in text for keyword in keywords):
                return crypto

        return None

    def fetch_latest_news(self, db, limit):
        session = db.Session()

        news = session.query(News).filter(News.analyzed == False).order_by(News.published_at.desc()).limit(limit).all()

        return news

    def analyze_sentiment(self, text):
        prompt = f"Please analyze the sentiment of this text: {text}"
        response = openai.Completion.create(engine="text-davinci-002", prompt=prompt, max_tokens=1)
        sentiment_score = response.choices[0].text

        return sentiment_score

    def fetch_unanalyzed_news(self, db):
        session = db.Session()
        unanalyzed_news = session.query(News).filter(News.analyzed == False).all()
        return unanalyzed_news
