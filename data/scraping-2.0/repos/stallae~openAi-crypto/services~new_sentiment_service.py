import requests
import openai
from datetime import datetime
from models import HistoricalPrices, Trends


class NewsSentimentService:
    def __init__(self):
        openai.api_key = "YOUR_OPENAI_API_KEY"
        self.crypto_keywords = {
            'bitcoin': ['bitcoin', 'btc'],
            'ethereum': ['ethereum', 'eth'],
            # Adicione outras criptomoedas e suas palavras-chave aqui
        }

    def analyze_news_sentiment(self, db):
        # Coleta as últimas notícias
        news_data = self.fetch_latest_news()

        # Calcula o sentimento das notícias e obtém a criptomoeda relacionada
        for article in news_data:
            title = article["title"]
            description = article["description"]
            published_at = datetime.fromisoformat(article["publishedAt"][:-1])
            url = article["url"]
            cryptocurrency = self.get_related_cryptocurrency(title, description)

            if cryptocurrency:
                sentiment_score = self.analyze_sentiment(title)
                trend = self.calculate_trend(cryptocurrency, sentiment_score)

                # Insere os dados na tabela Trends
                trends = Trends(timestamp=published_at, trend=trend, sentiment_score=sentiment_score)
                db.session.add(trends)

                # Obtem o preço da criptomoeda no horário da consulta
                price = self.fetch_crypto_price(cryptocurrency,published_at)

                # Insere os dados na tabela HistoricalPrices
                historical_prices = HistoricalPrices(cryptocurrency=cryptocurrency, date=published_at, open=price,
                                                     high=price, low=price, close=price, volume=0)
                db.session.add(historical_prices)

        db.session.commit()

    def fetch_latest_news(self):
        # Implementa a lógica para buscar as últimas notícias no banco de dados
        pass

    def analyze_sentiment(self, text):
        # Utiliza o GPT-3 para analisar o sentimento do texto
        prompt = "Please analyze the sentiment of this text: {}".format(text)
        response = openai.Completion.create(engine="text-davinci-002", prompt=prompt, max_tokens=1)
        sentiment_score = response.choices[0].text

        return sentiment_score

    def get_related_cryptocurrency(self, title, description):
        text = f"{title} {description}".lower()

        for crypto, keywords in self.crypto_keywords.items():
            if any(keyword.lower() in text for keyword in keywords):
                return crypto

        return None

    def calculate_trend(self, cryptocurrency, sentiment_score):
        # Implementa a lógica para calcular a tendência com base no sentimento da notícia e nos dados históricos de preços
        pass

    def fetch_crypto_price(self, cryptocurrency,published_at):
        # Utiliza uma API de preços de criptomoedas para obter o preço no momento da consulta
        pass
