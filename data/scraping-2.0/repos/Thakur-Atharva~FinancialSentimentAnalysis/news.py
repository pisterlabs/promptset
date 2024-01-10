import os
import requests
import sqlalchemy as db
import yfinance as yf
import pandas as pd
from sqlalchemy import text
import time
from ib_insync import *
# importing module
import time
import openai
import pandas as pd

def create_db():
    # create database:
    engine = db.create_engine('sqlite:///atradebot.db', echo=True)
    connection = engine.connect()
    metadata = db.MetaData()

    # create a single table:
    news = db.Table('news', metadata,
                        db.Column('id', db.Integer(), primary_key=True),
                        db.Column('symbol', db.String(255), nullable=True),
                        db.Column('title', db.String(255), nullable=True),
                        db.Column('news_date', db.String(255), nullable=True),
                        db.Column('url', db.String(255), nullable=True),
                        db.Column('source', db.String(255), nullable=True),
                        db.Column('text', db.String(255), nullable=True),
                    )
                    
    # create table in database:
    metadata.create_all(engine)
    return engine, connection, news

if __name__ == "__main__":
    engine, connection,  news = create_db()

    connection.execute(text("PRAGMA journal_mode=WAL"))

    # get list of stocks:
    stock_df = pd.read_excel('SP_500_Companies.xlsx')
    symbols = stock_df['Symbol'].tolist()

    #connecting to ib_insync
    ib = IB()
    ib.connect('127.0.0.1', 7496, clientId=1)
    news_providers = ib.reqNewsProviders()
    codes = '+'.join(news_provider.code for news_provider in news_providers)

    # get data for symbols:
    for i in symbols[:5]:
        trans = connection.begin_nested()
        try:
            # Fetch and store news articles
            try:
                stock = Stock(i, 'SMART', 'USD')
                ib.qualifyContracts(stock)
                headlines = ib.reqHistoricalNews(stock.conId, codes, '', '', 100)

                for headline in headlines:
                    article_date = headline.time.date()
                    article = ib.reqNewsArticle(headline.providerCode, headline.articleId)

                    # Insert the article into the database
                    news_info = {
                        'symbol': i,
                        'title': '',  # Title not needed
                        'news_date': str(article_date),
                        'url': '',  # URL not provided
                        'source': '',  # Source not provided
                        'text': article.articleText
                    }

                    # Insert the news into the database
                    query_news = db.insert(news)
                    ResultProxy = connection.execute(query_news, [news_info])

            # Error handling for news fetching
            except requests.exceptions.HTTPError as error_terminal:
                print("HTTPError while fetching news for symbol:", i)
            except Exception as e:  # General exception catch
                print("Unexpected error while fetching news for symbol:", i)


            trans.commit()
            time.sleep(1)

        # Error handling
        except Exception as e:  # General exception catch
            print("Unexpected error:", e)
            trans.rollback()

    # Fetch and print the first 5 stocks from the database after processing
    #query = db.select([data]).limit(5)
    query = news.select().limit(5)
    ResultProxy = connection.execute(query)
    ResultSet = ResultProxy.fetchall()
    i = 0
    for result in ResultSet:
        print(f"\n {i}")
        print(result)
        i = i+1
    connection.execute(text("PRAGMA journal_mode=WAL"))

    # Close the connection
    connection.close()