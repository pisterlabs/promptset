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
    stocks = db.Table('stocks', metadata,
                        db.Column('id', db.Integer(), primary_key=True),
                        db.Column('symbol', db.String(255), nullable=True),
                        db.Column('name', db.String(255), nullable=True),
                        db.Column('sector', db.String(255), nullable=True),
                        db.Column('industry', db.String(255), nullable=True),
                        db.Column('live_price', db.Float(), nullable=True),
                        db.Column('prev_close', db.Float(), nullable=True),
                        db.Column('open', db.Float(), nullable=True),
                        db.Column("volume", db.Integer(), nullable=True),
                    )
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
    return engine, connection, stocks, news

if __name__ == "__main__":
    engine, connection, stocks, news = create_db()

    connection.execute(text("PRAGMA journal_mode=WAL"))

    # get list of stocks:
    stock_df = pd.read_excel('src/atradebot/SP_500_Companies.xlsx')
    symbols = stock_df['Symbol'].tolist()

    ib = IB()
    ib.connect('127.0.0.1', 7496, clientId=1)
    news_providers = ib.reqNewsProviders()
    codes = '+'.join(news_provider.code for news_provider in news_providers)

    # get data for symbols:
    for i in symbols[:5]:
        trans = connection.begin_nested()
        try:
            # get stock info
            try:
                ticker = yf.Ticker(i)
                info = ticker.info

                stock_values = {
                    "symbol": i,
                    'name': info.get('shortName', 'NA'),
                    'sector': info.get('sector', 'NA'),
                    'industry': info.get('industry', 'NA'),
                    'live_price': info.get('regularMarketPrice', 0.0),
                    'prev_close': info.get('previousClose', 0.0),
                    'open': info.get('open', 0.0),
                    'volume': info.get('volume', 0)
                }

                # Insert the stock into the database
                query_stock = db.insert(stocks)
                ResultProxy = connection.execute(query_stock, [stock_values])

            # Error handling for stock info fetching
            except requests.exceptions.HTTPError as error_terminal:
                print("HTTPError while fetching stock info for symbol:", i)
            except Exception as e:  # General exception catch
                print("Unexpected error while fetching stock info for symbol:", i)

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

    openai.api_key = 'sk-'
    

    def user_req():
        num_stocks = 3
        return num_stocks

    stock_df = pd.read_excel('src/atradebot/SP_500_Companies.xlsx')
    symbols = stock_df['Symbol'].tolist()
    stock_df['sentiment_counter'] = 0

    i = 0
    for result in ResultSet:
        print(result[1])
        print(result[5])
        print(type(result[1]))
        print(type(result[5]))
        text_to_analyze = result[2] + ' ' + result[6]
        messages = [
            {"role": "user", "content": f"Analyze the sentiment of the following text and provide a one-word sentiment: '{text_to_analyze}'. The answer should just be 1 word."}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        # Extract the response content
        sentiment = response['choices'][0]['message']['content']
        sentiment = sentiment.lower()
        print(sentiment)

        if (sentiment == "positive"):
            sentiment_num = 1
        elif (sentiment == "negative"):
            sentiment_num = -1
        elif (sentiment == "neutral"):
            sentiment_num = 0
        else:
            sentiment_num = 0
        
        sent_count_index_x = symbols.index(result[1])
        stock_df.at[sent_count_index_x, 'sentiment_counter'] += sentiment_num
        
        i = i + 1
        if ( i >= 3):
            # adding 2 seconds time delay
            time.sleep(60)
            i = 0
    
    num_stocks = user_req()
    largest_values = stock_df['sentiment_counter'].nlargest(num_stocks)
    print(stock_df)

    # Convert to a DataFrame with index numbers and values
    result_df = pd.DataFrame({
        'Index': largest_values.index,
        'Value': largest_values.values
    })
    
    
    for i in list(result_df['Index']):
        print("Stock Name:", stock_df.loc[i, 'Symbol'])
