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
    stock_df = pd.read_excel('SP_500_Companies.xlsx')
    symbols = stock_df['Symbol'].tolist()


    # get data for symbols:
    for i in symbols[:5]:
        trans = connection.begin_nested()
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



    # Fetch and print the first 5 stocks from the database after processing
    #query = db.select([data]).limit(5)
    query = stocks.select().limit(5)
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

  