import os
import requests
import sqlalchemy as db
import yfinance as yf
from sqlalchemy import text
import time
from ib_insync import *
# importing module
import time
import openai
import pandas as pd
import datetime


def create_db():
    # create database:
    engine = db.create_engine('sqlite:///atradebot.db', echo=True)
    connection = engine.connect()
    metadata = db.MetaData()

    # Update the "stocks" table with the desired attributes
    stocks = db.Table('stocks', metadata,
        db.Column('id', db.Integer(), primary_key=True), # Primary key
        db.Column('symbol', db.String(255), nullable=False),  # Symbol is marked as non-nullable
        db.Column('date', db.Date(), nullable=False),  
        db.Column('close', db.Float(), nullable=True),
        db.Column('volume', db.Integer(), nullable=True),
        db.Column('open', db.Float(), nullable=True),  
        db.Column('high', db.Float(), nullable=True),
        db.Column('low', db.Float(), nullable=True)
    )


    # create table in database:
    metadata.create_all(engine)
    return engine, connection, stocks

if __name__ == "__main__":
    import os

    # Create a new database
    engine, connection, stocks = create_db()

    connection.execute(text("PRAGMA journal_mode=WAL"))

    # get list of stocks:
    stock_df = pd.read_csv('sp-500-index-10-29-2023.csv')
    symbols = stock_df['Symbol'].tolist()

    # use YFinance to create a dataframe of all the stocks in the S&P 500
    # store each DF in Stocks table in the database
    today = datetime.date.today().strftime('%Y-%m-%d')


    # Define the start date for new data
    start_date = '2020-01-01'  # Adjust this date as needed

    for symbol in symbols:
        # Check if data for this symbol and start_date already exists in the database
        query = f"SELECT COUNT(*) FROM stocks WHERE symbol = '{symbol}' AND date >= '{start_date}'"
        result = connection.execute(text(query)).fetchone()


        if result[0] > 0:
            print(f"Data for {symbol} from {start_date} onwards already exists. Skipping.")
        else:
        # get stock info
            try:
                # Get historical data using yf.download()
                stock_info = yf.download(symbol, start=start_date)  # API call to create DataFrame
                stock_info['symbol'] = symbol # Add a column for the symbol
                stock_info['date'] = stock_info.index.strftime('%Y-%m-%d')  # Add a column for the date
                stock_info = stock_info[['symbol', 'date', 'Close', 'Volume', 'Open', 'High', 'Low']]  # Reorder the columns
                stock_info.columns = ['symbol', 'date', 'close', 'volume', 'open', 'high', 'low']  # Rename the columns

                # Insert the aggregated data into the database
                stock_info.to_sql('stocks', con=engine, if_exists='append', index=False)
                print(f"Added {symbol} to the database")
            except Exception as e:
                print(f"Could not add {symbol} to the database")
                print(e)
            time.sleep(1)
