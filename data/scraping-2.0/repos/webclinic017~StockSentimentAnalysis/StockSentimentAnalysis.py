import os
import json
import time 
import random
import openai
import requests
import pandas as pd
import yfinance as yf

class StockSentimentAnalysis():
    """
    This class StockSentimentAnalysis() is used to perform sentiment analysis on different stocks by using API keys from openai. 
    The user has to provide stock tickers, start date and end date to get sentiment related data.
    """
    def __init__(self):
        """
        The __init__ constructor of StockSentimentAnalysis class initializes the following class attributes:
            apiKey: API key to access stock data.
            openai_apiKey: API key to access sentiment analysis data.
            stocks: List of stock tickers.
            start_date: Start date of the sentiment analysis.
            end_date: End date of the sentiment analysis.
            N: Number of stocks for sentiment analysis.
            sentiment_df: Dataframe with sentiment analysis data.
            sentiment_summary_df: Summary dataframe with sentiment analysis data.
            stock_roi_df: Dataframe with stock return on investment data.
        """
        
        self.apiKey = None
        self.openai_apiKey = None
        self.stocks = None
        self.start_date = None
        self.end_date = None
        self.N = None
        self.sentiment_df = None
        self.sentiment_summary_df = None
        self.stock_roi_df = None

    def get_company_news(self, stock_symbol,date,num_articles):
        """
        get_company_news(stock_symbol,date,num_articles)
            Returns the top N articles from a newsapi.org query for the given 
            stock symbol and date. 

            Args:
                stock_symbol (str): The stock symbol to use for the query.
                date (str): The date to use for the query (yyyy-mm-dd).
                num_articles (int): The number of articles to return. 

            Returns:
                A pandas DataFrame with the top N articles.

        """
        
        # Create the url
        url = ('http://newsapi.org/v2/everything?'
            'q=' + stock_symbol + '&'
            'from=' + date + '&'
            'sortBy=popularity&'
            'language=en&'
            'apiKey=' + self.apiKey)
        
        # Get the response
        response = requests.get(url)
        # Convert to json
        response_json = json.loads(response.text)
        # Get the articles
        articles = response_json['articles']

        # Create a dataframe with the keys
        df = pd.DataFrame.from_dict(articles)

        # Return the top N articles
        return df.head(num_articles)

    def sentiment(self,article):
        """
        Calculate sentiment of given article using OpenAI's text-davinci-003 model

        Parameters
        ----------
        article : string
            Text to be analyzed

        Returns
        -------
        sentiment : string
            Positive, neutral, or negative sentiment of article
        """

        openai.api_key = self.openai_apiKey

        instructions ="Decide whether the text sentiment is positive, neutral, or negative.\n\n"
        text = 'Text: \n' + article + ' \nSentiment: '
        prompt = instructions + text
        
        response = openai.Completion.create(
              model="text-davinci-003",
              prompt=prompt,
              temperature=0,
              max_tokens=60,
              top_p=1.0,
              frequency_penalty=0.5,
              presence_penalty=0.0
        )
        
        # Get the sentiment score
        sentiment = response['choices'][0]['text']
        return sentiment


    # Wrap the code in a function
    def get_stock_sentiment_dataset(self):
        """
        get_stock_sentiment_dataset(stocks, start_date, end_date)
            Returns a dataset of stock sentiment data. 

            Args:
                stocks (list): The list of stocks to use for the query.
                start_date (str): The start date to use for the query (yyyy-mm-dd).
                end_date (str): The end date to use for the query (yyyy-mm-dd).

            Returns:
                A pandas DataFrame with the stock sentiment data.

        """
        # Create empty dataframe
        df = pd.DataFrame()
        
        # Create date range
        date_range = pd.date_range(self.start_date, self.end_date)
        
        # Iterate through the stocks
        for stock in self.stocks:
            # Iterate through the date range
            for date in date_range:
                # Get the news articles
                articles = self.get_company_news(stock, date.strftime('%Y-%m-%d'), self.N)
                # Iterate through the articles
                for index, row in articles.iterrows():
                    # Get the sentiment
                    sentiment_score = self.sentiment(row['title'])
                    # Create a row
                    row = {'stock': stock, 'date': date, 'sentiment': sentiment_score}
                    # Add the row to the dataframe
                    df = df.append(row, ignore_index=True)
                    
                    n = random.randint(1, 2)
                    time.sleep(n)
                    
        # Return the dataframe
        return df


    def get_stock_price(self, stock_symbol, date, num_days):
        """
        get_stock_price(stock_symbol, date, num_days)
            Returns the stock prices for the given stock symbol and date range. 

            Args:
                stock_symbol (str): The stock symbol to use for the query.
                date (str): The date to use for the query (yyyy-mm-dd).
                num_days (int): The number of days to return. 

            Returns:
                A pandas DataFrame with the stock prices.

        """
        
        # Get the stock
        stock = yf.Ticker(stock_symbol)
        
        # Get the stock prices
        prices = stock.history(start=date, interval='1d')
        
        # Return the close prices
        return prices['Close'][1:num_days+1]

    def get_sentiment_summary(self):
        """
        get_sentiment_summary(df)
            Returns a summary of the sentiment data. 

            Args:
                df (DataFrame): The dataframe of sentiment data.

            Returns:
                A pandas DataFrame with the sentiment summary.

        """
        # Get the sentiment counts
        sentiment_counts = self.sentiment_df.groupby(['stock', 'date', 'sentiment']).size().reset_index(name='count')

        # Get the total sentiment counts
        total_sentiment_counts = sentiment_counts.groupby(['stock', 'date'])['count'].sum().reset_index(name='total')

        # Merge the dataframes
        sentiment_summary = pd.merge(sentiment_counts, total_sentiment_counts, on=['stock', 'date'])

        # Calculate the percentage
        sentiment_summary['percent'] = sentiment_summary['count'] / sentiment_summary['total']
        
        # Pivot the table
        sentiment_summary_pivot = sentiment_summary.pivot(index=['stock','date'], columns='sentiment', values='percent').reset_index().fillna(0)

        # Return the summary table
        return sentiment_summary_pivot

    def get_stock_roi(self, stock_symbol, date):
        """
        get_stock_roi(stock_symbol,date)
            Calculates the ROI of a stock over the next 3 days starting with
            the opening price of the stock on the next day. 

            Args:
                stock_symbol (str): The stock symbol to use for the query.
                date (str): The date to use for the query (yyyy-mm-dd).

            Returns:
                The ROI of the stock over the next 3 days.
        """
        
        # Get the stock prices for the next 3 days
        prices = self.get_stock_price(stock_symbol, date, 3)
        
        # Calculate the ROI
        roi = ((prices.iloc[2] - prices.iloc[0]) / prices.iloc[0]) * 100
        
        # Return the ROI
        return roi

    def get_final_dataframe(self):
        """
        get_final_dataframe()
            Generates the final dataframe with stock ROI data. 

        """
        # Create the dataframe
        df = pd.DataFrame()
        
        # Iterate through the stocks
        for stock in self.stocks:
            # Iterate through the dates
            for date in pd.date_range(self.start_date, self.end_date):
                # Get the ROI
                roi = self.get_stock_roi(stock, date.strftime('%Y-%m-%d'))
                # Create a row
                row = {'stock': stock, 'date': date, 'roi': roi}
                # Add the row to the dataframe
                df = df.append(row, ignore_index=True)
        
        # Merge the dataframes
        self.stock_roi_df = pd.merge(self.sentiment_summary_df, df, on=['stock', 'date'])
        
        # Return the final dataframe
        return self.stock_roi_df
