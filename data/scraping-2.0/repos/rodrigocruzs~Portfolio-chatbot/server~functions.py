import json
from langchain.sql_database import SQLDatabase
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool
from langchain.schema import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from pydantic import BaseModel, Field, Extra
import langchain
import yfinance as yf
from typing import Optional, Type, Any, Dict, Union
import os
import psycopg2
from dotenv import load_dotenv, find_dotenv
import json
from datetime import datetime, timedelta
from typing import List, Type
from urllib.parse import quote
import numpy as np
from sqlalchemy import create_engine, text, MetaData, Table
from datetime import date, datetime, timedelta
import holidays
from db import save_user_question
# from serpapi import GoogleSearch


load_dotenv(find_dotenv())

# Read environment variables
host = os.environ.get("DB_HOST")
dbname = os.environ.get("DB_NAME")
user = os.environ.get("DB_USER_LANGCHAIN")
password = os.environ.get("DB_PASSWORD_LANGCHAIN")
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Build the connection string
connection_string = f"postgresql+psycopg2://{user}:{password}@{host}:5432/{dbname}"
# Connect to the database
db = SQLDatabase.from_uri(
    connection_string,
    engine_args={
        "connect_args": {"sslmode": "require"},
    },
)

user_id = None
engine = None
engine = create_engine(connection_string, connect_args={"sslmode": "require"})

def init_db():
    global engine
    if engine is None:
        engine = create_engine(connection_string, connect_args={"sslmode": "require"})


def get_user_account_ids():
    """Get all account IDs associated with the currently logged in user."""
    global user_id
    if user_id is None:
        return {"error": "No user is logged in"}

    with engine.connect() as connection:
        result = connection.execute(
            """
            SELECT DISTINCT account_id
            FROM investment_view
            WHERE user_id = %s
            """, (user_id,)
        )
        account_ids = [row['account_id'] for row in result]
    return account_ids

def get_date_range(period: str):
    today = datetime.today()
    
    if period == "last year":
        start_date = today - timedelta(days=365)
    elif period == "last month":
        start_date = today - timedelta(days=30)
    elif period == "last week":
        start_date = today - timedelta(days=7)
    else:
        # Handle other periods or throw an error
        raise ValueError("Unknown period")
    
    end_date = today

    # Convert dates to string format
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    return start_date_str, end_date_str

# Base Model
class DateRangeInput(BaseModel):
    period: str = Field(
        ..., 
        description="The relative period for which to get the date range. Accepted values: 'last year', 'last month', 'last week'"
    )

# Base Tool
class DateRangeTool(BaseTool):
    name = "get_date_range"
    description = """
        Useful when you want to determine a specific date range based on a relative time period.
        It accepts phrases like 'last year', 'last month', and 'last week' and returns the start and end dates for the period.
        """
    args_schema: Type[BaseModel] = DateRangeInput

    def _run(self, period: str):
        try:
            start_date, end_date = get_date_range(period)
            return {"start_date": start_date, "end_date": end_date}
        except ValueError as e:
            return {"error": str(e)}

    def _arun(self, period: str):
        # Asynchronous method (if needed, otherwise raise NotImplementedError)
        raise NotImplementedError("This method does not support async")

def find_previous_business_day(date_str):
    date_format = "%Y-%m-%d"
    date = datetime.strptime(date_str, date_format)

    # Define the holidays for the USA stock market
    us_holidays = holidays.UnitedStates()

    # Adjust the date to find a business day (avoid weekends and public holidays)
    while date.weekday() >= 5 or date.strftime(date_format) in us_holidays:  # Saturday, Sunday, or public holiday
        date -= timedelta(days=1)
    
    return date.strftime(date_format)



# YF Tools


def get_current_stock_price(ticker=None, security_id=None):
    """Method to get current stock price"""
    # If a security_id is provided but not a ticker, retrieve the ticker from the database
    if security_id and not ticker:
        try:
            query = text("""
                SELECT ticker_symbol 
                FROM investment_view
                WHERE security_id = :security_id;
            """)
            params = {'security_id': security_id}
            
            with engine.connect() as connection:
                result = connection.execute(query, params).fetchone()
            
            if result:
                ticker = result[0]
                # print(f"Fetched ticker: {ticker}")
            else:
                return {"error": "No ticker found for the provided security ID"}
        except Exception as e:
            return {"error": str(e)}

    # Get the current stock price using the ticker symbol
    ticker_data = yf.Ticker(ticker)
    recent = ticker_data.history(period="1d")
    if recent.empty:
        return {"error": f"No data found for the ticker: {ticker}"}
    
    return {"price": recent.iloc[0]["Close"], "currency": ticker_data.info["currency"]}

class CurrentStockPriceInput(BaseModel):
    """Inputs for get_current_stock_price"""

    ticker: str = Field(description="Ticker symbol of the stock")

class CurrentStockPriceTool(BaseTool):
    name = "get_current_stock_price"
    description = """
        Useful when you want to get current stock price.
        You should enter the stock ticker symbol recognized by the yahoo finance
        """
    args_schema: Type[BaseModel] = CurrentStockPriceInput

    def _run(self, ticker: str):
        price_response = get_current_stock_price(ticker)
        return price_response

    def _arun(self, ticker: str):
        raise NotImplementedError("get_current_stock_price does not support async")


#Stock price on a specific date

def get_stock_price_on_date(ticker, date):
    """Method to get stock price on a certain date"""
    if not ticker:
        return {"error": "Ticker is not defined"}
    print(ticker)
    date = find_previous_business_day(date)
    ticker_data = yf.Ticker(ticker)
    date_obj = datetime.strptime(date, '%Y-%m-%d')
    
    end_date_obj = date_obj + timedelta(days=1)
    end_date_str = end_date_obj.strftime('%Y-%m-%d')
    
    history = ticker_data.history(start=date, end=end_date_str)
    
    if len(history) == 0:
        return {"error": "Data not available for the specified date"}
    
    return {"price": history.iloc[0]["Close"], "currency": ticker_data.info["currency"]}

class StockPriceOnDateInput(BaseModel):
    """Inputs for get_stock_price_on_date"""

    ticker: str = Field(description="Ticker symbol of the stock")
    date: str = Field(description="Specific date to get the stock price, in YYYY-MM-DD format")

class StockPriceOnDateTool(BaseTool):
    name = "get_stock_price_on_date"
    description = """
        Useful when you want to get the stock price on a certain date.
        You should enter the stock ticker symbol recognized by the Yahoo Finance,
        and the specific date in YYYY-MM-DD format.
        """
    
    args_schema: Type[BaseModel] = StockPriceOnDateInput
    
    def _run(self, ticker: str, date: str):
        price_response = get_stock_price_on_date(ticker, date)
        return price_response
    
    def _arun(self, ticker: str, date: str):
        raise NotImplementedError("get_stock_price_on_date does not support async")

#Tool to check stock performance

def get_stock_performance(ticker, days):
    """Method to get stock price change in percentage"""

    past_date = datetime.today() - timedelta(days=days)
    ticker_data = yf.Ticker(ticker)
    history = ticker_data.history(start=past_date)
    if not history.empty:
        old_price = history.iloc[0]["Close"]
    else:
        raise ValueError("The history DataFrame is empty.")
    current_price = history.iloc[-1]["Close"]
    return {"percent_change": ((current_price - old_price) / old_price) * 100}

class StockPercentChangeInput(BaseModel):
    """Inputs for get_stock_performance"""

    ticker: str = Field(description="Ticker symbol of the stock")
    days: int = Field(description="Timedelta days to get past date from current date")

class StockPerformanceTool(BaseTool):
    name = "get_stock_performance"
    description = """
        Useful when you want to check performance of the stock.
        You should enter the stock ticker symbol recognized by the yahoo finance.
        You should enter days as number of days from today from which performance needs to be check.
        output will be the change in the stock price represented as a percentage.
        """
    args_schema: Type[BaseModel] = StockPercentChangeInput

    def _run(self, ticker: str, days: int):
        response = get_stock_performance(ticker, days)
        return response

    def _arun(self, ticker: str):
        raise NotImplementedError("get_stock_performance does not support async")

# the best performing stock tool
def get_best_performing(stocks, days):
    best_stock = None
    best_performance = None
    for stock in stocks:
        try:
            performance_dict = get_stock_performance(stock, days)
            performance = performance_dict["percent_change"]
            if best_performance is None or performance > best_performance:
                best_stock = stock
                best_performance = performance
        except Exception as e:
            print(f"Could not calculate performance for {stock}: {e}")
    return best_stock, best_performance

class StockBestPerformingInput(BaseModel):
    """Input for Stock ticker check. for percentage check"""

    stocktickers: List[str] = Field(..., description="Ticker symbols for stocks or indices")
    days: int = Field(..., description="Int number of days to look back")

class StockGetBestPerformingTool(BaseTool):
    name = "get_best_performing"
    description = """Useful for when you want to know which stock had the best performance in a portfolio over a period. 
    You should input a list of stock tickers used on the yfinance API and also input the number of days to check the change over
    """

    def _run(self, stocktickers: List[str], days: int):
        price_change_response = get_best_performing(stocktickers, days)

        return price_change_response

    def _arun(self, stockticker: List[str], days: int):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = StockBestPerformingInput


def get_worst_performing(stocks, days):
    worst_stock = None
    worst_performance = None
    for stock in stocks:
        try:
            performance_dict = get_stock_performance(stock, days)
            performance = performance_dict["percent_change"]
            if worst_performance is None or performance < worst_performance:
                worst_stock = stock
                worst_performance = performance
        except Exception as e:
            print(f"Could not calculate performance for {stock}: {e}")
    return worst_stock, worst_performance

class StockWorstPerformingInput(BaseModel):
    """Input for Stock ticker check. for percentage check"""

    stocktickers: List[str] = Field(..., description="Ticker symbols for stocks or indices")
    days: int = Field(..., description="Int number of days to look back")

class StockGetWorstPerformingTool(BaseTool):
    name = "get_worst_performing"
    description = """Useful for when you want to know which stock had the worst performance in a portfolio over a period. 
    You should input a list of stock tickers used on the yfinance API and also input the number of days to check the change over
    """

    def _run(self, stocktickers: List[str], days: int):
        price_change_response = get_worst_performing(stocktickers, days)

        return price_change_response

    def _arun(self, stockticker: List[str], days: int):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = StockWorstPerformingInput


def get_stock_beta(stock_ticker, days):
    """Calculate the beta of a single stock based on historical data."""
    try:
        # Get historical data for the stock and the market (S&P 500)
        stock_data = yf.download(stock_ticker, period=f'{days}d')
        market_data = yf.download('^GSPC', period=f'{days}d')

        # Calculate the daily returns for the stock and the market
        stock_returns = stock_data['Close'].pct_change()[1:]
        market_returns = market_data['Close'].pct_change()[1:]

        # Calculate the beta using the formula
        covariance = np.cov(stock_returns, market_returns)[0, 1]
        variance = np.var(market_returns)
        beta = covariance / variance
        
        return beta
    except Exception as e:
        print(f"Could not calculate beta for {stock_ticker}: {e}")
        return None

def get_portfolio_beta(stocks, quantities, days):
    portfolio_beta = 0.0
    
    # Calculate the total quantity of stocks in the portfolio
    total_quantity = sum(quantities)
    
    # Calculate the weights of each stock in the portfolio
    weights = [quantity / total_quantity for quantity in quantities]
    
    for stock, weight in zip(stocks, weights):
        try:
            stock_beta = get_stock_beta(stock, days) # Here, you would calculate the beta for the individual stock
            portfolio_beta += stock_beta * weight
        except Exception as e:
            print(f"Could not calculate beta for {stock}: {e}")

    return portfolio_beta


class PortfolioBetaInput(BaseModel):
    """Input for get_portfolio_beta."""
    stocktickers: List[str] = Field(..., description="Ticker symbols for stocks or indices")
    quantities: List[int] = Field(..., description="The quantities of the respective stocks in the portfolio")
    days: int = Field(..., description="Int number of days to look back for beta calculation")

class GetPortfolioBetaTool(BaseTool):
    name = "get_portfolio_beta"
    description = """
    Useful for when you want to calculate the portfolio beta over a certain number of days
    to measure how risky a portfolio is. 
    It takes a list of stock tickers, their respective quantities in the portfolio, and the number of days to look back for the beta calculation.
    """

    def _run(self, stocktickers: List[str], quantities: List[int], days: int):
        return get_portfolio_beta(stocktickers, quantities, days)

    def _arun(self, stocktickers: List[str], quantities: List[int], days: int):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = PortfolioBetaInput

def get_stock_news(ticker, num_articles=5):
    """Method to get stock news"""
    
    ticker_data = yf.Ticker(ticker)
    news = ticker_data.news[:5]
    
    return {"news": news}

def get_latest_stock_news(tickers: List[str]):
    """Method to get the latest headlines for a list of tickers"""  
    
    news_response = {}

    # Determine the number of articles to fetch per ticker
    num_articles = 1 if len(tickers) > 1 else 5

    for ticker in tickers:
        try:
            news = get_stock_news(ticker, num_articles=num_articles)["news"]  # Use determined num_articles here
            headlines = []
            for i, article in enumerate(news):
                if i >= 2:  # Limit to 2 iterations
                    break
                
                print(article)  # Add this line to print the structure of the 'article' dictionary

                title = article.get('title', 'No title available')
                link = article.get('link', '#')
                
                headlines.append(
                    f"Title: {title}, "
                    f"<a href='{link}' target='_blank'>Link</a>"
                )
                
            news_response[ticker] = headlines
        except Exception as e:
            news_response[ticker] = str(e)

    return news_response


class LatestStockNewsInput(BaseModel):
    """Inputs for get_latest_stock_news"""
    
    tickers: List[str] = Field(
        ..., description="List of ticker symbols recognized by Yahoo Finance"
    )

class LatestStockNewsTool(BaseTool):
    name = "get_latest_stock_news"
    description = """
        Useful when you want to get the latest stock news for a list of tickers.
        Enter a list of stock ticker symbols recognized by Yahoo Finance.
        """
    args_schema: Type[BaseModel] = LatestStockNewsInput

    def _run(self, tickers: List[str]):
        news_response = get_latest_stock_news(tickers)
        return news_response

    def _arun(self, tickers: List[str]):
        raise NotImplementedError("get_latest_stock_news does not support async")

################################## SQL Tools ############################################## 



class BaseSQLDatabaseTool(BaseModel):
    """Base tool for interacting with a SQL database."""
    db: SQLDatabase = Field(exclude=True)
    
    class Config(BaseTool.Config):
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True
        extra = Extra.forbid


def get_total_investment_values(engine):
    global user_id
    try:
        if user_id is None:
            return {"error": "No user is logged in"}
        
        # Write your SQL query
        query = text("""
            SELECT SUM(institution_value) AS total_investment 
            FROM investment_view 
            WHERE user_id = :user_id AND type IN ('equity', 'mutual fund', 'etf');
        """)
        
        # Create a statement object to bind the parameters
        stmt = query.bindparams(user_id=user_id)
        
        # Execute the SQL query and fetch the result
        with engine.connect() as conn:
            result = conn.execute(stmt).fetchone()
        
        # Return the result
        return result[0] if result else None
    except Exception as e:
        # Log and re-raise the exception
        print(f"An error occurred: {e}")
        raise

class GetTotalInvestmentValuesInput(BaseModel):
    """Inputs for get_total_investment_values"""

class GetTotalInvestmentValuesTool(BaseTool):
    name = "get_total_investment_values"
    description = """
        Useful when you want to retrieve the total investment values from the 'investment_view' table in your database.
        This tool does not require any input as it is designed to specifically retrieve the total investment values by summing all the values in the 'institution_value' column.
        The output will be the total investment value calculated from the database.
        """
    args_schema: Type[BaseModel] = GetTotalInvestmentValuesInput

    def _run(self):
        global engine
        if engine is None:
            init_db()
        response = get_total_investment_values(engine)
        return response

    def _arun(self):
        raise NotImplementedError("get_total_investment_values does not support async")


def get_cash_position(engine):
    global user_id
    try:
        if user_id is None:
            return {"error": "No user is logged in"}
        # Write your SQL query to get the cash position from the database
        query = text("""
            SELECT SUM(institution_value) AS total_cash 
            FROM investment_view 
            WHERE user_id = :user_id AND type = 'cash';
        """)
        stmt = query.bindparams(user_id=user_id)
        
        # Execute the SQL query and fetch the result
        with engine.connect() as conn:
            result = conn.execute(stmt).fetchone()
        
        # Return the result
        return result[0] if result else None
    except Exception as e:
        # Log and re-raise the exception
        print(f"An error occurred: {e}")
        raise

class GetCashPositionInput(BaseModel):
    """Inputs for get_cash_position"""

class GetCashPositionTool(BaseTool):
    name = "get_cash_position"
    description = """
        Useful when you want to retrieve the total cash available in your account from the 'investment_view' table in your database.
        This tool does not require any input as it is designed to specifically retrieve the total cash by summing all the values in the 'institution_value' column.
        The output will be the total cash calculated from the database.
        """
    args_schema: Type[BaseModel] = GetCashPositionInput

    def _run(self):
        global engine
        if engine is None:
            init_db()  # Ensure to define a function to initialize your database connection
        response = get_cash_position(engine)
        if response is None:
            return "You have nothing allocated in cash at the moment."
        else:
            return f"Your current cash position is: ${response}"

    def _arun(self):
        raise NotImplementedError("get_cash_position does not support async")


def get_all_ticker_symbols(engine):
    global user_id 
    try:
        query_params = {"user_id": user_id}
        # Write your SQL query to get all ticker symbols
        query_text = """
            SELECT DISTINCT ticker_symbol AS stock_tickers 
            FROM investment_view 
            WHERE type IN ('equity', 'mutual fund', 'etf') AND user_id = :user_id;
        """

        query = text(query_text).bindparams(**query_params)       
        # Execute the SQL query and fetch the result
        with engine.connect() as conn:
            result = conn.execute(query).fetchall()
        if not result:
                return "You currently don't own any stocks"
        
        # Return the result - converting rows to a list of ticker symbols
        return [row[0] for row in result] if result else None
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

class GetAllTickerSymbolsInput(BaseModel):
    """Inputs for get_all_ticker_symbols"""

class GetAllTickerSymbolsTool(BaseTool):
    name = "get_all_ticker_symbols"
    description = """
        Useful when you want to retrieve all ticker symbols from the 'investment_view' table in your database. 
        This tool does not require any input as it is designed to specifically retrieve all the ticker symbols present in the 'ticker_symbol' column. 
        The output will be a list of all ticker symbols retrieved from the database (not unique).
        """
    args_schema: Type[BaseModel] = GetAllTickerSymbolsInput

    def _run(self):
        global engine
        if engine is None:
            init_db()
        response = get_all_ticker_symbols(engine)
        return response

    def _arun(self):
        raise NotImplementedError("get_all_ticker_symbols does not support async")


def get_stock_quantity(engine, ticker_symbols=None):    
    global user_id 
    try:
        if not user_id:
            return {"error": "No user is logged in"}
        
        query_params = {"user_id": user_id}
        if ticker_symbols is None:
            query_text = """
            SELECT ticker_symbol, quantity 
            FROM investment_view 
            WHERE user_id = :user_id;
            """
        else:
            query_text = """
            SELECT ticker_symbol, quantity 
            FROM investment_view 
            WHERE ticker_symbol IN :ticker_symbols AND user_id = :user_id;
            """
            query_params['ticker_symbols'] = tuple(ticker_symbols)

        query = text(query_text).bindparams(**query_params)
        
        with engine.connect() as conn:
            result = conn.execute(query).fetchall()
        
        if not result:
            if ticker_symbols:
                missing_stocks = ", ".join(ticker_symbols)
                return f"You currently don't own any {missing_stocks} stocks"
            else:
                return "You currently don't own any stocks"
        
        return [(row[0], row[1]) for row in result]
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

class GetStockQuantityInput(BaseModel):
    """Inputs for get_stock_quantity"""
    ticker_symbols: Optional[List[str]] = None  # Making ticker_symbols optional

class GetStockQuantityTool(BaseTool):
    name = "get_stock_quantity"
    description = """
        Useful when you want to retrieve the quantity of a specific stock from the 'investment_view' table in your database. 
        This tool receives a ticker symbol as input to retrieve the quantity of a specific stock. 
        The output will be the quantity retrieved from the database.
        """
    args_schema: Type[BaseModel] = GetStockQuantityInput

    def _run(self, *args, **kwargs):
        global engine
        if engine is None:
            init_db()

        # We'll extract the 'ticker_symbols' from the keyword arguments
        ticker_symbols = kwargs.get('ticker_symbols')
        
        response = get_stock_quantity(engine, ticker_symbols=ticker_symbols)

        return response
    
    def _arun(self):
        raise NotImplementedError("get_stock_quantity does not support async")


################################## Tools using investment_transaction ############################################## 

def get_transaction_summary(engine, transaction_types: List[str], start_date: Optional[str] = None, end_date: Optional[str] = None):
    global user_id
    try:
        if user_id is None:
            return {"error": "No user is logged in"}
        
        if start_date in ["last year", "last month", "last week"]:
            start_date, end_date = get_date_range(start_date)
        
        transaction_types_str = ", ".join([f"'{tt}'" for tt in transaction_types])
        query_text = f"""
            SELECT SUM(
                CASE 
                    WHEN amount < 0 THEN -amount
                    ELSE amount
                END
            ) AS total 
            FROM investment_transaction 
            WHERE user_id = :user_id AND subtype IN ({transaction_types_str})
        """        
        params = {'user_id': user_id}
        if start_date:
            query_text += " AND date >= :start_date"
            params['start_date'] = start_date
        if end_date:
            query_text += " AND date <= :end_date"
            params['end_date'] = end_date
        
        query = text(query_text).bindparams(**params)
        
        with engine.connect() as conn:
            result = conn.execute(query).fetchall()
        
        return result[0][0] if result else None
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

class GetTransactionSummaryInput(BaseModel):
    """Inputs for get_transaction_summary"""
    transaction_types: List[str]
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class GetTransactionSummaryTool(BaseTool):
    name = "get_transaction_summary"
    description = """
        Useful when you want to retrieve the total of transaction values (purchases, sales, dividends) from the 'investment_transaction' table in your database within a specified date range. 
        This tool requires the 'transaction_types' input to specify the types of transactions to include in the total (can be a list containing 'buy', 'sell', and/or 'dividend') and 'start_date' and 'end_date' as optional inputs to filter the transactions within a certain date range. If no dates are provided, it retrieves the total amount for all dates.
        The output will be the total value of the specified transactions retrieved from the database for the specified date range or the total value if no date range is specified.
        """
    args_schema: Type[BaseModel] = GetTransactionSummaryInput

    def _run(self, transaction_types: List[str], start_date: Optional[str] = None, end_date: Optional[str] = None):
        global engine
        if engine is None:
            init_db()        
        response = get_transaction_summary(engine, transaction_types, start_date, end_date)
        if response is None:
            return "No transactions found for the specified criteria."
        else:
            return f"The total amount for the specified transactions is: ${response}"

    def _arun(self):
        raise NotImplementedError("get_transaction_summary does not support async")
    


def get_top_stocks_by_dividend_yield(engine, top_n):
    try:
        query_text = """
            SELECT ticker_symbol, company_name, dividend_yield
            FROM stock_financial_data
            WHERE dividend_yield IS NOT NULL
            ORDER BY dividend_yield DESC
            LIMIT :top_n
        """
        params = {'top_n': top_n}
        query = text(query_text).bindparams(**params)
        
        with engine.connect() as conn:
            result = conn.execute(query).fetchall()
        
        return [{"ticker_symbol": row[0], "company_name": row[1], "dividend_yield": row[2]} for row in result]
    except Exception as e:
        print(f"An error occurred: {e}")
        return {"error": str(e)}

class GetTopStocksByDividendYieldInput(BaseModel):
    """Inputs for get_top_stocks_by_dividend_yield"""
    top_n: int


class GetTopStocksByDividendYieldTool(BaseTool):
    name = "get_top_stocks_by_dividend_yield"
    description = """
        Useful when you want to retrieve the top stocks with the highest dividend yield from the 'StockFinancialData' table in your database. 
        This tool requires the 'top_n' input to specify the number of top stocks to retrieve.
        The output will be a list of dictionaries, where each dictionary contains the ticker symbol, company name, and dividend yield of one of the top stocks.
        """
    args_schema: Type[BaseModel] = GetTopStocksByDividendYieldInput

    def _run(self, top_n: int):
        global engine
        if engine is None:
            init_db()        
        response = get_top_stocks_by_dividend_yield(engine, top_n)
        
        if isinstance(response, dict) and response.get("error"):
            return response["error"]
        elif not response:
            return "No stocks found meeting the criteria."
        else:
            return response

    def _arun(self):
        raise NotImplementedError("get_top_stocks_by_dividend_yield does not support async")

##################################################Portfolio Return Functions#########################################

# Helper Functions

def get_transactions(engine, user_id, security_id, start_date, end_date):
    try:
        # Check if the dates are not None and not already strings, then convert to string in 'YYYY-MM-DD' format
        if start_date and not isinstance(start_date, str):
            start_date_str = start_date.strftime('%Y-%m-%d')
        else:
            start_date_str = start_date
        
        if end_date and not isinstance(end_date, str):
            end_date_str = end_date.strftime('%Y-%m-%d')
        else:
            end_date_str = end_date

        # Define the SQL query to get transactions using a JOIN on security_id
        query = text(f"""
                    SELECT 
                        it.date, 
                        it.quantity, 
                        it.price, 
                        it.subtype,
                        it.type,
                        iv.ticker_symbol
                    FROM investment_transaction as it
                    JOIN investment_view as iv
                    ON it.security_id = iv.security_id
                    WHERE it.user_id = :user_id 
                        AND it.security_id = :security_id 
                        AND it.date BETWEEN :start_date AND :end_date;
        """)
        
        # Parameters to be used in the query
        params = {'user_id': user_id, 'security_id': security_id, 'start_date': start_date_str, 'end_date': end_date_str}
        
        # Obtain a connection object from the engine
        with engine.connect() as connection:
            # Execute the query and fetch all results
            result = connection.execute(query, params).fetchall()
        
        # Convert the result to a list of dictionaries
        transactions = [row._asdict() for row in result]

        return transactions

    except Exception as e:
        return {"error": str(e)}


# This function should query the investment_view table to get the current STOCKS portfolio of the user.
def fetch_current_portfolio(engine, user_id):
    """
    Fetch the current portfolio of a user from the investment_view table.

    Parameters:
    - engine: SQLAlchemy engine instance to connect to the database
    - user_id: The ID of the user whose portfolio we want to fetch

    Returns:
    - portfolio: A list of dictionaries containing the details of each security in the portfolio
    """
    try:
        # SQL query to fetch all records for the given user_id from the investment_view table
        query = text("""
            SELECT * 
            FROM investment_view 
            WHERE user_id = :user_id AND type IN ('equity', 'mutual fund', 'etf');
        """)

        # Execute the query with the provided user_id
        with engine.connect() as connection:
            result = connection.execute(query, {'user_id': user_id})

        # Fetch all rows from the result and construct the portfolio as a list of dictionaries
        portfolio = [row._asdict() for row in result]

        # If portfolio is empty, return an error message
        if not portfolio:
            return {"error": "No records found for the given user ID"}

        return portfolio
    except Exception as e:
        # Catch any exception that occurs and return an error message
        return {"error": str(e)}


# Function to reconcile the portfolio up to a certain date
def reconcile_portfolio_up_to_date(engine, user_id, start_date, current_portfolio):
    """
    Reconcile a user's portfolio up to a specific date using their transactions history.

    Parameters:
    - engine: SQLAlchemy engine instance to connect to the database
    - user_id: The ID of the user
    - start_date: The date up to which to reconcile the portfolio
    - current_portfolio: The current state of the user's portfolio

    Returns:
    - reconciled_portfolio: The state of the user's portfolio on the start_date
    """
    try:
        # Initialize an empty list to hold the reconciled portfolio
        start_date = find_previous_business_day(start_date)
        reconciled_portfolio = []

        end_date = datetime.now().strftime('%Y-%m-%d')
        end_date = find_previous_business_day(end_date)

        # Loop over each security in the current portfolio
        for security in current_portfolio:
            # Get the security_id for the current security
            security_id = security['security_id']
            security_type = security['type']
            # Fetch transactions for the current security up to the specified date
            transactions = get_transactions(engine, user_id, security_id, start_date, end_date)
            if 'error' in transactions:
                return {"error": "Could not fetch transactions for security ID: " + str(security_id)}

            # Reconcile the transactions to get the quantity held at the date
            quantity_on_date = security['quantity']
            for transaction in transactions:
                if transaction['subtype'] == 'buy':
                    quantity_on_date -= transaction['quantity']
                elif transaction['subtype'] == 'sell':
                    quantity_on_date -= transaction['quantity']
            
            # Add the reconciled data for the current security to the reconciled portfolio
            reconciled_portfolio.append({
                'security_id': security_id,
                'ticker_symbol': security['ticker_symbol'],
                'quantity': quantity_on_date,
                'type': security_type,
            })

        # Return the reconciled portfolio
        return reconciled_portfolio

    except Exception as e:
        # Return any other errors that occur
        return {"error": f"Error in reconcile_portfolio_up_to_date: {str(e)}"}

#Translate security ID to ticker symbol
def get_ticker_from_security_id(engine, security_id):
    """
    Get the ticker symbol for a given security ID.

    Parameters:
    - engine: SQLAlchemy engine instance to connect to the database
    - security_id: The ID of the security

    Returns:
    - ticker: The ticker symbol associated with the security ID
    """
    try:
        # Create a MetaData instance
        metadata = MetaData()

        # Reflect the investment_view table from the database
        investment_view = Table('investment_view', metadata, autoload_with=engine)

        # Create a select query to get the ticker symbol for the given security ID
        query = investment_view.select().where(investment_view.c.security_id == security_id)

        # Execute the query and fetch the result
        with engine.connect() as conn:
            result = conn.execute(query).fetchone()

        # Check if a result was found
        if result is not None:
            # Return the ticker symbol
            return result.ticker_symbol
        else:
            # If no result was found, return an error message
            return {"error": "No ticker found for the given security ID"}
    except Exception as e:
        # If any errors occur, return them
        return {"error": f"Error in get_ticker_from_security_id: {str(e)}"}


def calculate_average_purchase_price(transactions, start_quantity, start_date_price):
    # Initialize total_cost to 0. We will add to this as we loop through the transactions.
    if start_date_price is not None:
        total_cost = start_quantity * start_date_price
    else:
        total_cost = 0.0

    # If start_date_price is not None, add the initial investment amount to total_cost
    if start_date_price is not None:
        total_cost = start_quantity * start_date_price
    else:
        total_cost = 0.0
    # Initialize total_quantity to the quantity at the start date
    total_quantity = start_quantity

    # Loop through each transaction and update total_cost and total_quantity based on the transaction details
    for transaction in transactions:
        price_per_unit = transaction.get('price', 0)
        if transaction['type'] == 'buy':
            total_cost += transaction['quantity'] * price_per_unit
            total_quantity += transaction['quantity']


    # Calculate the average purchase price. If total_quantity is 0, set average_purchase_price to 0 to avoid division by zero error.
    if total_quantity != 0:
        average_purchase_price = total_cost / total_quantity
    else:
        average_purchase_price = 0.0

    return average_purchase_price


def get_all_transactions(engine, user_id: int, start_date: str, end_date: str):
    """
    Fetch all the investment transactions of a user within a specified date range from the investment_transactions table.

    Parameters:
    - engine: SQLAlchemy engine instance to connect to the database
    - user_id: The ID of the user whose transactions we want to fetch
    - start_date: The start date of the period for which we want to fetch the transactions (format: 'YYYY-MM-DD')
    - end_date: The end date of the period for which we want to fetch the transactions (format: 'YYYY-MM-DD')

    Returns:
    - transactions: A list of dictionaries containing the details of each transaction in the specified date range
    """
    try:
        # SQL query to fetch all records for the given user_id and date range from the investment_transactions table
        query = text("""
            SELECT * 
            FROM investment_transaction 
            WHERE user_id = :user_id AND date >= :start_date AND date <= :end_date
        """)

        # Execute the query with the provided user_id, start_date, and end_date
        with engine.connect() as connection:
            result = connection.execute(query, {'user_id': user_id, 'start_date': start_date, 'end_date': end_date})

        # Fetch all rows from the result and construct the transactions as a list of dictionaries
        transactions = [row._asdict() for row in result]

        # If transactions is empty, return an error message
        if not transactions:
            return []

        return transactions
    except Exception as e:
        # Catch any exception that occurs and return an error message
        return {"error": str(e)}


def get_portfolio_value(portfolio, date):
    total_value = 0.0

    for security in portfolio:
        # Ignore securities without a ticker symbol
        if not security['ticker_symbol']:
            continue
        
        # Get the price of the security on the specified date
        price_data = get_stock_price_on_date(security['ticker_symbol'], date)
        # print(f"Price data for {security['ticker_symbol']} on {date}: {price_data}")
        
        # If an error occurred while fetching the price, log the error and skip this security
        if 'error' in price_data:
            print(f"Error fetching price for {security['ticker_symbol']} on {date}: {price_data['error']}")
            continue
        
        # Calculate the value of this security on the specified date and add it to the total value
        security_value = price_data['price'] * security['quantity']
        total_value += security_value

    return total_value


def calculate_portfolio_returns(engine, start_date: str, end_date: str):
    global user_id
 
    if start_date in ["last year", "last month", "last week"]:
        start_date = get_date_range(start_date)
    if end_date in ["last year", "last month", "last week"]:
        end_date = get_date_range(end_date)
    #If the start date is not a business day, find the previous business day
    start_date = find_previous_business_day(start_date)
    if end_date > datetime.now().strftime('%Y-%m-%d'):
        end_date = datetime.now().strftime('%Y-%m-%d')
    end_date = find_previous_business_day(end_date)

    # Step 1: Get the initial and final value of the portfolio
    current_portfolio = fetch_current_portfolio(engine, user_id)
    initial_portfolio = reconcile_portfolio_up_to_date(engine, user_id, start_date, current_portfolio)
    # print(f"Initial portfolio: {initial_portfolio}")
    if 'error' in initial_portfolio:
        # print(f"Error in fetching initial portfolio: {initial_portfolio['error']}")
        initial_portfolio = []

    final_portfolio = reconcile_portfolio_up_to_date(engine, user_id, end_date, current_portfolio)
    # print(f"Final portfolio: {final_portfolio}")
    if 'error' in final_portfolio:
        # print(f"Error in fetching final portfolio: {final_portfolio['error']}")
        final_portfolio = []

    initial_portfolio_value = get_portfolio_value(initial_portfolio, start_date)
    # print(f"Initial portfolio value: {initial_portfolio_value}")
    final_portfolio_value = get_portfolio_value(final_portfolio, end_date)
    # print(f"Final portfolio value: {final_portfolio_value}")

    # Step 2: Identify and compute all the cash flows
    all_transactions = get_all_transactions(engine, user_id, start_date, end_date)  # You will need to create this function

    
    # Step 3: Calculate weighted cash flow
    start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    total_days_in_period = (end_date_obj - start_date_obj).days

    weighted_cash_flow = 0
    net_cash_flow = 0
    for transaction in all_transactions:
        transaction_date_obj = transaction['date'] if isinstance(transaction['date'], date) else datetime.strptime(transaction['date'], '%Y-%m-%d')
        days_since_start = (transaction_date_obj - start_date_obj.date()).days
        weight = (total_days_in_period - days_since_start) / total_days_in_period
        
        # Determine the cash flow amount considering the subtypes
        cash_flow = 0
        if transaction['type'] in ['buy', 'sell', 'dividend', 'dividend reinvestment', 'interest reinvestment']:
            cash_flow = transaction['amount']  
        
        weighted_cash_flow += cash_flow * weight
        net_cash_flow += cash_flow

    # print(f"Weighted cash flow: {weighted_cash_flow}")
    # print(f"Net cash flow: {net_cash_flow}")
    # Step 4: Calculate Modified Dietz return
    modified_dietz_return = (final_portfolio_value - initial_portfolio_value - net_cash_flow) / (initial_portfolio_value + weighted_cash_flow)
    total_returns = final_portfolio_value - initial_portfolio_value - net_cash_flow

    return {
        "percentage_return": modified_dietz_return,
        "money_return": total_returns
    }


class CalculatePortfolioReturnsInput(BaseModel):
    input_data: Union[dict, str] = Field(
        ..., 
        description="The input data containing start_date and end_date for calculating the portfolio returns.",
        example={"start_date": "2023-01-01", "end_date": "2023-09-12"}
    )



# Define the tool class
class CalculatePortfolioReturnsTool(BaseTool):
    name = "calculate_portfolio_returns"
    description = """
        Calculate the portfolio returns over a specified period using the Modified Dietz method.
        The tool takes in a start date and an end date to calculate the returns over that period.
        It calculates realized and unrealized gains, dividend returns, total return, and the overall portfolio return percentage.
    """
    args_schema: Type[BaseModel] = CalculatePortfolioReturnsInput

    def _run(self, input_data: Union[dict, str]):
        # Assuming `engine` is a global variable or passed in some other way

        # Check if the input_data is a string and convert it to a dictionary
        if isinstance(input_data, str):
            try:
                input_data = json.loads(input_data)
            except json.JSONDecodeError as e:
                return {"error": f"Failed to parse input string as JSON: {e}"}

        # Now extract start_date and end_date
        start_date = input_data.get('start_date')
        end_date = input_data.get('end_date')
        
        if not start_date or not end_date:
            return {"error": "start_date and end_date must be provided"}
        
        try:
            returns_response = calculate_portfolio_returns(engine, start_date, end_date)
            return returns_response
        except Exception as e:
            return {"error": str(e)}

    def _arun(self, input_data: Union[dict, str]):
        raise NotImplementedError("calculate_portfolio_returns does not support async")




def determine_investment_profile(data):
    score = 0
    
    # Age scoring
    if 18 <= data['age'] <= 30:
        score += 5
    elif 31 <= data['age'] <= 45:
        score += 4
    elif 46 <= data['age'] <= 60:
        score += 3
    else:
        score += 2

    # Marital status scoring
    marital_status_scores = {
        'Single': 5,
        'Married': 4,
        'Divorced': 3,
        'Widowed': 2
    }
    score += marital_status_scores.get(data['maritalStatus'], 0)

    # Number of Dependents scoring
    if data['dependents'] == 0:
        score += 5
    elif 1 <= data['dependents'] <= 2:
        score += 4
    elif 3 <= data['dependents'] <= 4:
        score += 3
    else:
        score += 2

    # Employment Status scoring
    employment_status_scores = {
        'Full-time': 5,
        'Part-time': 4,
        'Self-employed': 5,
        'Unemployed': 2,
        'Retired': 3
    }
    score += employment_status_scores.get(data['employmentStatus'], 0)

    # Annual Income scoring (assuming income is in thousands for simplicity)
    if 0 <= data['annualIncome'] <= 30:
        score += 2
    elif 30 < data['annualIncome'] <= 60:
        score += 3
    elif 60 < data['annualIncome'] <= 100:
        score += 4
    else:
        score += 5

    # Financial Goals scoring
    goal_scores = {
        'Buying a home': 4,
        'Retirement': 3,
        'Children\'s education': 3,
        'Travel': 5,
        'Starting a business': 5
    }
    for goal in data['goals']:
        score += goal_scores.get(goal, 0)

    # Major Expenses in Next 5 Years scoring
    major_expenses_scores = {
        'Wedding': 3,
        'Home purchase': 3,
        'World tour': 4
    }
    for expense in data['majorExpensesNext5Years']:
        score += major_expenses_scores.get(expense, 0)

    # Ongoing Financial Commitments scoring
    if data['financialCommitments']:
        score += 3
    else:
        score += 5


    # Investment Duration scoring
    if 0 <= data['investmentDuration'] <= 2:
        score += 2
    elif 3 <= data['investmentDuration'] <= 5:
        score += 3
    elif 6 <= data['investmentDuration'] <= 10:
        score += 4
    else:
        score += 5

    # Investment Knowledge scoring
    score += data['investmentKnowledgeRating']

    # Investment Style scoring
    investment_style_scores = {
        'Very Conservative': 1,
        'Conservative': 2,
        'Moderate': 3,
        'Aggressive': 4,
        'Very Aggressive': 5
    }
    score += investment_style_scores.get(data['riskTolerance'], 0)

    # Reaction to Investment Drop scoring
    investment_reaction_scores = {
        'Sell all': 1,
        'Sell some': 2,
        'Do nothing': 3,
        'Buy more': 5
    }
    score += investment_reaction_scores.get(data['investmentReaction'], 0)

    # Capital Preference scoring
    capital_preference_scores = {
        'Preserving capital': 2,
        'Growing capital': 5
    }
    score += capital_preference_scores.get(data['capitalPreference'], 0)

    # Feelings About Debt scoring
    debt_feeling_scores = {
        'Avoid at all costs': 2,
        'Necessary evil': 3,
        'Useful tool for growth': 5
    }
    score += debt_feeling_scores.get(data['debtFeeling'], 0)

    # Anticipated Changes in Income or Expenses in Next 3-5 Years scoring
    if data['anticipatedChangesNext3_5Years']:
        score += 3
    else:
        score += 5

    # Review and Adjust Investments Frequency scoring
    review_frequency_scores = {
        'Monthly': 5,
        'Quarterly': 4,
        'Bi-annually': 3,
        'Annually': 2,
        'Only when contacted': 1
    }
    score += review_frequency_scores.get(data['reviewFrequency'], 0)

    # Determine profile based on total score
    if score <= 50:
        return "Conservative"
    elif 50 < score <= 75:
        return "Moderate"
    else:
        return "Aggressive"


def recommend_portfolio(engine):
    global user_id
    # Basic Allocation Profiles
    portfolios = {
        "Conservative": {"Stocks": 15, "Bonds": 80, "Cash": 5},
        "Moderate": {"Stocks": 35, "Bonds": 60, "Cash": 5},
        "Aggressive": {"Stocks": 50, "Bonds": 45, "Cash": 5}
    }

    # Query the database to fetch the user's investment profile
    query = text("""
        SELECT profile 
        FROM investment_profile
        WHERE user_id = :user_id
    """)
    try:
        # Create a statement object to bind the parameters
        stmt = query.bindparams(user_id=user_id)
        
        # Execute the SQL query and fetch the result
        with engine.connect() as conn:
            result = conn.execute(stmt).fetchone()

        # If no result found, return an error or default portfolio
        if not result:
            return {"error": "User's investment profile not found"}

        # Return the recommended portfolio based on the fetched investment profile
        return portfolios.get(result[0], {"error": "Invalid investment profile"})

    except Exception as e:
        print(f"Error fetching investment profile: {e}")
        return {"error": "An error occurred while fetching the investment profile"}

class RecommendPortfolioInput(BaseModel):
    """Inputs for recommend_portfolio"""

class RecommendPortfolioTool(BaseTool):
    name = "recommend_portfolio"
    description = """
        This tool recommends an investment portfolio based on the user's profile. 
        The output will be a portfolio allocation recommendation in terms of percentages for Stocks, Bonds, and Cash.
        """
    args_schema: Type[BaseModel] = RecommendPortfolioInput

    def _run(self) -> Dict[str, int]:
        global user_id
        recommended_portfolio = recommend_portfolio(engine)
        return recommended_portfolio

    def _arun(self):
        raise NotImplementedError("recommend_portfolio does not support async")

# AI agent analyst

def stock_analysis(user_input: str, chathistory: List[dict], user_id_param: int):
    global user_id
    user_id = user_id_param
    save_user_question(user_id_param, user_input)
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
    tools = [
             CurrentStockPriceTool(), 
             DateRangeTool(),
             StockPerformanceTool(), 
             StockPriceOnDateTool(), 
             StockGetBestPerformingTool(),
             StockGetWorstPerformingTool(),
             CalculatePortfolioReturnsTool(),
             GetPortfolioBetaTool(),
             GetTotalInvestmentValuesTool(),
             GetAllTickerSymbolsTool(),
             GetStockQuantityTool(),
             GetCashPositionTool(),
             LatestStockNewsTool(),
             GetTransactionSummaryTool(),
             GetTopStocksByDividendYieldTool(),
             RecommendPortfolioTool()
            ]

    system_message = SystemMessage(content="""
You are an experienced investment advisor and your goal is to help the user manage their investments and provide advice about securities to clients.
You should recommend suitable investments: securities or investment products that align with the client's goals and risk tolerance. You should be able to explain the rationale behind each recommendation.  
You also helps monitoring the client's portfolio and provide advice on how to improve the portfolio's performance. You should never recommend the user to look for a different advisor or broker.
If the user asks for a portfolio recommendation, you should first use RecommendPortfolioTool first to know the user's risk tolerance and then recommend a portfolio allocation based on the user's risk tolerance.
-ALWAYS use first the `DateRangeTool` to determine the exact date ranges when users inquire involves specific time frames such as "last year," "last month," or "last week." Here’s when and how to use it:
    1. **Performance Analysis** – For requests involving the performance of a portfolio or a single security over specified relative time frames, like “last year.” Utilize the tool to get the precise dates.
    2. **Comparative Analysis** – When comparing performance with a benchmark or another security using phrases such as "in the last year." Retrieve uniform date ranges for both data sets using the tool.
    3. **Stock Analysis** – When asked about the best or worst performing stocks in a set time frame, like "last month." Use the tool to find out the exact dates for analysis.
    For instance, if a query is, "What was the return of my portfolio in the last year?", use the tool with the period set to "last year" to obtain the correct date range for analysis.
    Remember, always align the date ranges for accurate and meaningful results.
- When asked to "compare the portfolio against a market index" or to detail the "overall performance of the portfolio", 
    employ the 'CalculatePortfolioReturnsTool()' to ascertain the portfolio's return before contrasting it with the most appropriate market index. 
    Detail the portfolio's return formatted with two decimal places and punctuation. You should add to the answer information on the best and worst performing assets in the portfolio
        using BestPerformingStockTool() and WorstPerformingStockTool() respectively.
- If the user inquires "What is the return of my portfolio?", follow the similar steps as above but only provide the return of the portfolio without comparing it to any market index. 
    The portfolio return should be calculated using 'CalculatePortfolioReturnsTool()'.
-Remember to always check the current date when calculating Year-to-Date (YTD) or Month-to-Date (MTD) performance for a query period that includes the current date.
When a user asks about their performance "in 2023" or "in May" during those respective periods, interpret this as a request for YTD or MTD performance, respectively. Calculate the performance from the start of the specified year or month to today's date.
For instance, if a user asks about their 2023 performance and today is September 12, 2023, calculate the performance from January 1, 2023, to September 12, 2023.
In the 'calculate_portfolio_returns' function, set the 'end_date' parameter to today's date in order to get accurate YTD or MTD performance for queries encompassing the present date.
- In queries regarding the "best performing stock in my portfolio", begin by retrieving the stock information from the database using the 'get_all_ticker_symbols' method, 
    followed by utilizing the 'StockGetBestPerformingTool()' to answer the query adequately.
- When assessing the risk level of the portfolio through questions such as "how risky is my portfolio?", fetch the stock tickers using the 'get_all_ticker_symbols' method 
    and the number of shares via 'get_stock_quantity'. Use 'GetPortfolioBetaTool()' afterwards to calculate the portfolio beta.
- For questions about the cash position such as "what is my cash position?", use the 'GetCashPositionTool'. 
    If the output reads "None", inform the user that there is no allocation in cash at present.
- Always adhere to checking the query meticulously before execution to avoid errors. In case an error is encountered, rewrite and reattempt the query. 
    Refrain from executing DML statements (INSERT, UPDATE, DELETE, DROP, etc.) in the database.
- Treat "stocks" synonymously with "equity" and "market" as equivalent to "S&P 500" when querying the database.
- Present all numerical values with two decimal points and appropriate punctuation to ensure clarity and professionalism.
- Prioritize double-checking your final response to maintain accuracy and reliability in assisting the user.
- If the user asks "which bank gives best APY?" or something similar, you should use GoogleSearchTool to search Google for the best APY.
By steadfastly following these instructions, provide the user with proficient assistance in understanding and managing their investment portfolio.
                            """)

    MEMORY_KEY = "chat_history"
    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name=MEMORY_KEY)]
    )
    chat_memory = ChatMessageHistory()
    for message in chathistory:
        if message["role"] == "user":
            chat_memory.add_user_message(message["content"])
        else:
            chat_memory.add_message(SystemMessage(content=message["content"]))
    memory = ConversationBufferMemory(memory_key=MEMORY_KEY, return_messages=True, chat_memory=chat_memory)

    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt, memory=memory, verbose=True)
    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=False)


    response = agent_executor.run(user_input)

    return response


langchain.debug = True
