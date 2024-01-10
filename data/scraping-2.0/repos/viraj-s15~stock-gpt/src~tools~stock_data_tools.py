import requests
import yfinance as yf
from args import GenericStockInfoArgs, GetTickerArgs
from langchain.tools import StructuredTool, Tool, tool
from pydantic import BaseModel, Field


@tool("ticker", args_schema=GetTickerArgs)
def get_ticker(company_name) -> str:
    """
    Provides the ticker for a given company name
    """
    try:
        yfinance = "https://query2.finance.yahoo.com/v1/finance/search"
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
        params = {"q": company_name, "quotes_count": 1, "country": "United States"}

        res = requests.get(
            url=yfinance, params=params, headers={"User-Agent": user_agent}
        )
        data = res.json()
        company_code = data["quotes"][0]["symbol"]
        return company_code
    except Exception as e:
        return f"Error getting ticker: {str(e)}"


@tool("generic_stock_information", args_schema=GenericStockInfoArgs)
def get_generic_stock_information(stock_name: str) -> str:
    """
    Function which retrieves general information related to the stock, such as current price, etc.
    """
    try:
        ticker = yf.Ticker(get_ticker(stock_name))
        info = dict()
        info["Current Price"] = ticker.info.get("currentPrice")

        info["Previous Close"] = ticker.info["previousClose"]
        info["Open Price"] = ticker.info["open"]
        info["Day Low"] = ticker.info["dayLow"]
        info["Day High"] = ticker.info["dayHigh"]
        info["Regular Market Previous Close"] = ticker.info[
            "regularMarketPreviousClose"
        ]
        info["Regular Market Previous Open"] = ticker.info["regularMarketOpen"]
        info["Regular Market Day Low"] = ticker.info["regularMarketDayLow"]
        info["Regular Market Day High"] = ticker.info["regularMarketDayHigh"]
        info["Market Cap"] = ticker.info["marketCap"]
        info["52 Week low"] = ticker.info["fiftyTwoWeekLow"]
        info["52 Week high"] = ticker.info["fiftyTwoWeekHigh"]
        info["Target High Price"] = ticker.info["targetHighPrice"]
        info["Target Low Price"] = ticker.info["targetLowPrice"]
        info["Target Mean Price"] = ticker.info["targetMeanPrice"]
        info["Target Median Price"] = ticker.info["targetMedianPrice"]

        return str(info)
    except Exception as e:
        return f"Error getting generic stock information: {str(e)}"


def get_historical_data_relative_time(stock_name: str, time_period: str) -> str:
    """
    Function which retrieves historical data with respect to the time period.
    Accepted values are: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    """
    try:
        ticker = yf.Ticker(get_ticker(stock_name))
        res = ticker.history(period=time_period)
        return str(res)
    except Exception as e:
        return f"Error getting historical data relative time: {str(e)}"


get_historical_data_relative_time_tool = StructuredTool.from_function(
    get_historical_data_relative_time
)


def get_historical_data_date_wise(stock_name: str, date: str) -> list[dict]:
    """
    Function which gets historical data based on a specific date.
    Date format: YYYY-MM-DD, the date must be in the past
    """
    try:
        ticker = yf.Ticker(get_ticker(stock_name))
        res = ticker.history(start=date, end=None)
        return str(res)
    except Exception as e:
        return f"Error getting historical data date wise: {str(e)}"


get_historical_data_date_wise_tool = StructuredTool.from_function(
    get_historical_data_date_wise
)

stock_data_tools = [
    get_ticker,
    get_generic_stock_information,
    get_historical_data_relative_time_tool,
    get_historical_data_date_wise_tool,
]
