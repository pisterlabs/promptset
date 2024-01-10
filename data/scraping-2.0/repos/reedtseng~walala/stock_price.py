from langchain.tools import BaseTool
from langchain.agents import AgentType
from typing import Optional, Type
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from FinMind.data import DataLoader

today = datetime.now()
formated_today = today.strftime("%Y-%m-%d")
yesterday = today - timedelta(days=1)

tw_stocks = {
    "0050": {
        "aliases": ["0050.TW", "元大台灣50"],
        "price": None,
    },
    "2317": {
        "aliases": ["2317.TW", "鴻海", "海公公", "阿海"],
        "price": None,
    },
    "2330": {
        "aliases": ["台積電", "TSM", "台積", "2330.TW"],
        "price": None,
    },
    "2412": {
        "aliases": ["2412.TW", "中華電", "種花"],
        "price": None,
    },
    "2883": {
        "aliases": ["2883.TW", "開發金", "開發"],
        "price": None,
    },
    "3293":	{
        "aliases": ["3293.TW", "鈊象", "小象"],
        "price": None,
    },
    "3443":	{
        "aliases": ["3443.TW", "創意"],
        "price": None,
    },
    "3661": {
        "aliases": ["3661.TW", "世芯", "世芯-KY"],
        "price": None,
    },
    "6279": {
        "aliases": ["6279.TW", "胡連", "小胡"],
        "price": None,
    },
    "6669":	{
        "aliases": ["6669.TW", "緯穎"],
        "price": None,
    },
}
reversed_stocks = {}


def load_tw_stocks():
    # with open('tw_stocks.txt', 'r', encoding='utf-8') as file:
    #     for line in file:
    #         key, value = line.strip().split(':')
    #         tw_stocks[key] = value
    # print(tw_stocks)
    for key, value in tw_stocks.items():
        for alias in value["aliases"]:
            reversed_stocks[alias] = key


load_tw_stocks()

start_date = yesterday.strftime("%Y-%m-%d")
dl = DataLoader()


def get_stock_latest_price(symbol):

    ticker_no = reversed_stocks.get(symbol, symbol)
    ticker = tw_stocks.get(ticker_no, None)
    if ticker == None:
        return "無此{symbol}股票資料"

    # 如果沒記錄 price, 才要再 call一次 API
    if ticker["price"] is None:
        stock_data = dl.taiwan_stock_daily(
            stock_id=ticker_no, start_date=start_date)
        ticker["price"] = float(stock_data.iloc[-1]['close'])
    return ticker["price"]


class StockPriceCheckInput(BaseModel):
    """Input for Stock price check."""

    stockticker: str = Field(...,
                             description="Ticker symbol for stock")


class StockPriceTool(BaseTool):
    name = "get_stock_ticker_price"
    description = "Useful for when you need to find out the price of stock. You should input the stock ticker symbol"

    def _run(self, stockticker: str):
        price_response = get_stock_latest_price(stockticker)

        return price_response

    def _arun(self, stockticker: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = StockPriceCheckInput
