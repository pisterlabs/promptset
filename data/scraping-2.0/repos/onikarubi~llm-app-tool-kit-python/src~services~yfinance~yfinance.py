from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools.base import BaseTool
from asyncer import asyncify
from typing import Optional, Type
from src.models.ticker_symbol import TickerSymbol
from pydantic import BaseModel
import pytest
import yfinance as yf


def get_crypt_data(symbol: str):
    gld = yf.Ticker(symbol)
    return gld.info


class YahooFinanceTool(BaseTool):
    name = "get_crypt_data"
    description = "YahooFinanceのAPIを使用して、通貨の価格や株価を取得します"

    def _run(self, symbol: str):
        result = get_crypt_data(symbol)
        return result

    def _arun(self, symbol: str):
        return asyncify(self._run, cancellable=False)(symbol)

    args_schema: Optional[Type[BaseModel]] = TickerSymbol
