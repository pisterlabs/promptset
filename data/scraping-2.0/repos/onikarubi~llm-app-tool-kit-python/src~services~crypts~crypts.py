from pycoingecko import CoinGeckoAPI
from pydantic import BaseModel, Field
from typing import Type
from enum import Enum
from langchain.tools import BaseTool
from asyncer import asyncify
from typing import Optional, Type
from src.models.crypts import GetCryptocurrencyPriceInput
import json


cg = CoinGeckoAPI()


def get_cryptocurrency_price(crypts: list[str], vs_currencies: str):
    results = cg.get_price(ids=crypts, vs_currencies=vs_currencies)
    return json.dumps(results)


class CryptocurrencyPriceTool(BaseTool):
    name = "get_cryptocurrency_price"
    description = "必要な他のサポート通貨での暗号通貨の現在の価格を取得します"

    def _run(self, crypts: list[str], vs_currencies: str):
        result = get_cryptocurrency_price(crypts, vs_currencies)
        return result

    def _arun(self, crypts: list[str], vs_currencies: str):
        return asyncify(self._run, cancellable=False)(crypts, vs_currencies)

    args_schema: Optional[Type[BaseModel]] = GetCryptocurrencyPriceInput