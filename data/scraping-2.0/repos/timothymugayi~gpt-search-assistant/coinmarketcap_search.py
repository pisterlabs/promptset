"""Util that calls Coinmarketcap API."""
import os
import json
import pytz
import logging

from typing import Any, Dict, Optional, Union, List
from datetime import datetime

from pydantic import BaseModel, Extra, root_validator
from langchain.tools import BaseTool
from langchain.utils import get_from_dict_or_env


logger = logging.getLogger(__name__)


class CryptocurrencySearchAPIWrapper(BaseModel):
    """Wrapper for CryptoCurrency coinmarketcap Search API.

    1. Install coinmarketcap pypi package
    - pip install python-coinmarketcap
    - If you don't already have a Coinmarketcap Developer account, navigate to https://coinmarketcap.com/api/ and signup
    2. Once you have logged in you should see your API key under Overview tab
    3. Copy out the API key and add it to your environment variables COINMARKETCAP_API_KEY
    """
    cmc: Any
    coinmarketcap_api_key: Optional[str] = None
    coin_index_cache_duration: int = 1
    cache_file_name: str = "cryptocurrency_map.json"

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def _get_coin_list(self) -> List[Dict[str, Any]]:

        def _get_cached_coin_index_file(file_name: str) -> Union[str, None]:
            local_tz = pytz.timezone("UTC")
            if os.path.exists(file_name):
                mod_time = datetime.fromtimestamp(os.path.getmtime(file_name), tz=local_tz)
                if (datetime.now(tz=local_tz) - mod_time).days >= self.coin_index_cache_duration:
                    try:
                        os.remove(file_name)
                    except OSError:
                        pass
                    return None
                else:
                    return file_name
            else:
                return None

        if not _get_cached_coin_index_file(self.cache_file_name):
            start = 1
            limit = 5000
            results = []
            while True:
                rep = self.cmc.cryptocurrency_map(start=start, limit=limit)
                if not rep.data:
                    break
                results.extend(rep.data)
                start += limit
            with open(self.cache_file_name, 'w') as f:
                json.dump(results, f)
            logger.info("total cached coins = {}".format(len(results)))
            return results
        else:
            with open(self.cache_file_name, 'r') as f:
                results = json.load(f)
                logger.info("total cached coins = {}".format(len(results)))
                return results

    def _parse_search_query(self, search_term: str) -> Union[str, None]:
        if not search_term:
            raise ValueError("No query found")
        _search_values = search_term.lower().split(",")
        found_items = {}
        slug_or_symbols = [slug_or_symbol.lower().strip() for slug_or_symbol in _search_values]
        for result in self._get_coin_list():
            if result["name"].lower() in slug_or_symbols or result["symbol"].lower() in slug_or_symbols:
                found_items[result["id"]] = result
            if len(found_items) == len(_search_values):
                break
        if found_items:
            key_ids = ",".join(str(_id) for _id in found_items.keys())
            logger.debug("Found key_ids: {}".format(key_ids))
            return key_ids
        else:
            return None

    def _coinmarketcap_search_results(self, search_term: str) -> Dict[str, Any]:
        try:
            logger.debug("search_term: {}".format(search_term))
            coin_ids = self._parse_search_query(search_term)
            logger.debug("Fetched coin_ids: {}".format(coin_ids))
            if not coin_ids:
                return {}
            rep = self.cmc.cryptocurrency_info(id=coin_ids)
            return rep.data
        except Exception as e:
            logger.error(e)

            from coinmarketcapapi import CoinMarketCapAPIError

            if isinstance(e, CoinMarketCapAPIError):
                raise e

            return dict()

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        coinmarketcap_api_key = get_from_dict_or_env(
            values, "coinmarketcap_api_key", "COINMARKETCAP_API_KEY"
        )
        values["coinmarketcap_api_key"] = coinmarketcap_api_key

        try:
            from coinmarketcapapi import CoinMarketCapAPI
            values["cmc"] = CoinMarketCapAPI(api_key=coinmarketcap_api_key)
        except ImportError:
            raise ImportError(
                "python-coinmarketcap package is not installed. "
                "Please install it with `pip install python-coinmarketcap`"
            )
        return values

    def run(self, query: str) -> str:
        """Run query through coinmarketcap and parse result."""
        snippets = []
        results = self._coinmarketcap_search_results(query)
        logger.debug("results: {}".format(results))
        if len(results) == 0:
            return "No good Coinmarketcap Search Result was found"
        for v in results.values():
            coin_name = v["name"]
            symbol = v["symbol"]
            desc = v["description"]
            urls = v["urls"]
            coin_type = v["category"]
            fields = [(key, value) for key, value in urls.items() if value]
            coin_urls = []
            for field in fields:
                value = field[1]
                if isinstance(value, list):
                    value = ", ".join(value)
                coin_urls.append(f"{field[0]}: {value}")

            snippet = """
            Crypto Currency Coin name: {coin_name}
            {coin_name} Symbol: {symbol}
            {coin_name} Coin Details: {desc} 
            Coin Type: {coin_type}
            """.format(coin_name=coin_name, symbol=symbol, desc=desc, coin_type=coin_type)

            snippets.append(snippet)
        logger.debug("Found data: {}".format(len(snippets)))
        return "\n".join(snippets)


class CryptocurrencySearchQueryRun(BaseTool):
    """Tool that adds the capability to search using the Coinmarketcap API."""

    name = "Crypto Currency Search"
    description = (
        "Use this tool when you need to answer questions about Cryptocurrency prices or altcoins prices "
        "Input should be the Cryptocurrency coin name only or the ticker symbol"
    )
    api_wrapper: CryptocurrencySearchAPIWrapper

    def _run(self, query: str) -> str:
        """Use the toCurrency coinmarketcap tool."""
        return self.api_wrapper.run(query)

    async def _arun(self, query: str) -> str:
        """Use the toCurrency coinmarketcap tool asynchronously."""
        raise NotImplementedError("CryptocurrencySearchQueryRun does not support async")
