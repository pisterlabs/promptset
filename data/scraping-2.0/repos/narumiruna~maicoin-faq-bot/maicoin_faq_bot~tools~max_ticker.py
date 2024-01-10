import requests
from langchain.tools import BaseTool

base_url = 'https://max-api.maicoin.com'


class MAXTicker(BaseTool):
    name: str = "query_max_ticker"

    description: str = ('Useful for when you want to get the ticker of a currency pair on MAX exchange. '
                        'Input is a market.')

    def _run(self, market: str) -> str:
        market = market.replace('/', '').lower()

        url = f'{base_url}/api/v2/tickers/{market}'

        resp = requests.get(url)

        if resp.status_code != 200:
            return f'failed to query ticker: {resp.status_code}'

        data = resp.json()

        return (f'Market: {market}\n'
                f'Buy: {data["buy"]}\n'
                f'Sell: {data["sell"]}\n'
                f'Last: {data["last"]}\n')

    def _arun(self, symbol: str):
        return self._run(symbol)
