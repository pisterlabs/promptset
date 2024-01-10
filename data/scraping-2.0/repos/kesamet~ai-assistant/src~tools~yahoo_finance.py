from typing import Optional

from requests.exceptions import HTTPError, ReadTimeout
from urllib3.exceptions import ConnectionError

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.base import BaseTool


class YahooFinanceTool(BaseTool):
    """Tool that searches financial data on Yahoo Finance."""

    name: str = "yahoo_finance"
    description: str = (
        "Useful for when you need to find financial data about a public company. "
        "Input should be a company ticker. For example, AAPL for Apple, MSFT for Microsoft."
    )

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Yahoo Finance News tool."""
        try:
            import yfinance
        except ImportError:
            raise ImportError(
                "Could not import yfinance python package. "
                "Please install it with `pip install yfinance`."
            )

        try:
            company = yfinance.Ticker(query)
        except (HTTPError, ReadTimeout, ConnectionError):
            return f"Company ticker {query} not found."

        try:
            df = company.history()
        except (HTTPError, ReadTimeout, ConnectionError):
            return f"No data found for company that searched with {query} ticker."

        if df.empty:
            return f"No news found for company that searched with {query} ticker."
        return df
