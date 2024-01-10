import os
from typing import Optional

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.base import BaseTool

SOURCES = "bbc-news,the-verge,the-wall-street-journal"


class NewsAPITool(BaseTool):
    """Tool that searches news using News API."""

    name: str = "news"
    description: str = (
        "Useful when you need to get top headlines from major news sources "
        "such as BBC News and Wall Street Journal."
    )
    top_k: int = 10
    """The number of results to return."""

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use NewsAPI tool."""
        try:
            from newsapi import NewsApiClient
        except ImportError:
            raise ImportError(
                "Could not import yfinance python package. "
                "Please install it with `pip install newsapi-python`."
            )

        try:
            newsapi = NewsApiClient(api_key=os.environ["NEWSAPI_API_KEY"])
        except KeyError:
            raise ("NEWSAPI_API_KEY is not found in environ.")

        top_headlines = newsapi.get_top_headlines(q=query, sources=SOURCES)

        result = "\n\n".join(
            [
                "\n".join([n["title"], n["description"]])
                for n in top_headlines["articles"]
            ]
        )
        if not result:
            return f"No news found for '{query}'."
        return result
