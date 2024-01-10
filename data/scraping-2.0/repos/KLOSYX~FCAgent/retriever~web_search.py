from __future__ import annotations

from langchain.tools import BaseTool
from langchain.tools import BingSearchResults
from langchain.tools import DuckDuckGoSearchResults
from langchain.tools import GoogleSearchResults
from langchain.utilities import BingSearchAPIWrapper
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.utilities import GoogleSearchAPIWrapper

from config import config


def get_web_searcher():
    if config.search_engine == "duckduckgo":
        wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
        web_search = DuckDuckGoSearchResults(api_wrapper=wrapper)
    elif config.search_engine == "bing":
        wrapper = BingSearchAPIWrapper(k=5)
        web_search = BingSearchResults(api_wrapper=wrapper)
    else:
        wrapper = GoogleSearchAPIWrapper(k=5)
        web_search = GoogleSearchResults(api_wrapper=wrapper)
    return web_search


class WebSearchTool(BaseTool):
    name = "web_search_tool"
    description = (
        "use this tool when you need to search web page. the query could be in English or Chinese."
        "use parameter `query` as input."
    )

    def _run(self, query: str) -> str:
        """use string 'query' as input. could be any language."""
        return get_web_searcher().run(query) + "\n"

    def _arun(self, query: str) -> list[str]:
        raise NotImplementedError("This tool does not support async")
