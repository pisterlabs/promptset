import json
from typing import List, Any

import requests
from dotenv import load_dotenv
from langchain.utilities import GoogleSerperAPIWrapper
from pydantic import BaseModel

from morpher.tools import Tool

load_dotenv()


class WebSearch(BaseModel):
    tool_info: List[Tool] = []

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.tool_info = [
            Tool(
                name="web_search",
                description="Useful to get latest information about the query by searching the web. Input must be a string.",
                func=self.web_search
            ),
        ]

    @staticmethod
    def web_search(query: str):
        """
        Useful for searching the web based on the search query using GoogleSerperAPI from Langchain. Needs
        SERPER_API_KEY and OPENAI_API_KEY as an environment variable.

        :param query: Search query as a string.
        :return: Search result as a string.
        """
        search = GoogleSerperAPIWrapper()
        return search.run(query)

    @staticmethod
    def deep_search(query: str):
        # TODO: implement web_crawler with browserless or something similar for in-depth searches
        pass

    @staticmethod
    def manual_serper_call(query: str):
        """
        Useful for searching the web based on a search query using Google's Serper manually.

        :param query: Search query as a string.
        :returns: Top 5 search results.
        """
        url = "https://google.serper.dev/search"
        payload = json.dumps({
            "q": query
        })
        headers = {
            'X-API-KEY': '',
            'Content-Type': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        raw_snippets = json.loads(response.text)['organic']
        for s in raw_snippets:
            print(s['snippet'])

        return raw_snippets
