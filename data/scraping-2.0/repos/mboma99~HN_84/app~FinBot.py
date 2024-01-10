import os

import serpapi
from langchain.llms import OpenAI

from apikey import apikey
from apikey import serp_apikey


class FinBot:

    def __init__(self):
        os.environ['OPENAI_API_KEY'] = apikey
        self.llm = OpenAI(temperature=0.9)

    def getInsights(self, symbol):
        return self.llm(f'give brief insight of the on stock ticker {symbol}')

    def getSearchResults(self, search_prompt):
            os.environ['SERPAPI_KEY'] = serp_apikey
            api_key = os.getenv('SERPAPI_KEY')

            client = serpapi.Client(api_key=api_key)
            result = client.search(
                q=f"{search_prompt}",
                engine="google",
                location="United Kingdom",
                hl="en",
                gl="us",
                num=5,
            )
            organic_results = result["organic_results"]
            displayed_links = [entry['link'].split(' ')[0] for entry in organic_results]
            return displayed_links
