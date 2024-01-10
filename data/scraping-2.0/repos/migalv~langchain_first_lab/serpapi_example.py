import os
from langchain import SerpAPIWrapper
import json

"""
PydanticOutputParser - LLM which allows users to specify an arbitrary JSON schema and query LLMs for JSON outputs that 
conform to that schema
JSONFormer library - Structured decoding of a subset of the JSON Schema
"""

"""
There is an alternative with 1000 free queries 
https://python.langchain.com/en/latest/modules/agents/tools/examples/google_serper.html
"""


def serpapi_search(query: str, api_key: str) -> str:
    """
    Performs a http request to do a Google search and returns a single string that answers the query
    This does spend free 1 search query from the SERP API
    """
    os.environ["SERPAPI_API_KEY"] = api_key
    search = SerpAPIWrapper()
    result = search.run(query)
    return result


def serpapi_get_results(query: str, api_key: str) -> dict:
    """
    Retrieve the raw SERP results from the SERP API for the Google Search query.
    This does NOT spend free 1 search query from the SERP API
    """
    os.environ["SERPAPI_API_KEY"] = api_key
    search = SerpAPIWrapper()
    results = search.results(query)
    return results


async def serpapi_get_aresults(query: str, api_key: str) -> dict:
    """
    Performs a http request to do a Google Search and returns the raw SERP results from
    the SERP API for the Google Search query.
    This does spend free 1 search query from the SERP API
    """
    os.environ["SERPAPI_API_KEY"] = api_key
    search = SerpAPIWrapper()
    raw_results = await search.aresults(query)
    return raw_results


def run_serpapi_example():
    # RETRIEVE RESULTS FROM SERP API
    serp_api_key = os.getenv("SERP_API_KEY")
    results = serpapi_get_results(query="What is LangChain?", api_key=serp_api_key)
    json_results = json.dumps(results, indent=2)
    print(json_results)
