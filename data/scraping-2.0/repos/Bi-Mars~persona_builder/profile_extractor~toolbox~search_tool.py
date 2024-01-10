from langchain.utilities.serpapi import SerpAPIWrapper
from dotenv import (
    load_dotenv,
    find_dotenv,
)  # These imports are used to find .env file and load them

load_dotenv(find_dotenv())

""" 
    1. Leverage SerpApi: A Wrapper around Google Search API.
    2. Provide the query and the SerpApi will seach that on google and provide the result result in JSON format
"""


def get_profile_url(text: str) -> str:
    """
    This tool can be used to search for the linkedin profile page.
    """

    """ Step-1: LangChain provides wrapper around SerpApi"""
    search = SerpAPIWrapper()

    result = search.run(f"{text}")
    return result