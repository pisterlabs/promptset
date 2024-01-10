import logging
from serpapi import GoogleSearch
import config

# Configure logging
logger = logging.getLogger(__name__)


class GoogleSearchManager:
    """
    A class to perform web searches and scrape web content.
    """

    def google_search(self, query, num_results=3, location="United States"):
        """
        Performs a Google Search and returns a list of URLs.
        """
        params = {
            "api_key": config.serpapi_key,
            "engine": "google",
            "q": query,
            "num": str(num_results),
            "tbm": "nws",  # Search type: news, images, videos, shopping, books, apps
            "location": location,
            "hl": "en",  # language
            "gl": "us",  # country code to search from (e.g. United States = us, Germany = de)
            "google_domain": "google.com",  # google domain to search from
            "output": "json",
            "safe": "active",
        }
        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            news_results = results.get("news_results", [])
            urls = [result["link"] for result in news_results]
            return urls
        except Exception as e:
            return f"Error in performing Google Search: {e}"


# Usage Example
if __name__ == "__main__":
    web = GoogleSearchManager()
    result = web.google_search("Is Sam Altman fired from OpenAI?")
    if result is not None:
        print(result)
    else:
        logger.error("Failed to process the search query.")
