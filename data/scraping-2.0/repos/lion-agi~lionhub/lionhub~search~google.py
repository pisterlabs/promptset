import os
from typing import Dict, List, Any, Dict
from lionagi import lcall
from ..utils import get_url_content, get_url_response

google_key_scheme = 'GOOGLE_API_KEY'
google_engine_scheme = 'GOOGLE_CSE_ID'

google_api_key = os.getenv(google_key_scheme)
google_engine = os.getenv(google_engine_scheme)

class GoogleSearch:
    api_key = google_api_key
    search_engine = google_engine
    search_url = (
        """
        https://www.googleapis.com/customsearch/v1?key={key}&cx={engine}&q={query}&start={start}
        """
        )

    # get fields of a google search item 
    @classmethod
    def _get_search_item_field(cls, item: Dict[str, Any]) -> Dict[str, str]:
        try:
            long_description = item["pagemap"]["metatags"][0]["og:description"]
        except KeyError:
            long_description = "N/A"
        url = item.get("link")
        
        return {
            "title": item.get("title"),
            "snippet": item.get("snippet"),
            "url": item.get("link"),
            "long_description": long_description,
            "content": get_url_content(url)
        }

    @classmethod
    def _format_search_url(cls, url, api_key, search_engine, query, start):
        url = url or cls.search_url
        url = url.format(
            key=api_key or cls.api_key, 
            engine=search_engine or cls.search_engine, 
            query=query, 
            start=start
        )
        return url
    
    @classmethod
    def search(
        cls, 
        query: str =None, 
        search_url = None,
        api_key = None,
        search_engine=None,
        start: int = 1, 
        timeout: tuple = (0.5, 0.5), 
        content=True,
        num=5
        ):
        url = cls._format_search_url(
            url = search_url, query=query, api_key=api_key,
            search_engine=search_engine, start=start
            )
        response = get_url_response(url, timeout=timeout)
        response_dict = response.json()
        items = response_dict.get('items')[:num]
        if content:
            items = lcall(items, cls._get_search_item_field, dropna=True)
        return items

    @classmethod
    def create_agent_engine(cls):
        try:
            from llama_index.agent import OpenAIAgent
            from llama_index.tools.tool_spec.load_and_search.base import LoadAndSearchToolSpec
            from llama_hub.tools.google_search.base import GoogleSearchToolSpec

            google_spec = GoogleSearchToolSpec(key=cls.api_key, engine=cls.engine_id)

            # Wrap the google search tool as it returns large payloads
            tools = LoadAndSearchToolSpec.from_defaults(
                google_spec.to_tool_list()[0],
            ).to_tool_list()

            # Create the Agent with our tools
            agent = OpenAIAgent.from_tools(tools, verbose=True)
            return agent.achat
        
        except Exception as e:
            raise ImportError(f"Error in importing OpenAIAgent from llama_index: {e}")