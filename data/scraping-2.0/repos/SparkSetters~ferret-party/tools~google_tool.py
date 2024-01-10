from langchain.agents import Tool
from langchain.utilities import GoogleSearchAPIWrapper

def get_search_tool():
    search_instance = GoogleSearchAPIWrapper()
    return Tool(
        name="Google Search",
        description="Search Google for recent results.",
        func=search_instance.run,
    )

