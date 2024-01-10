# The instruction is taken from here:
# https://python.langchain.com/docs/integrations/tools/google_search.html
import os

# from haystack.nodes.search_engine import WebSearc
SEARCH_ENGINE_ID=#<YOUR-ENGINE-ID>
GOOGLE_API_KEY=#<YOUR-API-KEY>

os.environ["GOOGLE_CSE_ID"] = SEARCH_ENGINE_ID
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper

search = GoogleSearchAPIWrapper()

tool = Tool(
    name="Google Search",
    description="Search Google for recent results.",
    func=search.run,
)

result = tool.run("Who did discover electron first time?")

print(result)
