from langchain.agents import Tool
from langchain.utilities.serpapi import SerpAPIWrapper


search = SerpAPIWrapper()

SearchTool = Tool(
    name="search",
    description="Scrape Google and other search engines from the fast, easy, and complete API."
                "The usage of this tool is limited to 100 searches per month."
                "Please limit your usage to avoid exceeding the limit.",
    func=search.run,
)

