from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
import datetime
import urllib
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from googlesearch import search
    
class SearchInput(BaseModel):
    """Input for web search tool."""

    query: str = Field(
        ...,
        description="Search query")
    
    result_message: str = Field(
        ...,
        description="Result messages that has been searched on internet")


class SearchInfoTool(BaseTool):
    name = "find_information_in_web"
    description = "Perform a web search on Google related to the key words and list key points and URLs."

    def _run(self, query: str, result_message: str):
        print("Search info: ", query)

        return result_message

    args_schema: Optional[Type[BaseModel]] = SearchInput

