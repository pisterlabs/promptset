from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
import datetime
import urllib
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from googlesearch import search
    
class SummarizeInput(BaseModel):
    """Input for Summarizing tool."""
    summary: str = Field(
        ...,
        description="Summary of the chat")

    

class SummarizeTool(BaseTool):
    name = "group_message_summarizer"
    description = f"Briefly summarize the latest input chat messages contents when seeing 'summary' in 200 Traditional Chinese words."

    def _run(self, summary: str):
        return summary

    args_schema: Optional[Type[BaseModel]] = SummarizeInput


