from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
import datetime
import urllib
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from googlesearch import search
    
class MeetingInput(BaseModel):
    """Input for meeting time generator."""
    meeting_period: str = Field(
        ...,
        description="The time that user need to hold a meeting."
    )
    meeting_time: str = Field(
        ...,
        description="The time period and the user specify whether he/she is available or not.")

    

class MeetingTool(BaseTool):
    name = "group_meeting_time_generator"
    description = f"""Arange the meeting time.
    Give all the specific possible meeting times that user can have a meeting, consider the available time of all group members when seeing 'meeting time summary'.
    """

    def _run(self, meeting_period: str, meeting_time: str):
        print('meeting_time:', meeting_time)
        return meeting_time

    args_schema: Optional[Type[BaseModel]] = MeetingInput


