from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict

class Result(BaseModel):
    name: str = Field(description="The specific action or method that you propose to execute based on your analysis.")
    answer: str = Field(description="The answer you arrived at through your reasoning, this must be a scalar string value.")
    next_action: str = Field(description="The next action you propose to take based on your analysis. An example would be asking the user if they want an event scheduled or updated. If there isn't a next action, just set this field to None.")


class Thoughts(BaseModel):
    text: str = Field(description="Provide a brief overview of the current situation or context based on your analysis of the calendar events.")
    reasoning: str = Field(description="Explain the logic behind your analysis and the conclusions you've reached. This includes any patterns or conflicts you've identified, as well as the criteria you used to determine the best date/time for the event.")
    plan: str = Field(description="Short description of your strategy for proceeding with the task, including any next steps or considerations for future actions.")
    criticism: str = Field(description="Identify any limitations or potential issues with your analysis. Provide constructive self-criticism.")
    speak: str = Field(description="Summarize your thoughts in a brief, conversational statement that can be presented to the user.")

class Analysis(BaseModel):
    events_dates_comparison: str = Field(description="Understand and process the date and time information of each event. Compare these to identify the events that occur first, last, or simultaneously. ")
    event_conflict_detection: str = Field(description="Examine all events within the desired time frame to see if there are any overlaps. Conflicting events are those that have overlapping time periods. ")
    available_time_slot_detection: str = Field(description="Identify periods within the desired date range when there are no events scheduled. These are potential slots for scheduling new events.")
    best_date_time_selection: str = Field(description="Based on the user's requirements and your analysis, determine the best date and time to schedule a new event. The best date/time should meet the user's requirements and avoid any event conflicts.")
    answer: str = Field(description="The answer you arrived at through your reasoning, this must be a scalar string value.")

class Command(BaseModel):
    name: str = Field(description="The name of the command you want to execute.")
    args: Dict[str,str] = Field(description="The arguments to pass to the command, expressed as key value pairs. If there are no arguments, just set this field to an empty JSON object.")

class CalendarAnalysisResponse(BaseModel):
    thoughts: Thoughts = Field(description="An object containing your thoughts and reasoning behind your analysis.")
    analysis: Analysis = Field(description="An object containing the actions you took to analyze the calendar events.")
    command: Command = Field(description="An object containing the command you want to execute and the arguments to pass to it.")

class CalendarAnalyzerOutputParserFactory:
    """Factory class for creating an output parser for the Calendar Analyzer Chain"""
    @staticmethod
    def create_output_parser():
        return PydanticOutputParser(pydantic_object=CalendarAnalysisResponse)
