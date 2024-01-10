from typing import Optional, Type, Union
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from gcsa.google_calendar import GoogleCalendar
from gcsa.recurrence import Recurrence, YEARLY, DAILY, WEEKLY, MONTHLY
from gcsa.event import Event
from os import environ
from datetime import date, datetime
from langchain.agents import AgentType, initialize_agent
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain import OpenAI, LLMChain
from dotenv import load_dotenv
import agent_prompt
from langchain.llms import OpenAI
from langchain.llms.openai import OpenAI
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.agents import ZeroShotAgent
from langchain.memory import ConversationBufferMemory
from langchain.agents.initialize import initialize_agent
from typing import List, Union
from dateutil import tz
import os


load_dotenv()

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from enum import Enum, auto
from typing import Optional

alt_date_time_format = '%Y-%m-%d %H:%M:%S'
alt_date_time_format = "%Y-%m-%d %H:%M:%S"
alt_date_time_format = "%Y-%m-%d %H:%M"


class Recurrence(Enum):
    DAILY = auto()
    WEEKLY = auto()
    MONTHLY = auto()
    YEARLY = auto()

# def filter_events(events: List[Event], start: str, end: str) -> List[Event]:
#     start_dt = datetime.fromisoformat(start)
#     end_dt = datetime.fromisoformat(end)
#     filtered_events = [event for event in events if start_dt <= event.datetime < end_dt]
#     return filtered_events

# def filter_events_from_text(event_strs: List[str], start: str, end: str) -> List[str]:
#     start_dt = datetime.fromisoformat(start)
#     end_dt = datetime.fromisoformat(end)
    
#     filtered_events = []
#     for event_str in event_strs:
#         datetime_str, _ = event_str.split(' - ', 1)
#         event_dt = datetime.fromisoformat(datetime_str)
        
#         if start_dt <= event_dt < end_dt:
#             filtered_events.append(event_str)
            
#     return filtered_events

def filter_events(events: List[str], start: str, end: str) -> List[str]:
    # Create datetime objects for start and end times, converting to the local time zone
    start_dt = datetime.fromisoformat(start).replace(tzinfo=tz.tzutc()).astimezone(tz.tzlocal())
    end_dt = datetime.fromisoformat(end).replace(tzinfo=tz.tzutc()).astimezone(tz.tzlocal())
    
    filtered_events = []
    for event_str in events:
        try:
            # Split the string into datetime and description
            datetime_str, _ = event_str.split(' - ', 1)
            event_dt = datetime.fromisoformat(datetime_str)
            
            # If the event_dt is naive (has no timezone info), we assume it is in UTC
            if event_dt.tzinfo is None:
                event_dt = event_dt.replace(tzinfo=tz.tzutc())
            
            # Convert to local time for comparison
            event_dt = event_dt.astimezone(tz.tzlocal())
            
            # Perform the comparison and append to the list if the condition is met
            if start_dt <= event_dt < end_dt:
                filtered_events.append(event_str)
        except Exception as e:
            print(f"An error occurred while processing event: {event_str}. Error: {e}")

    return filtered_events


# def filter_events_from_text(events: List[Event], start: str, end: str) -> List[Event]:
#     # Convert start and end to time zone aware datetimes
#     start_dt = datetime.fromisoformat(start).replace(tzinfo=tz.tzutc()).astimezone(tz.tzlocal())
#     end_dt = datetime.fromisoformat(end).replace(tzinfo=tz.tzutc()).astimezone(tz.tzlocal())
    
#     return [event for event in events if start_dt <= event.datetime < end_dt]


class CalenderCreateEventTool(BaseTool):
    """A tool used to create events on google calendar."""
    name = "create google event"
    description = "a tool used to create events on google calendar"

    def _run(
        self,
        summary: str,
        start: datetime,
        end: Union[datetime, None],
        recurrence: Optional[Recurrence] = None,  # Changed from Optional[Recurrence] to Optional[str]
        run_manager: Optional['CallbackManagerForToolRun'] = None,
        current_datetime: datetime = datetime.strptime(str((datetime.now()))[:16], alt_date_time_format),
    ) -> str:


        # ddd = datetime.strptime(str(datetime.now()), alt_date_time_format)
        GOOGLE_EMAIL = environ.get('GOOGLE_CALENDER_EMAIL')
        GMAIL_CREDENTIALS_PATH = environ.get('GMAIL_CREDENTIALS_PATH')

        # calendar = GoogleCalendar(
        #     GOOGLE_EMAIL, 
        #     save_token=False,
        #     GMAIL_CREDENTIALS_PATH=GMAIL_CREDENTIALS_PATH
        #     )

        calendar = GoogleCalendar(GOOGLE_EMAIL,credentials_path=GMAIL_CREDENTIALS_PATH)
        

        date_time_format = '%Y-%m-%dT%H:%M:%S'

        event = Event(
            summary=summary, 
            start=datetime.strptime(start,date_time_format), 
            end=datetime.strptime(end,date_time_format)
            )
        calendar.add_event(event)
        
    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
    

class CalenderViewEventTool(BaseTool):
    """A tool used to create events on google calendar."""
    name = "view google events"
    description = "a tool used to view events on google calendar "

    def _run(
        self,
        # summary: str,
        start: Union[datetime, None],
        end: Union[datetime, None],
        # recurrence: Optional[Recurrence] = None,  # Changed from Optional[Recurrence] to Optional[str]
        # run_manager: Optional['CallbackManagerForToolRun'] = None,
         current_datetime: datetime = datetime.strptime(str((datetime.now()))[:16], alt_date_time_format),
    ) -> str:


        # ddd = datetime.strptime(str(datetime.now()), alt_date_time_format)
        GOOGLE_EMAIL = environ.get('GOOGLE_CALENDER_EMAIL')
        GMAIL_CREDENTIALS_PATH = environ.get('GMAIL_CREDENTIALS_PATH')

        calendar = GoogleCalendar(
            GOOGLE_EMAIL, 
            GMAIL_CREDENTIALS_PATH=GMAIL_CREDENTIALS_PATH
            )
        date_time_format = '%Y-%m-%dT%H:%M:%S'

        # event = Event(
        #     summary=summary, 
        #     start=datetime.strptime(start,date_time_format), 
        #     end=datetime.strptime(end,date_time_format)
        #     )
        # calendar.add_event(event)

        
        calendar = GoogleCalendar(
            GOOGLE_EMAIL, 
            GMAIL_CREDENTIALS_PATH=GMAIL_CREDENTIALS_PATH
            )

        # calendar = GoogleCalendar(GOOGLE_EMAIL)

        
        for event in calendar:
            print(event)

        events = [str(event) for event in calendar]

        # events = filter_events(events, start, end)
        # events = filter_events_from_text(events, start, end)

        events = filter_events(events, start, end)
        
        return events
        
    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
    

# print("hi")
# print(os.environ)
# print(os.getenv('GMAIL_CREDENTIALS_PATH'))
# open(os.getenv('GMAIL_CREDENTIALS_PATH'), 'r').read()

# if __name__ == "__main__":


# tools = [
#     CalenderCreateEventTool(),
#         CalenderViewEventTool()
# ]


# prefix = agent_prompt.prefix
# suffix = """Begin!"

# {chat_history}
# Question: {input}
# {agent_scratchpad}"""

# prompt = ZeroShotAgent.create_prompt(
#     prefix=prefix,
#     suffix=suffix,
#     input_variables=["input", "chat_history", "agent_scratchpad"],
#     tools=tools,
# )

# memory = ConversationBufferMemory(memory_key="chat_history")


# llm = OpenAI(temperature=0)

# agent_executor = initialize_agent(
#     tools,
#     llm,
#     agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True,
#     prompt=prompt,
# )


# agent_executor.run("set up an weekly event for 5pm where i jog for an hour")

# agent_executor.run("set up an event for 5pm where i jog for an hour today")

# agent_executor.run("set an event for me to jog 2 pm every day this week")

# agent_executor.run("set an event for me to run 3 pm every thursday for a month")

# agent_executor.run("Do i have any events tomorrow?")

# dd = os.environ['GMAIL_CREDENTIALS_PATH'] = environ.get('GMAIL_CREDENTIALS_PATH')
# print(dd)

# agent_executor.run("If i have an event today to go jogging. I want to go to the bar instead")