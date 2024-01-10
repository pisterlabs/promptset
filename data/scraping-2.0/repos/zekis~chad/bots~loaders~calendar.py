import traceback
import config

from datetime import datetime
from typing import Any, Dict, Optional, Type

from common.rabbit_comms import publish, publish_event_card, publish_list
from common.utils import tool_description, tool_error

from O365 import Account

from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain.tools import BaseTool

def authenticate():
    credentials = (config.APP_ID, config.APP_PASSWORD)
    account = Account(credentials,auth_flow_type='credentials',tenant_id=config.TENANT_ID, main_resource=config.OFFICE_USER)
    account.authenticate()
    return account

def search_calendar(start_date, end_date):
    account = authenticate()
    schedule = account.schedule()
    calendar = schedule.get_default_calendar()
    print(calendar.name)

    q = calendar.new_query('start').greater_equal(start_date)
    q.chain('and').on_attribute('end').less_equal(end_date)

    events = calendar.get_events(query=q, include_recurring=True)  # include_recurring=True will include repeated events on the result set.
    if events:
        return events
    return None

def get_event(eventID):
    account = authenticate()
    schedule = account.schedule()
    calendar = schedule.get_default_calendar()
    event = calendar.get_event(eventID)  # include_recurring=True will include repeated events on the result set.
    return event


class MSGetCalendarEvents(BaseTool):
    parameters = []
    optional_parameters = []
    name = "GET_CALENDAR_EVENTS"
    summary = """Useful for when you need to retrieve meetings and appointments."""
    parameters.append({"name": "start_date", "description": "date"})
    parameters.append({"name": "end_date", "description": "date"})
    description = tool_description(name, summary, parameters, optional_parameters)
    return_direct = False

    def _run(self, start_date: str, end_date: str, publish: str = "True", run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            ai_summary = ""
            human_summary = []

            events = search_calendar(start_date, end_date)
            if events:
                for event in events:
                    ai_summary = ai_summary + " - Event: " + event.subject + ", At " + event.start.strftime("%A, %B %d, %Y at %I:%M %p") + "\n"
                    title = event.subject + " - " + event.start.strftime("%A, %B %d, %Y at %I:%M %p")
                    value = "Please use the GET_CALENDAR_EVENT tool using ID: " + event.object_id
                    human_summary.append((title, value))
                
                if publish.lower() == "true":
                    title_message = f"Events Scheduled {start_date} - {end_date}"
                    publish_list(title_message, human_summary)
                    return config.PROMPT_PUBLISH_TRUE
                else:
                    return ai_summary
            
            else: 
                return "No Events"
            
        except Exception as e:
            traceback.print_exc()
            return tool_error(e, self.description)
    
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("GET_CALENDAR_EVENTS does not support async")

class MSGetCalendarEvent(BaseTool):
    parameters = []
    optional_parameters = []
    name = "GET_CALENDAR_EVENT"
    summary = "Useful for when you need to retrieve a single calander meeting or appointment"
    parameters.append({"name": "eventID", "description": "unique event ID"})
    description = tool_description(name, summary, parameters, optional_parameters)
    return_direct = False

    def _run(self, eventID: str, publish: str = "True", run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            ai_summary = ""
            human_summary = []

            event = get_event(eventID)
            if event:
                ai_summary = ai_summary + " - Event: " + event.subject + ", At " + event.start.strftime("%A, %B %d, %Y at %I:%M %p") + "\n"

                if publish.lower() == "true":
                    title_message = f"Event Review"
                    publish_event_card(title_message, event)
                    return config.PROMPT_PUBLISH_TRUE
                else:
                    return ai_summary
            else:
                raise Exception(f"Could not find event {eventID}")
            
        except Exception as e:
            traceback.print_exc()
            return tool_error(e, self.description)
    
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("GET_CALENDAR_EVENTS does not support async")

class MSCreateCalendarEvent(BaseTool):
    parameters = []
    optional_parameters = []
    name = "CREATE_CALENDAR_EVENT"
    summary = """useful for when you need to create a meetings or appointment in the humans calander"""
    parameters.append({"name": "subject", "description": "email subject"})
    parameters.append({"name": "start_datetime", "description": "event start date time format as %Y-%m-%d %H:%M:%S"})
    optional_parameters.append({"name": "end_datetime", "description": "event end date time format as %Y-%m-%d %H:%M:%S"})
    optional_parameters.append({"name": "is_all_day", "description": "set True to make all day event"})
    optional_parameters.append({"name": "remind_before_minutes", "description": "set reminder notice"})
    description = tool_description(name, summary, parameters, optional_parameters)
    return_direct = False

    def _run(self, subject, start_datetime: str, end_datetime: str = None, is_all_day: str = None, location: str = None, remind_before_minutes: str = None, publish: str = "True", run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            
            ai_summary = ""
            human_summary = []

            account = authenticate()
            schedule = account.schedule()
            calendar = schedule.get_default_calendar()
            new_event = calendar.new_event()

            format = "%Y-%m-%d %H:%M:%S"
            formatted_start_datetime = datetime.strptime(start_datetime, format)
            formatted_end_datetime = datetime.strptime(end_datetime, format)

            new_event.subject = subject
            new_event.start = formatted_start_datetime
            if is_all_day:
                new_event.is_all_day = is_all_day
            else:
                new_event.end = formatted_end_datetime
            
            
            if location:
                new_event.location = location
            
            if remind_before_minutes:
                new_event.remind_before_minutes = remind_before_minutes

            new_event.save()    
        
            ai_summary = "New Calander Event: " + new_event.subject + ", At " + new_event.start.strftime("%A, %B %d, %Y at %I:%M %p") + "\n"

            if publish.lower() == "true":
                title_message = f"New Calander Event"
                publish_event_card(title_message, new_event)
                return config.PROMPT_PUBLISH_TRUE
            else:
                return ai_summary
            

        except Exception as e:
            traceback.print_exc()
            return tool_error(e, self.description)
    
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("CREATE_CALENDAR_EVENT does not support async")