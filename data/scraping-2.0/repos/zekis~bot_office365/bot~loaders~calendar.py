import sys
import traceback
import bot_config


from datetime import datetime, timedelta
import pytz
from typing import Any, Dict, Optional, Type

sys.path.append("/root/projects")
import common.bot_logging
from common.bot_comms import publish_event_card, publish_list, send_to_another_bot, send_to_user, send_to_me, publish_error
from common.bot_utils import tool_description, tool_error

from O365 import Account

from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain.tools import BaseTool


#common.bot_logging.bot_logger = common.bot_logging.logging.getLogger('ToolLogger')
#common.bot_logging.bot_logger.addHandler(common.bot_logging.file_handler)

def authenticate():
    credentials = (bot_config.APP_ID, bot_config.APP_PASSWORD)
    account = Account(credentials,auth_flow_type='credentials',tenant_id=bot_config.TENANT_ID, main_resource=bot_config.OFFICE_USER)
    account.authenticate()
    return account

def search_calendar(start_date, end_date):
    account = authenticate()
    schedule = account.schedule()
    calendar = schedule.get_default_calendar()
    #print(calendar.name)

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

sent_reminders = set()
"""loop to check for upcomming meetings and create journal entries"""
def check_for_upcomming_event():
    global sent_reminders
    
    timezone = pytz.timezone(bot_config.TIME_ZONE)
    start_date = datetime.now(timezone)
    end_date = datetime.now(timezone) + timedelta(hours=12)

    events = search_calendar(start_date, end_date)

    common.bot_logging.bot_logger.debug(f"Checking events between {start_date} and {end_date}")

    for event in events:
        reminder_time = event.start.astimezone(timezone) - timedelta(minutes=event.remind_before_minutes)
        #common.bot_logging.bot_logger.debug(f"Checking event {event.subject} starting {reminder_time}")
        if (reminder_time < datetime.now(timezone) and event.is_all_day == False and event.is_reminder_on and event.object_id not in sent_reminders):
            #str_attendees = ""
            str_attendees = get_string_from_list(event.attendees, 10)
            # for attendee in event.attendees:
            #     str_attendees = str(attendee) + "," + str_attendees
            str_location = event.location.get('displayName')
            if str_location == "":
                str_location = 'No location'
            if str_attendees:
                send_to_user(f"You have a meeting starting in {event.remind_before_minutes}min about {event.subject} with {event.organizer} and {str_attendees}. Let me search for any relevent emails")
                send_to_me(f"please find any emails relating to {event.subject}")
                send_to_another_bot('journal',f"add to my journal a heading and subheading of details to keep my notes for the meeting at {event.start.astimezone(timezone)} about {event.subject} in {str_location} with {event.organizer} and {str_attendees}")
            else:
                send_to_user(f"You have a meeting starting in {event.remind_before_minutes}min about {event.subject} with {event.organizer}. Let me search for any relevent emails")
                send_to_me(f"please find any emails relating to {event.subject}")
                send_to_another_bot('journal',f"add to my journal a heading and subheading of details to keep my notes for the meeting {event.start.astimezone(timezone)} about {event.subject} in {str_location} with {event.organizer}")
            sent_reminders.add(event.object_id)

"""This function trunctes lists as strings"""
def get_string_from_list(address_list, max_contacts):
    if len(address_list) > max_contacts:
        truncated_count = len(address_list) - max_contacts
        return ", ".join([addr.address for addr in address_list[:max_contacts]]) + f", ... {truncated_count} others"
    else:
        return ", ".join([addr.address for addr in address_list])



class MSGetCalendarEvents(BaseTool):
    parameters = []
    optional_parameters = []
    name = "GET_CALENDAR_EVENTS"
    summary = """Useful for when you need to retrieve meetings and appointments."""
    parameters.append({"name": "start_date", "description": "event start date time format as %Y-%m-%d %H:%M:%S"})
    parameters.append({"name": "end_date", "description": "event end date time format as %Y-%m-%d %H:%M:%S"})
    description = tool_description(name, summary, parameters, optional_parameters)
    return_direct = False

    def _run(self, start_date: str, end_date: str = None, publish: str = "True", run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        common.bot_logging.bot_logger.debug(f"Searching for events between {start_date} and {end_date}")
        try:
            ai_summary = ""
            human_summary = []
            if not end_date or start_date == end_date:
                start_date_format = '%Y-%m-%d'
                # Convert start_date string to datetime object
                start_date = datetime.strptime(start_date, start_date_format)
                # Add 12 hours to start_date
                end_date = start_date + timedelta(hours=24)
                # If you need end_date as a string in the same format
                end_date = end_date.strftime(start_date_format)

            events = search_calendar(start_date, end_date)
            
            if events:
                for event in events:
                    common.bot_logging.bot_logger.debug(event)
                    ai_summary = ai_summary + " - Event: " + event.subject + ", At " + event.start.strftime("%A, %B %d, %Y at %I:%M %p") + "\n"
                    title = event.subject + " - " + event.start.strftime("%A, %B %d, %Y at %I:%M %p")
                    value = "Please use the GET_CALENDAR_EVENT tool using ID: " + event.object_id
                    human_summary.append((title, value))
                
                if publish.lower() == "true":
                    title_message = f"Events Scheduled {start_date} - {end_date}"
                    publish_list(title_message, human_summary)
                    self.return_direct = True
                    return None
                else:
                    self.return_direct = False
                    return ai_summary
            
            else: 
                return "No Events"
            
        except Exception as e:
            #traceback.print_exc()
            tb = traceback.format_exc()
            publish_error(e, tb)
            return tool_error(e, tb, self.description)
    
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
                    self.return_direct = True
                    return None
                else:
                    self.return_direct = False
                    return ai_summary
            else:
                raise Exception(f"Could not find event {eventID}")
            
        except Exception as e:
            #traceback.print_exc()

            tb = traceback.format_exc()
            publish_error(e, tb)
            return tool_error(e, tb, self.description)
    
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
            send_to_another_bot('journal',f"Please add to my journal that I created {ai_summary}")
            if publish.lower() == "true":
                title_message = f"New Calander Event"
                publish_event_card(title_message, new_event)
                self.return_direct = True
                return None
            else:
                return_direct = False
                return ai_summary
            

        except Exception as e:
            #traceback.print_exc()
            tb = traceback.format_exc()
            publish_error(e, tb)
            return tool_error(e, tb, self.description)
    
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("CREATE_CALENDAR_EVENT does not support async")