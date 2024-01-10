from datetime import datetime
from typing import Any, List

import dateutil.parser as parser
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from langchain.tools import tool
from urlextract import URLExtract

from db import create_meeting
from oauth.google import get_google_oauth_creds
from tools.summarizer import summarize

## Google Calendar API 
## https://developers.google.com/calendar/api/v3/reference
url_extractor = URLExtract()
creds = get_google_oauth_creds()
if creds is None:
    raise Exception("No credentials found")

service = build('calendar', 'v3', credentials=creds)

@tool
def create_calendar_event(source_message_ts: str, title: str, description: str, start_time: datetime, end_time: datetime, invitees: List[str]) -> str:
    """
        Creates a Google Calendar event for the user with the given title, description, start time, and end time.
        It also invites the invitees (which is list of emails).
        Any URL present in the conversation should not be summarized, and should be included in the description as it is.
        Returns the event ID of the created event.
        source_message_ts: The timestamp of the source message from which the meeting was created.
    """
    print("Creating event")
    print(f"Title: {title}")
    print(f"Start time: {start_time.astimezone().isoformat()}")
    print(f"End time: {end_time.astimezone().isoformat()}")
    print(f"Invitees: {invitees}")

    try:
        attendees = [{'email': email} for email in invitees]
        event = {
            'summary': title,
            'description': description,
            'start': {
                'dateTime': start_time.astimezone().isoformat(),
            },
            'end': {
                'dateTime': end_time.astimezone().isoformat(),
            },
            'attendees': attendees,
            'reminders': {
                'useDefault': True,
            },
        }
        event = service.events().insert(calendarId='primary', body=event).execute()
        print(event)
        if event is None:
            raise Exception("Event creation failed")
        create_meeting(source_message_ts, event['id'], title, start_time.astimezone().isoformat(), end_time.astimezone().isoformat())
        return "Event successfully created. Do not create again. Stop now."

    except HttpError as error:
        return f"Event creation failed due to an error {error}"


def fetch_calendar_events() -> List[Any]:
    try:
        page_token = None
        today = datetime.utcnow().date().isoformat()
        events = []
        while True:
            events_result = service.events().list(
                calendarId='primary',
                timeMin=today + 'T00:00:00Z',
                timeMax=today + 'T23:59:59Z',
                singleEvents=True,
                orderBy='startTime',
                pageToken=page_token
            ).execute()
            events.extend(events_result.get('items', []))
            page_token = events_result.get('nextPageToken')
            if not page_token: break
        return events
    except HttpError as error:
        print(f"An error occurred: {error}")
        raise HttpError

@tool
def get_calendar_events() -> List[Any]:
    """
        Returns list of calendar events for the user for the current day.
    """
    events = fetch_calendar_events()
    if not events:
        return "No events found for today"
    else:
        returnVal = ""
        print('Events for today:')
        returnVal += 'Events for today:\n'
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            end = event['end'].get('dateTime', event['end'].get('date'))
            print(f"{start} - {end} - {event['summary']}")
            returnVal += f"{start} - {end} - {event['summary']}\n"
        return returnVal

def update_meeting_body(calendar_meeting_id: str, new_description: str) -> Any:
    # Retrieve the event
    event = service.events().get(calendarId='primary', eventId=calendar_meeting_id).execute()

    current_description = event['description']
    new_description = current_description + "\n" + new_description
    urls = url_extractor.find_urls(new_description)
    print(urls)
    print(f"Summarizing new description: {new_description}")
    new_description = summarize(new_description)
    # Remove duplicate urls
    urls = list(set(urls))
    new_description = new_description + "\n\n" + "\n".join(urls)
    # Update the description
    event['description'] = new_description
    print(f"Updating meeting {calendar_meeting_id} with new description {new_description}")

    # Update the event
    updated_event = service.events().update(calendarId='primary', eventId=calendar_meeting_id, body=event).execute()

    print("Event description updated.")
    return updated_event


@tool
def get_conflicting_meetings():
    """
        Returns list of conflicting meetings for the user for the current day, if any.
    """
    events = fetch_calendar_events()
    if not events:
        return "No events found for today"
    event_summary = []
    for event in events:
        start = event['start'].get('dateTime', event['start'].get('date'))
        start = parser.isoparse(start)
        end = event['end'].get('dateTime', event['end'].get('date'))
        end = parser.isoparse(end)
        event_summary = event_summary + [{
            'title': event['summary'],
            'start': start,
            'end': end,
        }]

    conflicts = []
    event_summary.sort(key=lambda x: x['start'])
    for i in range(len(event_summary) - 1):
        if event_summary[i]['end'] > event_summary[i + 1]['start']:
            conflicts = conflicts + [(event_summary[i], event_summary[i + 1])]
    
    if not conflicts:
        return "No conflicts found for today"
    else:
        result = "Conflicts found for today: \n"
        for e1, e2 in conflicts:
            edesc = lambda e : f"{e['title']} at {e['start'].strftime('%H:%M')}"
            result += f"{edesc(e1)} conflicts with {edesc(e2)}. \n"
        return result
