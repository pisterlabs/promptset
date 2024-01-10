import logging
import datetime as dt
import openai
from googleapiclient.errors import HttpError
from google_calendar_integration import *


def analyze_weekly_data():
    creds = authorize_google_calendar()
    if not creds or not creds.valid:
        logging.error('Failed to obtain valid credentials for Google Calendar.')
        return

    service = build('calendar', 'v3', credentials=creds)
    start_datetime = get_current_datetime_in_local_timezone() - dt.timedelta(days=8)
    end_datetime = get_current_datetime_in_local_timezone() + dt.timedelta(hours=5)

    try:
        events_result = service.events().list(
            calendarId=CALENDAR_ID,
            timeMin=start_datetime.isoformat(),
            timeMax=end_datetime.isoformat(),
            maxResults=2500,  
            singleEvents=True,
            orderBy='startTime'
        ).execute()

        events = events_result.get('items', [])

        if not events:
            print('No events found for the past week.')
            return

        data = ''
        for event in events:
            summary = event.get('summary')
            if(summary == 'TeamNeuronX'):
                description = event.get('description')
                start_date = event.get('start').get('dateTime')
                data += f"Event Start Time: {start_date}\n"
                data += f"Event Description: {description}\n"

        prompt = f"Extract Area of improment for each person with names to perform better from the following event (list down area of improvement in seperate line):\n{data}\n"  
        response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=200,
        temperature=0.3,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
             )
        areaofimp = response.choices[0].text.strip().split("\n")
    
        return areaofimp
        
    except HttpError as err:
        logging.error('An error occurred while fetching events from Google Calendar: %s', err)


