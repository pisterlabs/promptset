import os
import json
import datetime
import time 
from datetime import  timedelta
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from supabase import create_client
from dotenv import load_dotenv
load_dotenv()
# def get_calendar_events(email):
#     load_dotenv()

#     supabase_url = os.getenv('SUPABASE_URL')
#     supabase_key = os.getenv('SUPABASE_KEY')
#     supabase_client = create_client(supabase_url, supabase_key)

#     access_token = supabase_client.table('slack_app').select('accesstoken').eq('email', email).single().execute()
#     access_token_data = access_token.data  # Extract the JSON data
#     token_data = json.loads(access_token_data['accesstoken'])
#     credentials = Credentials.from_authorized_user_info(token_data)
#     service = build('calendar', 'v3', credentials=credentials)

#     # Call the Calendar API
#     now = datetime.datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
#     print('Getting the upcoming 10 events')
#     events_list = []
#     try:
#         events_result = service.events().list(calendarId='primary', timeMin=now,
#                                           maxResults=10, singleEvents=True,
#                                           orderBy='startTime').execute()
#         events = events_result.get('items', [])

#         if not events:
#             return 'No upcoming events found.'
#         else:
#             for event in events:
#                 start = event['start'].get('dateTime', event['start'].get('date'))
#                 summary = event['summary']
#                 events_list.append((start, summary))
                
#         return events_list
#     except Exception as error:
#         return f'An error occurred: {error}'

# start_time = time.time()
# events_list = get_calendar_events("keviinkibe@gmail.com")
# print(events_list)
# end_time = time.time()
# duration = end_time - start_time
# print(duration)

def schedule_event(summary, start_time, end_time, attendees):
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_KEY')
    supabase_client = create_client(supabase_url, supabase_key)
    email = "keviinkibe@gmail.com"
    access_token = supabase_client.table('slack_app').select('accesstoken').eq('email', email).single().execute()
    access_token_data = access_token.data  
    token_data = json.loads(access_token_data['accesstoken'])
    credentials = Credentials.from_authorized_user_info(token_data)
    service = build('calendar', 'v3', credentials=credentials)

    # Define the event
    event = {
        'summary': summary,
        'start': {
            'dateTime': start_time.isoformat(),
            'timeZone': 'Africa/Nairobi',
        },
        'end': {
            'dateTime': end_time.isoformat(),
            'timeZone': 'Africa/Nairobi',
        },
        'attendees': [{'email': attendee} for attendee in attendees],
    }

    # Call the Calendar API to create the event
    event = service.events().insert(calendarId='primary', body=event).execute()

    return print(f"Event created: {event['htmlLink']}")


start_time = datetime.datetime(year=2023, month=10, day=27, hour=21)
attendees = ['kchegz234@gmail.com', 'nawariholdings@gmail.com']
end_time = start_time + timedelta(hours=1)
schedule_event('My Future Event', start_time, end_time, attendees)

# import pytz
# import datetime
# nairobi = pytz.timezone('Africa/Nairobi')
# nairobi_time = datetime.datetime.now(nairobi)
# # nairobi_time= nairobi_time.isoformat()
# # now = datetime.datetime.utcnow().isoformat() +'Z'
# print(nairobi_time)

























# import os
# from dotenv import load_dotenv
# from langchain.agents import initialize_agent
# from langchain.chat_models import ChatOpenAI
# from langchain.chains.conversation.memory import ConversationBufferWindowMemory
# from agent_tools import DataFetchingTool
# from langchain.schema.messages import HumanMessage, AIMessage


# llm = ChatOpenAI(
#     openai_api_key="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",
#     temperature=0,
#     model_name='gpt-3.5-turbo'
# )
# # initialize conversational memory
# conversational_memory = ConversationBufferWindowMemory(
#     memory_key='chat_history',
#     k=5,
#     return_messages=True,
#     message_class=HumanMessage,
# )

# tools = [DataFetchingTool()]


# # initialize agent with tools
# agent = initialize_agent(
#     agent='chat-conversational-react-description',
#     tools=tools,
#     llm=llm,
#     verbose=True,
#     max_iterations=3,
#     early_stopping_method='generate',
#     memory=conversational_memory,
#     return_messages = True
# )
# response =agent("how much did the company make?") 
# print(response['chat_history'])
# print(response['output'])