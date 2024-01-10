import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient import discovery as google_services
from datetime import datetime, timedelta
from typing import List, Any, Dict
import json
import base64
from typing import Tuple
import pytz

from googleapiclient.errors import HttpError
from collections import defaultdict
from datetime import datetime
from langchain.chains import AnalyzeDocumentChain
from langchain.chains.question_answering import load_qa_chain
from langchain.base_language import BaseLanguageModel
from langchain.tools import BaseTool

# The scope depends on what you want to access. For read-only access to Calendar and Gmail:
SCOPES = ['https://www.googleapis.com/auth/calendar.readonly',
          'https://www.googleapis.com/auth/gmail.readonly']

CREDENTIALS_FILE = 'oauth_secret.json'


def authenticate_google_api():
    creds: Credentials = None

    # The file token.json stores the user's access and refresh tokens.
    # It is created automatically when the authorization flow completes for the first time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    # If there are no (valid) credentials available, prompt the user to log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=18080)

        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return creds


def get_user_email(user_id='me'):
    credentials = authenticate_google_api()
    service = google_services.build('gmail', 'v1', credentials=credentials)
    profile = service.users().getProfile(userId=user_id).execute()
    return profile.get('emailAddress')


def get_message_body(service, user_id, message_id) -> Tuple[str, str]:
    """
    Fetches and decodes the email body (both plain text and HTML) from a given message.

    Args:
    service: The Gmail API service instance.
    user_id: User's email address or 'me' for the authenticated user.
    message_id: The ID of the message to fetch.

    Returns:
    A tuple containing the plain text body and HTML body of the email.
    """
    try:
        message = service.users().messages().get(userId=user_id, id=message_id, format='full').execute()
        payload = message.get('payload', {})
        parts = payload.get('parts', [])
        text_plain = ""
        text_html = ""

        if 'body' in payload and 'data' in payload['body']:
            # Handling non-multipart (simple) messages
            text_plain = base64.urlsafe_b64decode(payload['body']['data'].encode('ASCII')).decode('utf-8')
        elif parts:
            # Handling multipart messages
            for part in parts:
                part_body = part.get('body', {})
                part_data = part_body.get('data')
                part_headers = part.get('headers', [])
                part_mime_type = part.get('mimeType', '')

                if part_data is not None:
                    decoded_part = base64.urlsafe_b64decode(part_data.encode('ASCII')).decode('utf-8')

                    if part_mime_type == 'text/plain':
                        text_plain = decoded_part
                    elif part_mime_type == 'text/html':
                        text_html = decoded_part

        return text_plain, text_html
    except Exception as error:
        print(f'An error occurred: {error}')
        return "", ""


def count_reply_recipients(max_pages: int, user_id='me') -> Dict[str, int]:
    """
    Counts the number of times each email address has been replied to by the user.
    @param max_pages:
    @param user_id:
    @return:
    """
    user_email = get_user_email(user_id)
    credentials = authenticate_google_api()
    service = google_services.build('gmail', 'v1', credentials=credentials)
    reply_count = defaultdict(int)
    response = service.users().messages().list(userId=user_id, q="from:me").execute()
    pages_processed = 0
    oldest_date = datetime.now().replace(tzinfo=pytz.UTC)
    while pages_processed < max_pages:
        for message in response.get('messages', []):
            msg_details = service.users().messages().get(userId=user_id, id=message['id']).execute()
            headers = msg_details.get('payload', {}).get('headers', [])
            is_reply = any(header for header in headers if header['name'] in ['In-Reply-To', 'References'])
            from_header = next((header for header in headers if header['name'] == 'From'), None)
            datestr = next((header for header in headers if header['name'] == 'Date'), None)
            datestr = datestr['value'].split(' (')[0]
            if ', ' in datestr:
                datestr = datestr.split(', ')[1]
            date = datetime.strptime(datestr, "%d %b %Y %H:%M:%S %z")
            if date < oldest_date:
                oldest_date = date
            if is_reply and from_header and user_email in from_header['value']:
                to_header = next((header for header in headers if header['name'] == 'To'), None)
                if to_header:
                    email_addresses = to_header['value'].split(',')
                    for email in email_addresses:
                        reply_count[email.strip()] += 1
        if 'nextPageToken' not in response:
            break
        page_token = response['nextPageToken']
        response = service.users().messages().list(userId=user_id, pageToken=page_token).execute()
        print("Processed page " + str(pages_processed) + " of " + str(max_pages) +
              " pages. Oldest date: " + str(oldest_date))
        pages_processed += 1
    return reply_count


class GmailTool(BaseTool):
    name = "Gmail"
    description = "read-only access to Gmail, with the ability to filter by label set, and max_results, and search"
    "by search queries such as 'from:someone@example.com' or 'subject:\"Important\"' or"
    " 'after:2021/01/01' (or boolean combinations of queries like these)"
    chain: AnalyzeDocumentChain

    @staticmethod
    def from_llm(llm: BaseLanguageModel):
        qa_chain = load_qa_chain(llm, chain_type="map_reduce")
        chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)
        return GmailTool(chain=chain)

    @staticmethod
    def _fetch_messages(query: str,
                        label_ids: List[str] = ['INBOX'],
                        max_results: int = 10) -> List[Dict[str, Any]]:
        credentials = authenticate_google_api()
        service = google_services.build('gmail', 'v1', credentials=credentials)
        response = service.users().messages().list(
            userId='me', q=query, labelIds=label_ids, maxResults=max_results
        ).execute()

        messages = response.get('messages', [])
        detailed_messages = []

        for message in messages:
            try:
                msg = service.users().messages().get(userId='me', id=message['id'], format='full').execute()

                headers = msg['payload']['headers']
                subject = next(header['value'] for header in headers if header['name'] == 'Subject')
                from_email = next(header['value'] for header in headers if header['name'] == 'From')

                # Extract the plaintext body
                parts = msg['payload'].get('parts', [])
                body = None
                for part in parts:
                    if part['mimeType'] == 'text/plain':
                        body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                detailed_messages.append({
                    'subject': subject,
                    'from_email': from_email,
                    'body': body
                })
            except HttpError as error:
                print(f'An error occurred: {error}')

        return detailed_messages

    def _summarize(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for message in messages:
            message['summary'] = self.chain({"input_document": message['body'],
                                             "question": "Summarize the following:"})['output_text']
        return messages

    def _run(self, query: str, label_ids: List[str] = ['INBOX'], max_results: int = 10) -> List[Dict[str, Any]]:
        messages = self._summarize(GmailTool._fetch_messages(query, label_ids, max_results))
        return messages

    async def _arun(self, query: str, label_ids: List[str] = ['INBOX'], max_results: int = 10) -> List[Dict[str, Any]]:
        return self._run(query, label_ids, max_results)


def get_calendar_events(time_min: str = None,
                        time_max: str = None,
                        max_results: int = 10,
                        fields: str = "*",
                        calendar_id: str = 'primary'):
    """
    Returns a list of events on the user's primary calendar, starting from the (current time + hours_from_now)
    and going forward, up to a total of max_results events.
    :param time_min: time-range filter for events to return. Must be an RFC3339 timestamp with mandatory time zone
     offset, for example, 2011-06-03T10:00:00-07:00
    :param time_max: time-range filter for events to return. Must be an RFC3339 timestamp with mandatory time zone
     offset, for example, 2011-06-03T10:00:00-07:00
    :param max_results: the maximum number of results to return
    :param fields: the fields to return for each event
    :param calendar_id: the calendar ID to search
    :return: a list of events
    """
    credentials = authenticate_google_api()
    service = google_services.build('calendar', 'v3', credentials=credentials)
    time_min = time_min or datetime.utcnow().isoformat() + 'Z'
    results = service.events().list(
        calendarId=calendar_id,
        timeMin=time_min,
        timeMax=time_max,
        maxResults=max_results,
        singleEvents=True,
        orderBy='startTime',
        fields=fields).execute()
    return results.get('items', [])


def test_google_calendar_api():
    credentials = authenticate_google_api()
    service = google_services.build('calendar', 'v3', credentials=credentials)
    # Get the current time in RFC3339 format
    now = datetime.utcnow().isoformat() + 'Z'
    # Call the Calendar API
    print('Getting the upcoming 10 events')
    events_result = service.events().list(calendarId='primary', timeMin=now,
                                          maxResults=10, singleEvents=True,
                                          orderBy='startTime', fields="*").execute()
    events = events_result.get('items', [])

    if not events:
        print('No upcoming events found.')
        return

    for event in events:
        print(event['summary'], json.dumps(event, indent=2))


def test_gmail_api():
    credentials = authenticate_google_api()
    service = google_services.build('gmail', 'v1', credentials=credentials)
    # Call the Gmail API
    print('Listing the recent 10 emails')
    results = service.users().messages().list(userId='me', maxResults=10, labelIds=['INBOX'], fields="*").execute()
    messages = results.get('messages', [])

    if not messages:
        print('No messages found.')
        return

    for message in messages:
        msg = service.users().messages().get(userId='me', id=message['id']).execute()
        headers = msg['payload']['headers']
        subject = next(header['value'] for header in headers if header['name'].lower() == 'subject')
        from_email = next(header['value'] for header in headers if header['name'].lower() == 'from')
        # fix up some duplicate work here
        message_body = get_message_body(service, 'me', message['id'])

        print(f"From: {from_email}, Subject: {subject}\nMessage:\n" + message_body[0] + "\n\n")


# Test the Calendar API
# test_google_calendar_api()

# Test the Gmail API
# test_gmail_api()
