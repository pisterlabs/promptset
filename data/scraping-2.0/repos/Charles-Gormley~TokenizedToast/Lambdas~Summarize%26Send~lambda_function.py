import openai
import idna
import logging

import base64
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import os
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import boto3

from email.mime.text import MIMEText
from article_creation import create_content, create_email_html

import json
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_gmail_credentials():
    creds = None

    SCOPES = ['https://www.googleapis.com/auth/gmail.send']

    if 'AWS_EXECUTION_ENV' in os.environ:
        s3_client = boto3.client('s3')
        bucket = 'toast-summarizer'
        fn = 'token.json'
        fn_path = f'/tmp/{fn}'
        s3_client.download_file(bucket, fn, fn_path)
        # Grab token.json from s3 bucket --> /tmp/token.json.

    if os.path.exists(fn_path):
        logging.info("Found token json file")
        creds = Credentials.from_authorized_user_file(fn_path)
    if not creds or not creds.valid:
        logging.info("Invalid Credentials")
        if creds and creds.expired and creds.refresh_token:
            logging.info("Refreshing Credentials")
            creds.refresh(Request())
        else:
            logging.info("Grabbing creds from credentials json")
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(fn_path, 'w') as token:
            logging.info("writing creds to token.json.")
            token.write(creds.to_json())
    return creds

def send_email(subject:str, body:str, recipient:str):
    creds = get_gmail_credentials()

    service = build('gmail', 'v1', credentials=creds)

    html_content = body
    message = MIMEText(html_content, 'html')
    message['to'] = recipient
    message['subject'] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
    body = {'raw': raw}
    

    try:
        message = (service.users().messages().send(userId="me", body=body).execute())
        print('Message Id: %s' % message['id'])
        return 'Email sent!'
    except HttpError as error:
        print('An error occurred: %s' % error)
        return 'Failed to send email'
    


def lambda_handler(event, context):
    # Extract event details
    print(event)
    preferences = event['preferences']
    articles = event['articles']
    email_address = event['email_address']
    request_type = event['request_type'] 
    

    # Summarize articles with OpenAI
    summarized_articles = []
    for article in articles:
        content_dict = create_content(preferences, article)
        summarized_articles.append(content_dict)

    if request_type == "email":
        email = create_email_html(summarized_articles, preferences, style="light-mode")
 

        # Send the email
        send_email(email['subject'], email['body'], email_address)

        return {
            'statusCode': 200,
            'body': json.dumps('Email sent successfully!')
        }

if __name__ == "__main__":
    send_email("Testing", "Testing", "cgormley07@gmail.com")