# Developers: r47dzt3ch / github.com/r47dzt3ch
import os
import time
import re
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import base64
import json
import requests
import openai

client_secret_path = os.path.join(os.getcwd(), 'Credentials\client_secret.json')
scope = ['https://mail.google.com', 'https://www.googleapis.com/auth/gmail.modify', 'https://www.googleapis.com/auth/gmail.readonly']
creds =  Credentials.from_authorized_user_file(client_secret_path, scope)
service = build('gmail', 'v1', credentials=creds)


def realtime_email_read_content(domain_filter):
    # Call the Gmail API to get a list of messages
    msg_str = None
    try:
        while True:
            results = service.users().messages().list(userId='me', labelIds=['INBOX'], q='is:unread', maxResults=10).execute()
            messages = results.get('messages', [])

            # If no new unseen messages, wait for 10 seconds before fetching again
            if not messages:
                time.sleep(10)
                continue

            # Reverse the list of messages to start from the latest ones
            messages.reverse()

            for message in messages:
                msg = service.users().messages().get(userId='me', id=message['id']).execute()
                service.users().messages().modify(userId='me', id=message['id'], body={'removeLabelIds': ['UNREAD']}).execute()
                # add to starred
                service.users().messages().modify(userId='me', id=message['id'], body={'addLabelIds': ['STARRED']}).execute()
                headers = msg['payload']['headers']
                f_email = None
                subject = None
                domain = None
                reply_msg = None
                for header in headers:
                    if header['name'].lower() == 'dkim-signature':
                        signature = header['value']
                        domain_start = signature.find('d=')  # Find the position of 'd=' in the DKIM-Signature
                        if domain_start != -1:
                            domain_end = signature.find(';', domain_start)  # Find the position of the next ';' after 'd='
                            domain = signature[domain_start + 2:domain_end]  # Extract the domain from the DKIM-Signature

                # Check if the extracted domain matches the domain_filter
                if domain == domain_filter:
                    print("Domain:", domain)

                    # Extract the sender (From) and subject of the email
                    for header in headers:
                        if header['name'].lower() == 'from':
                            f_email = header['value']
                            print("From:", f_email)
                        if header['name'].lower() == 'subject':
                            subject = header['value']
                            print("Subject:", subject)

                    # Extract the email body
                    if 'payload' in msg and 'parts' in msg['payload'] and len(msg['payload']['parts']) > 0 or 'snippet' in msg:
                        plain_text_part = None
                        try:
                            for part in msg['payload']['parts']:
                                if part['mimeType'] == 'text/plain':
                                    plain_text_part = part
                                    break

                            body_data = plain_text_part['body']['data']
                            msg_str = base64.urlsafe_b64decode(plain_text_part).decode()
                          

                        except:
                            msg_str = msg['snippet']
                        # Print the email body
                        print("Body:", msg_str)
                        # reply to the email
                        reply_msg = message_assistant(msg_str)
                        print("Reply:", reply_msg)
                        subject = "Re: "
                        sender_email = "jeraldjose16@gmail.com"
                        recepient = re.search(r"<(.*?)>", f_email)
                        send_email(sender_email, recepient.group(1), subject, reply_msg)
                    else:
                        print("No 'payload' or 'parts' found in the email message.")

            # Wait for 10 seconds before fetching the next batch of emails
            time.sleep(10)

    except HttpError as error:
        print('An error occurred: %s' % error)


# def analyze_email_content(msg_str): #anaylyze the message using chatgpt model
#     with open('Credentials\api_keys.json', 'r') as file:
#         Credentials = json.load(file)

#     api_key = Credentials['huggingface_godel_apikey']
#     url = "https://api-inference.huggingface.co/models/microsoft/GODEL-v1_1-base-seq2seq"
#     payload = {"inputs": msg_str}
#     headers = {"Authorization": f"Bearer {api_key}"}
#     response = requests.post(url, headers=headers, json=payload)
#     if response.status_code == 200:
#         return response.json()
#     else:
#         return {"error": response.text}

def message_assistant(msg_str):
    url = "https://chatgpt-gpt4-ai-chatbot.p.rapidapi.com/ask"
    with open(r'Credentials/api_keys.json', 'r') as file:
        Credentials = json.load(file)
    payload = { "query": msg_str }
    headers = {
        "content-type": "application/json",
        "X-RapidAPI-Key": Credentials['rapid-apiKey'],
        "X-RapidAPI-Host": "chatgpt-gpt4-ai-chatbot.p.rapidapi.com"
    }

    response = requests.post(url, json=payload, headers=headers)
    return response.json()['response']


def send_email(sender_email, to_email, subject, body):
    message = {
        "raw": base64.urlsafe_b64encode(
            f"From: {sender_email}\r\nTo: {to_email}\r\nSubject: {subject}\r\n\r\n{body}".encode("utf-8")
        ).decode("utf-8")
    }
    try:

        message = service.users().messages().send(userId='me', body=message).execute()
        print("Message sent successfully. Message ID:", message["id"])
    except HttpError as error:
        print('An error occurred while sending the email:', error)


# def text_translate_english(text):
#     with open('Credentials\api_keys.json', 'r') as file:
#         Credentials = json.load(file)
#     url = "https://google-translate1.p.rapidapi.com/language/translate/v2"
#     payload = {
#         "q": text,
#         "target": "en"
#     }
#     headers = {
#         "content-type": "application/x-www-form-urlencoded",
#         "Accept-Encoding": "application/gzip",
#         "X-RapidAPI-Key": Credentials['rapid-apiKey'],
#         "X-RapidAPI-Host": "google-translate1.p.rapidapi.com"
#     }
#     response = requests.post(url, data=payload, headers=headers)
#     return response.json() 

if __name__ == "__main__":
    realtime_email_read_content('gmail.com')
