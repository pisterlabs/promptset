#!/usr/bin/env python3

import os
import sys
sys.executable = '/usr/bin/python3'
import base64
import email
import openai
import time
from bs4 import BeautifulSoup
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.generator import Generator
from io import BytesIO

USER = sys.argv[1]

def decode_mail(email_data):
    msg = email.message_from_bytes(email_data)

    from_name = email.utils.parseaddr(msg['From'])[0]
    from_email = email.utils.parseaddr(msg['From'])[1]
    to_email = email.utils.parseaddr(msg['To'])[1]
    subject = msg['Subject']
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                body = part.get_payload(decode=True)
            elif part.get_content_type() == 'text/html':
                html_content = part.get_payload(decode=True)
                soup = BeautifulSoup(html_content, "html.parser")
                body = soup.get_text()
            elif part.get_content_type() == 'text/base64':
                base64_content = part.get_payload(decode=True)
                body = base64.b64decode(base64_content).decode('utf-8')
    else:
        body = msg.get_payload(decode=True)
    return from_name, from_email, to_email, subject, body

def send_to_ai(from_name, subject, body):
    openai.api_key = '{changeapikey}'

    chat_models = 'gpt-3.5-turbo'

    system_message = "You are an AI and are tasked with writing replies to emails. Write your replies as if you were the human to whom the email was sent and in the following format:\nHello FROM NAME,\n\nYOUR REPLY\n\nBest regards"
    user_message = f"This email is from:{from_name}. This email has a subject of: {subject}. This email's body is: {body}"

    result = openai.ChatCompletion.create(
        model=chat_models,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    )
    return result.choices[0].message['content']

def create_reply(from_email, subject, body, new_msg, to_email):
    re_subject = "Re: " + subject
    text = new_msg + "\n\n" + body
    user_email, domain = USER.split('@')

    msg = MIMEMultipart()
    msg['From'] = to_email
    msg['To'] = from_email
    msg['Subject'] = re_subject

    msg.attach(MIMEText(text, 'plain'))

    # Specify the location of the user's draft directory in Dovecot
    draft_dir = f"/home/{changeusername}/mail/{domain}/{user_email}/.Drafts/cur"
    draft_filename = f"{user_email}-draft-{str(int(time.time()))}.eml"

    # Ensure the draft directory exists
    os.makedirs(draft_dir, exist_ok=True)

    # Write the email to the draft file
    with open(os.path.join(draft_dir, draft_filename), 'w') as f:
        gen = Generator(f)
        gen.flatten(msg)

def process_email(email_data):
    from_name, from_email, to_email, subject, body = decode_mail(email_data)
    ai_response = send_to_ai(from_name, subject, body)
    create_reply(from_email, subject, body, ai_response, to_email)

email_data = sys.stdin.buffer.read()
process_email(email_data)
