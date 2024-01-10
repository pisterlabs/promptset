#!/usr/bin/env python3
"""
-----------------------------------------------------------
Script Name: ChatGPT Email Assistant
Author: Vontainment
Created: 2023-06-16
Updated: 2023-06-16

Description:
This script is part of a project to automate email responses
using OpenAI's GPT-3 model. It takes an email as input and drafts
a reply based on the content of the email, storing the drafted
email into a specified location.
-----------------------------------------------------------
"""

import os
import re
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
user_email, domain = USER.split('@')

def get_nixuser_from_conf(domain):
    # Read the domain's .conf file
    with open(f'/etc/dovecot/conf.d/domains/{domain}.conf', 'r') as f:
        conf_file_content = f.read()

    # Use a regular expression to find the directory after /home/
    match = re.search(r'/home/([^/]+)', conf_file_content)
    if match:
        return match.group(1)  # Return the matched directory name
    else:
        raise Exception(f"Could not find directory after /home/ in {domain}.conf")

nixuser = get_nixuser_from_conf(domain)

def get_config_values(nixuser):
    # Read the user's ai.conf file
    with open(f'/home/{nixuser}/mail/ai.conf', 'r') as f:
        ai_conf_content = f.read()

    # Use regular expressions to find the user key, instructions, and signature
    userkey_match = re.search(r'userkey="([^"]*)"', ai_conf_content)
    instructions_match = re.search(r'instructions="([^"]*)"', ai_conf_content)
    signature_match = re.search(r'signature=|||([^|||]*)|||', ai_conf_content)

    if userkey_match and instructions_match and signature_match:
        # Return the matched user key, instructions, and signature
        return userkey_match.group(1), instructions_match.group(1), signature_match.group(1)
    else:
        raise Exception("Could not find userkey, instructions, or signature in ai.conf")


signature, userkey, instructions = get_config_values(nixuser)

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
    openai.api_key = userkey

    chat_models = 'gpt-3.5-turbo'

    system_message = f"{instructions}"
    user_message = f"This email is from:{from_name}. This email has a subject of: {subject}. This email's body is: {body}"

    result = openai.ChatCompletion.create(
        model=chat_models,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    )
    return result.choices[0].message['content']

def create_reply(from_email, subject, body, new_msg, to_email, signature):
    re_subject = "Re: " + subject
    text = new_msg + "<br><br>" + signature

    msg = MIMEMultipart('alternative')
    msg['From'] = to_email
    msg['To'] = from_email
    msg['Subject'] = re_subject

    part = MIMEText(text, 'html')
    msg.attach(part)

    # Specify the location of the user's draft directory in Dovecot
    draft_dir = f"/home/{nixuser}/mail/{domain}/{user_email}/.Drafts/cur"
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
    create_reply(from_email, subject, body, ai_response, to_email, signature)

email_data = sys.stdin.buffer.read()
process_email(email_data)