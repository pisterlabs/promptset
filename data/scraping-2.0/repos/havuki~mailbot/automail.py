
import openai
import os
import requests
import re
from colorama import Fore, Style, init
import base64
import datetime

# import ytfetch and execute the main function
import emfetch

#initialize colorama
init()

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read().strip()
    
# Add the code for sending email using Mailgun API

def send_email(subject, body, recipient):
    api_key = <ADD YOUR API KEY HERE>
    domain = <ADD YOUR DOMAIN HERE>
    url = f"https://api.mailgun.net/v3/{domain}/messages"
    return requests.post(
        url,
        auth=("api", api_key),
        data={"from": f"AI Newsletter Service <mailgun@{domain}>",
              "to": recipient,
              "subject": subject, 
              "text": body})

  
# Load the optimized system prompt for writing the email newsletter
system_prompt = open_file('prompt1.txt')

# Set up OpenAI API key
openai.api_key = open_file('openaiapikey.txt')

# Generate the email content using OpenAI GPT-3
response = openai.Completion.create(
    engine='text-davinci-003',
    prompt=system_prompt + '\n\n',
    max_tokens=500
 )
email_content = response.choices[0].text.strip()

# Print the generated email content
print(email_content)

recipient = open_file('recipiencs.txt')
response = send_email("AI Newsletter", email_content, recipient)
if response.status_code == 200:
    print("Email sent successfully!")
else:
    print("Failed to send email. Response:", response.text)

