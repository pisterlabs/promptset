#get the most recent emails to the user from a postgres database, if the emails haven't already been checked, then apply a function on them which calls open ai api on them as a prompt
#and then return the response from the api

import os
import sys
import json
import requests
from gmail import watch_new_emails
import psycopg2
import openai #import openai

openai.api_key = os.environ['OPENAI_API_KEY']



psycopg2.connect(
    host = os.environ['DB_HOST'],
    database = os.environ['DB_NAME'],
    user = os.environ['DB_USER'],
    password = os.environ['DB_PASSWORD']
)

def get_recent_emails(user_id):
    user_id = watch_new_emails()
    

    return []

def chatgpt(msg):
    #call the open ai api on the message
    #return the response

    response = openai.Completion.create(
        engine="davinci",
        prompt=msg,
        temperature=0.9,
        max_tokens=150,
    )

    return response



