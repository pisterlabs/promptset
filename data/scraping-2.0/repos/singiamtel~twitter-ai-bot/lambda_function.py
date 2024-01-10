#!/usr/bin/env python
# This bot is a simple bot that will write a tweet every 12 hours, using AWS Lambda

import os
import openai
from dotenv import load_dotenv
import requests
import datetime
from requests_oauthlib import OAuth1

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

consumer_key = os.getenv("APIKEY")
consumer_secret = os.getenv("APIKEYSECRET")
oauth_token = os.getenv("OAUTHTOKEN")
oauth_token_secret = os.getenv("OAUTHTOKENSECRET")


def connect_to_oauth(consumer_key, consumer_secret, oauth_token, oauth_token_secret):
    url = "https://api.twitter.com/2/tweets"
    auth = OAuth1(consumer_key, consumer_secret,
                  oauth_token, oauth_token_secret)
    return url, auth


def create_tweet():
    # Create a tweet using OpenAI's GPT-3 API
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt="Write a tweet with a random fact",
        temperature=1,
        max_tokens=70,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    text = response.choices[0].text.strip()
    return text


def lambda_handler(event, context):
    # Post a tweet to Twitter
    text = create_tweet()
    payload = {"text": text}
    url, auth = connect_to_oauth(
        consumer_key, consumer_secret, oauth_token, oauth_token_secret
    )
    request = requests.post(
        auth=auth, url=url, json=payload, headers={
            "Content-Type": "application/json"}
    )
    return {
        "code": request.status_code,
        "text": text,
        "time": str(datetime.datetime.now())
    }