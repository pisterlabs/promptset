import os
import re
import time
import openai
import logging

from openai.error import RateLimitError
from openai.error import Timeout
from openai.error import APIConnectionError

from retry import retry


import requests


def parse_gpt_response(text_response):
    tokens = text_response.split(' ')
    int_vals = []
    for token in tokens:
        int_val = int(token) if token and token.isdecimal() else None
        if int_val and int_val <= 100 and int_val >= 0:
            int_vals.append(int_val)
    # if there are multiple values, take first one
    # but later check keywords like 'out of '
    # i.e. 20 out of 100
    return int_vals[0]*0.01 if int_vals else 0



def parse_bloom_response(text_response):
    nums = re.findall(r'\d+', text_response)
    return nums[0] if nums else None



@retry((Timeout, RateLimitError, APIConnectionError), tries=5, delay=1, backoff=2)
def ask_gpt(input_text):
    time.sleep(0.05)
    ''' Prompt ChatGPT or GPT3 level of importance of one message directly
        TODO: save not only parsed value but also explanation
        TODO: decice where None values should be handled and throw exception
    '''
    openai.api_key = os.getenv("OPEN_AI_KEY")
    # todo might be worth specifying what type of data a bit ( if not independent of metadata )
    prompt = 'Rate this message text from 0 to 100 by level of importance: %s' % input_text

    system_prompt = '''
        You are assisting human with incoming messages prioritisation.
        Please try to guess what emails are generic or sent automatically and
        non-urgent and don't give them scores above 50.
        Above 50 is for emails that need actions from receiving person.
        Please keep advertisements under 20.
        Slack messages are not less important than emails.
        Work related slack messages and not requiring actions from 40 to 60.
        And work related  slack messages and requiring actions from you should be around 80.
    '''
    response = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
          , timeout=20
          , max_tokens=100
        )

    text_response = response['choices'][0]['message']['content']

    priority_score = parse_bloom_response(text_response)
    model_justification = text_response.replace('%s ' % priority_score, '')
    priority_score = int(priority_score)*0.01 if priority_score else None
    return priority_score, model_justification
