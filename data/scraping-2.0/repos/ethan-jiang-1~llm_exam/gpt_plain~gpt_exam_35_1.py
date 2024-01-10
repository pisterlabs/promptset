# import json
import openai
import os
from dotenv import load_dotenv
import logging
import sys


load_dotenv()

logger = logging.getLogger("openai")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


openai.api_key = os.environ["OPENAPI_KEY"]

def get_completion_from_messages(messages, model="gpt-3.5-turbo-0613", temperature=0, max_tokens=500):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, 
        max_tokens=max_tokens, 
    )
    print(response)
    return response.choices[0].message["content"]

delimiter = "####"
system_message = f"""
You will be provided with customer service queries. \
The customer service query will be delimited with \
{delimiter} characters.
Classify each query into a primary category \
and a secondary category. 
Provide your output in json format with the \
keys: primary and secondary.

Primary categories: Billing, Technical Support, \
Account Management, or General Inquiry.

Billing secondary categories:
Unsubscribe or upgrade
Add a payment method
Explanation for charge
Dispute a charge

Technical Support secondary categories:
General troubleshooting
Device compatibility
Software updates

Account Management secondary categories:
Password reset
Update personal information
Close account
Account security

General Inquiry secondary categories:
Product information
Pricing
Feedback
Speak to a human

"""
user_message = """\
I want you to delete my profile and all of my user data"""


def exam1():
    messages = [{'role': 'system', 
                'content': system_message},    
                {'role': 'user', 
                'content': f"{delimiter}{user_message}{delimiter}"}, ] 
    response = get_completion_from_messages(messages)
    print(response)


if __name__ == "__main__":
    print(os.environ["OPENAPI_KEY"])
    exam1()
