# import json
import openai
import os
#from dotenv import load_dotenv
import logging
import sys

os.environ["OPENAPI_KEY"] = ""

#load_dotenv()

logger = logging.getLogger("openai")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

openai.api_base = "https://rwkv.ai-creator.net"

print(openai.api_key)

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
system_message = """
"""
user_message = """\
Aesthetics deals with objects that are_____."""


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
