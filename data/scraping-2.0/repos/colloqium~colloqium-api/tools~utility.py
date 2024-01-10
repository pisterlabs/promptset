# from logs.logger import logger
import random
import openai
import re
import time
from typing import List, Dict

def add_message_to_conversation(conversation: List[Dict[str, str]], message: Dict[str, str]) -> List[Dict[str, str]]:
    """
    This function should append the new message to the recipient_communication conversation.
    """
    (f"Updating conversation with new message: {message}")
    conversation = conversation.copy()
    conversation.append({"role": "user", "content": message})
    return conversation


def get_llm_response_to_conversation(conversation, functions = []):
    conversation = conversation.copy()
    response_content = ""


    # have a random wait time between 60 and 90 seconds to avoid hitting the rate limit
    wait_time =  random.randint(60, 90)

    max_retries = 50
    retry_count = 0

    while retry_count <= max_retries:
        try:
            # generate a new response from OpenAI to continue the conversation

            if functions == []:
                completion = openai.ChatCompletion.create(model="gpt-4",
                                                      messages=conversation,
                                                      temperature=0.9)
            else:
                completion = openai.ChatCompletion.create(model="gpt-4-0613",
                                                      messages=conversation,
                                                      functions=functions,
                                                      function_call="auto",
                                                      temperature=0.9)

            '''
            Response in the following formats:

                    {
                        "id": "chatcmpl-123",
                        ...
                        "choices": [{
                            "index": 0,
                            "message": {
                            "role": "assistant",
                            "content": null,
                            "function_call": {
                                "name": "get_current_weather",
                                "arguments": "{ \"location\": \"Boston, MA\"}"
                            }
                            },
                            "finish_reason": "function_call"
                        }]
                    }

                    or

                    {
                        "id": "chatcmpl-123",
                        ...
                        "choices": [{
                            "index": 0,
                            "message": {
                            "role": "assistant",
                            "content": "The weather in Boston is currently sunny with a temperature of 22 degrees Celsius.",
                            },
                            "finish_reason": "stop"
                        }]
                    }
            '''
            response_content = completion.choices[0].message


            conversation.append(response_content)
            # print(f"Adding OpenAI response to conversation: {response_content}")
            conversation = conversation
            break
        except openai.error.RateLimitError:
            # sleep for a while before retrying
            print(f"Model hit rate limit, waiting for {wait_time} seconds before retry...")
            time.sleep(wait_time)
            retry_count += 1
            continue
        except openai.error.ServiceUnavailableError:
            print(f"Model unavailable, waiting for {wait_time} seconds before retry...")
            time.sleep(wait_time)
            retry_count += 1
            continue

    return conversation[-1]


def initialize_conversation(system_prompt: str) -> List[Dict[str, str]]:
    return [{"role": "system", "content": system_prompt}]


def remove_trailing_commas(json_like):
    json_like = re.sub(",[ \t\r\n]+}", "}", json_like)
    json_like = re.sub(",[ \t\r\n]+\]", "]", json_like)
    return json_like

def format_phone_number(phone_number: str) -> str:
    """
    This function should format the phone number to be in the format +1xxxxxxxxxx

    It should check whether or not the number has the +1 country code, if not, it should add it.
    """
    digits = [char for char in phone_number if char.isdigit()]
    if len(digits) == 10:
        return "+1" + "".join(digits)
    elif len(digits) == 11:
        return "+" + "".join(digits[0:])
    else:
        return phone_number