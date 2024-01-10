import os

import openai
import json

from dotenv import load_dotenv


def get_phone_number(name: str):
    """Returns the phone number for the person with the provided name"""
    person = {
        "name": name,
        "phone_number": "0101234567"
    }

    return json.dumps(person)


def get_email_address(name: str):
    """returns the email address for the person with the provided name"""
    person = {
        "name": name,
        "email": "jettro.coenradie@gmail.com"
    }

    return json.dumps(person)


def get_contact_information(name: str):
    """returns all the content information for the person with the provided name"""
    person = {
        "name": name,
        "phone_number": "0101234567",
        "email": "jettro.coenradie@gmail.com"
    }

    return json.dumps(person)


def run_conversation(message_content: str):
    # Step 1: send the conversation and available functions to GPT
    messages = [{"role": "user", "content": message_content}]
    functions = [
        {
            "name": "get_phone_number",
            "description": "Get the phone number for the person with the provided name",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The full name of the person, e.g. Daniel Spee"
                    }
                },
                "required": ["name"]
            }
        },
        {
            "name": "get_email_address",
            "description": "Get the email address for the person with the provided name",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The full name of the person, e.g. Daniel Spee"
                    }
                },
                "required": ["name"]
            }
        },
        {
            "name": "get_contact_information",
            "description": "Get all contact information for the person with the provided name",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The full name of the person, e.g. Daniel Spee"
                    }
                },
                "required": ["name"]
            }
        },
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=functions,
        function_call="auto",  # auto is default, but we'll be explicit
    )

    response_message = response["choices"][0]["message"]

    # Step 2: check if GPT wanted to call a function
    if response_message.get("function_call"):
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "get_phone_number": get_phone_number,
            "get_email_address": get_email_address,
            "get_contact_information": get_contact_information,
        }  # only one function in this example, but you can have multiple
        function_name = response_message["function_call"]["name"]
        fuction_to_call = available_functions[function_name]
        function_args = json.loads(response_message["function_call"]["arguments"])
        function_response = fuction_to_call(
            name=function_args.get("name"),
        )

        # Step 4: send the info on the function call and function response to GPT
        messages.append(response_message)  # extend conversation with assistant's reply
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )  # extend conversation with function response
        second_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
        )  # get a new response from GPT where it can see the function response
        return second_response


if __name__ == '__main__':
    load_dotenv()
    openai.api_key = os.getenv('OPEN_AI_API_KEY')

    # message = "What is mobile phone number for Jettro Coenradie?"
    # message = "What is the emailaddress for Jettro Coenradie?"
    # message = "What is the contact information for Jettro Coenradie?"
    message = "Do you have the email address for Jettro Coenradie"
    phone_number_response = run_conversation(message_content=message)
    print(phone_number_response["choices"][0]["message"]["content"])
