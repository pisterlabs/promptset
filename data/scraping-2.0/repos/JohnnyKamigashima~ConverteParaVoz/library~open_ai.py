""" Funções para usar OpenAi Api
"""

import json
from math import e
import requests
from requests.exceptions import HTTPError
from openai import OpenAI


def open_ai(system, prompt, api_key, model, max_retries=3) -> str:
    """
    Improved version of the open_ai function.

    Args:
        prompt (dict): The prompt for the OpenAI API.
        api_key (str): The API key for authentication.
        model (str): The model to use for the API request.
        base_url (str): The base URL for the API request.
        max_retries (int): The maximum number of retries in case of exception.

    Returns:
        str: The final result from the API response.
    """

    client = OpenAI(api_key=api_key)
    retries = 0
    while retries < max_retries:
        try:
            # This code is for v1 of the openai package: pypi.org/project/openai
            response = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": f"{system}"
                    },
                    {
                        "role": "user",
                        "content": f"{prompt}"
                    }
                ],
                temperature=1,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )

            result = response.choices[0].message.content
            result_json = json.loads(result)
            concatenated_values = ''

            for key, value in result_json.items():
                concatenated_values += str(value) + '\n'

            final_result = concatenated_values.rstrip()

            return final_result
        except HTTPError as error:
            # Handle specific HTTP errors here
            print(f"HTTP Error occurred: {error}")
            retries += 1
        except requests.exceptions.RequestException as error:
            # Handle other request-related errors here
            print(f"Request Exception occurred: {error}")
            retries += 1
        except json.decoder.JSONDecodeError as error:
            # Handle JSON decoding errors here
            print(f"JSON Decode Error occurred: {error}")
            retries += 1
        except AttributeError as error:
            # Handle attribute errors here
            print(f"Attribute Error occurred: {error}")
            retries += 1
    return ''

def query_openai(prompt, model, api_key_value, bot_personality_value, texto_indesejado) -> str:
    """
    Queries the OpenAI API with a given prompt and returns the bot's response.

    Args:
        prompt (str): The prompt to send to the OpenAI API.
        model (str): The ID of the OpenAI model to use.
        api_key_value (str): The API key for the OpenAI API.
        bot_personality_value (str): The personality of the bot to use in the response.
        texto_indesejado (list[str]): The list of words that the bot should not use in the response.

    Returns:
        str: The bot's response to the given prompt.
    """
    prompt = prompt.strip()
    print("\nPROMPT => " + prompt)
    bot_response: str = open_ai(
        bot_personality_value,
        prompt,
        api_key_value,
        model
    )
    bot_response = bot_response.replace('\n', '. ').strip()
    bot_response = bot_response.replace('..', '.')

    for i in texto_indesejado:
        if i in bot_response:
            bot_response = bot_response.replace(i, '')
    return bot_response

def write_response(response_file, bot_response):
    """
    Writes the bot's response to a text file.

    Args:
        response_file (str): The name of the file to write the response to.
        bot_response (str): The response to write to the file.
    """
    with open(response_file + '.txt', "w", encoding="utf-8") as file:
        file.write(bot_response)
