import os
from typing import Optional

import openai
import requests
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

from api_gpt.services.constants import CHATGPT_MODEL
from flask import current_app

openai.api_key = os.getenv("OPENAI_API_KEY")

if openai.api_key is None:
    raise ValueError(
        "OPENAI_API_KEY environment variable not set, please add it in .env file."
    )

llm = OpenAI(openai_api_key=openai.api_key)
chatopenai = ChatOpenAI(temperature=0)


def get_chat_gpt_response(max_tokens, system, content) -> Optional[map]:
    """Generates a GPT-3.5-turbo chat model response using OpenAI API.

    This function sends a POST request to the OpenAI API, which generates
    a response from the GPT-3.5-turbo chat model based on the provided system
    and user messages.

    Args:
        max_tokens (int): The maximum length of the model's response. This
                          controls how much text the model will generate.
        system (str): The initial system message that sets the behavior of the
                      assistant. For example, you can instruct the assistant to
                      speak like Shakespeare.
        content (str): The message from the user that the model will respond to.

    Returns:
        dict: A dictionary containing the API's JSON response if the request is
              successful. The response includes the model's message and other
              information. If an error occurs during the request, this function
              returns None.

    """
    try:
        data = {
            "model": "gpt-3.5-turbo",
            "temperature": 0,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": system,
                },
                {"role": "user", "content": content},
            ],
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + openai.api_key,
        }
        openai_response = requests.post(
            url="https://api.openai.com/v1/chat/completions", headers=headers, json=data
        )
        json_response = openai_response.json()
        return json_response
    except Exception as e:
        return None


def get_gpt3_generation_result(
    model, max_tokens, system_prompt, user_prompt
) -> Optional[map]:
    """Generates a response from the GPT-3 model using OpenAI API.

    This function sends a POST request to the OpenAI API to generate a
    response from the specified GPT-3 model based on the provided system
    and user prompts. If no model is specified, it uses the "text-davinci-003" model.

    Args:
        model (str, optional): The model to generate the response. If None,
                               "text-davinci-003" is used.
        max_tokens (int): The maximum length of the model's response.
        system_prompt (str): The initial system prompt that sets the behavior
                             of the model.
        user_prompt (str): The user's prompt that the model will respond to.

    Returns:
        dict: A dictionary containing the API's JSON response if the request
              is successful. This includes the model's generated text and other
              information. If an error occurs during the request, this function
              returns None.
    """
    try:
        prompt = system_prompt + "\n\n" + user_prompt

        if model is None:
            model = "text-davinci-003"
        data = {
            "model": model,
            "prompt": prompt,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + openai.api_key,
        }
        openai_response = requests.post(
            url="https://api.openai.com/v1/completions", headers=headers, json=data
        )
        json_response = openai_response.json()
        return json_response
    except Exception as e:
        return None


def get_openai_response_string(
    model: str, max_tokens: int, system_prompt: str, user_prompt: str
) -> Optional[str]:
    """Generates a response string from a GPT-3 or GPT-3.5-turbo model using OpenAI API.

    This function chooses the appropriate function to call based on the specified
    model. It then parses the JSON response from the API to extract the generated
    text. If an error occurs, it checks if a global debug flag is set and, if so,
    prints the error message.

    Args:
        model (str): The model to generate the response from. This should be either
                     "gpt-3.5-turbo" or a GPT-3 model like "text-davinci-003".
        max_tokens (int): The maximum length of the model's response.
        system_prompt (str): The initial system prompt that sets the behavior of
                             the model. This is used for the "gpt-3.5-turbo" model.
        user_prompt (str): The user's prompt that the model will respond to. This
                           is used for both models.

    Returns:
        str: The generated text from the model. If an error occurs during the request
             or while parsing the response, this function returns None.
    """
    try:
        if model == "gpt-3.5-turbo":
            json_response = get_chat_gpt_response(
                max_tokens=max_tokens, system=system_prompt, content=user_prompt
            )
            response = json_response["choices"][0]["message"]["content"]
            return response
        else:
            json_response = get_gpt3_generation_result(
                model,
                max_tokens=1000,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            response = json_response["choices"][0]["text"]
            return response
    except Exception as exception:
        current_current_app.logger.error(f"failed in openai repsonse: {exception}")
        return None
