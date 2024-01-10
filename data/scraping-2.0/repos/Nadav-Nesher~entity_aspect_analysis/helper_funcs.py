"""
This module contains helper functions and configurations for interacting with the OpenAI API.

It includes:
- Setting the API key for OpenAI.
- Defining an enumeration for specifying response formats.
- Function to get completions from the OpenAI model.

Imports:
- openai: OpenAI's API client.
- Enum: A Python standard library for creating enumerations..
- OPENAI_API_KEY from secret: A module to securely fetch the OpenAI API key.
"""


from openai import OpenAI
from enum import Enum
from typing import Dict, List
# TODO: move API key to env var
from secret import OPENAI_API_KEY

# Initialize an OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


# Define the ResponseFormat enum
class ResponseFormat(Enum):
    """
    An enumeration class to define response format types for OpenAI API requests.

    Attributes:
        TEXT: Specifies a response format of type text.
        JSON_OBJECT: Specifies a response format as a JSON object.
    """
    TEXT = {"type": "text"}
    JSON_OBJECT = {"type": "json_object"}


def get_completion_from_messages(messages: List[Dict[str, str]],
                                 model: str = "gpt-3.5-turbo-1106",
                                 frequency_penalty: float = 0,
                                 n: int = 1,
                                 temperature: float = 0,
                                 response_format: ResponseFormat = ResponseFormat.TEXT) -> str:
    """
    Retrieves a response from the OpenAI API based on the provided messages and parameters.

    Parameters:
        messages (List[Dict[str,str): A list of messages (dicts) to send to the OpenAI API for generating completions.
        model (str): The model to use for generating completions. Default is "gpt-3.5-turbo-1106".
        frequency_penalty (float): A penalty to apply to increase or decrease the likelihood of new information. Default is 0.
        n (int): The number of completions to generate. Default is 1.
        temperature (float): Controls randomness in the response generation. Lower values mean less random responses. Default is 0.
        response_format (ResponseFormat): The format of the response from the OpenAI API. Default is ResponseFormat.TEXT.

    Returns:
        str: The content of the response message from the OpenAI API.
    """

    response = client.chat.completions.create(
        messages=messages,
        model=model,
        frequency_penalty=frequency_penalty,
        n=n,
        temperature=temperature,
        response_format=response_format.value
    )

    return response.choices[0].message.content