"""
This is the main module for getting completion from OpenAI
"""

import os
from dotenv import load_dotenv
import openai


def get_completion(prompt, model="gpt-3.5-turbo-16k"):
    """
    Get a response from the OpenAI GPT-3 model using the provided prompt.

    This function retrieves an API key from the environment, 
    sets up a series of messages with a specific role and content, 
    and then initiates a chat completion operation with OpenAI. 
    The response from the model is then returned.

    Parameters:
    - prompt (str): The text that will be sent to the model for completion.
    - model(str, optional): The id of the model that will be queried. 
        Defaults to "gpt-3.5-turbo-16k".

    Returns:
    - response.choices[0].message["content"] (str): 
        The completion result returned from the openai.ChatCompletion.create() method.
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")
    load_dotenv()
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]
