# Import the necessary modules.
# `openai` provides functions to work with the OpenAI API.
# `config` from `decouple` allows for retrieving environment variables or configuration parameters.
import openai
from decouple import config

# Set the OpenAI API key using the value fetched from the configuration or environment variable.
openai.api_key = config("CHATGPT_API_KEY")


def gpt_3_5_turbo_0613(messages, functions=None, function_call="auto"):
    """
    Function to make a request to the OpenAI API for the GPT-3.5 Turbo model.

    Parameters:
    - messages (list): A list of message objects that instruct the model.
    - functions (list, optional): List of functions that the model can call.
    - function_call (str, optional): Mode of function call, default is "auto".

    Returns:
    - response from the OpenAI ChatCompletion API.
    """

    # Set the default parameters for the API call.
    params = {
        "model": "gpt-3.5-turbo-0613",
        "messages": messages,
    }
    # If the caller provides functions, include them in the parameters.
    if functions:
        params["functions"] = functions
        params["function_call"] = function_call

    # Make the API request and return the response.
    return openai.ChatCompletion.create(**params)
