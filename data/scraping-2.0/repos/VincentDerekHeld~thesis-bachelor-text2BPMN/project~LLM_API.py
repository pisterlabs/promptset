import logging
import os

import openai
from openai import OpenAI
import requests


def generate_response_GPT3_instruct_model(prompt: str) -> str:
    """
    Generate a response from the GPT3.5-instruct model based on the provided prompt.
    Args:
        prompt: the prompt to generate a response from
    Returns:
        response_text: the generated response text
    """
    debug_mode = False
    try:
        if debug_mode: print(f"*** Prompt: *** len: {len(prompt).__str__()} \n {prompt} \n")
        api_key = os.environ["OPENAI_API_KEY"]
        # Define the headers for the HTTP request
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        # Define the data payload for the API request
        data = {
            "model": "gpt-3.5-turbo-instruct",
            "prompt": prompt,
            "max_tokens": 1000,
            "temperature": 0
        }
        # Make the POST request to the OpenAI API
        response = requests.post('https://api.openai.com/v1/completions', headers=headers, json=data)
        # Extract the response text
        response_data = response.json()
        if response_data['choices']:
            response_text = response_data['choices'][0]['text'].strip()
            # response_text = filter_quotation_marks(response_text)
        else:
            raise ValueError("No LLM response received in data")

        # response_text = response_text.strip()
        if debug_mode:
            print(f"*** LLM Response: ***\n {response_text}")
            print("*" * 50)

        return response_text

    except requests.exceptions.Timeout:
        raise TimeoutError("Request timed out while calling the OpenAI API")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")


def generate_response_GPT4_model(prompt: str) -> str:
    """
       Generate a response from the GPT4 model based on the provided prompt.
       Uses the v1/chat/completions endpoint API
       Args:
           prompt: the prompt to generate a response from
       Returns:
           response_text: the generated response text
       """
    try:
        logging.getLogger('openai').setLevel(logging.ERROR)
        logging.getLogger('urllib3').setLevel(logging.ERROR)
    except:
        raise Exception("Could not set logging level for openai logger")
    model_engine = "gpt-4-0613"
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    intro = ""
    response = client.chat.completions.create(
        model=model_engine,
        messages=[
            {"role": "system", "content": intro},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": prompt},
        ]
    )
    response_text = response.choices[0].message.content
    # response_text = filter_quotation_marks(response_text)
    response_text = response_text.strip()
    return response_text


def generate_response_text_davinci_003(prompt):
    """    Generate a response from the text-davinci-003 model based on the provided prompt.
           Uses the completions API
           Args:
               prompt: the prompt to generate a response from
           Returns:
               response_text: the generated response text
           """
    model_engine = "text-davinci-003"
    response = openai.Completion.create(
        engine=model_engine, prompt=prompt, max_tokens=1024, n=1, stop=None, temperature=0.0)
    response_text = response["choices"][0]["text"]
    text_response = response_text.strip()
    return text_response


def print_GPT_model_overview():
    """Print an overview of the available GPT models"""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    print(client.models.list())


def filter_quotation_marks(text: str) -> str:
    """
    Filter quotation marks from the provided text.
    Args:
        text: the text to filter quotation marks from
    Returns:
        The filtered text
    """
    return text.replace('\"', "").replace("\'", "")


def normalize_boolean_result(result: str) -> bool:
    """
    Normalize the result of the LLM to a boolean value.
    Args:
        result: the result to normalize to an boolean value
    Returns:
        The normalized boolean value
    """
    lower_result = result.lower()
    if lower_result in ["true", "yes"]:
        return True
    elif lower_result in ["false", "no"]:
        return False
    else:
        raise ValueError("No valid LLM response received in data: '{}'".format(result))


if __name__ == '__main__':
    print("Hello, World!")
    print_GPT_model_overview()
