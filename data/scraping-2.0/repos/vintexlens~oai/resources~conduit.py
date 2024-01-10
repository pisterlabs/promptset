import sys

import openai

from resources import config

def get_completion(prompt):
    openai.api_key = config.get_api_key()
    engine = config.get_model()

    if prompt is None:
        print("Prompt is empty. Please enter a prompt.")

    # token calculator
    # count characters in prompt
    tokens_prompt = len(prompt) / 4
    max_tokens = int(4000 - tokens_prompt)
    # print(f"Max tokens: {max_tokens}")
    # print(f"Tokens in prompt: {tokens_prompt}")
    try:
        response = openai.Completion.create(
            engine=engine,
            prompt=prompt,
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=0.7,
        ).choices[0].text
    except openai.error.APIError as e:
        # Handle API error here, e.g. retry or log
        print(f"OpenAI API returned an API Error: {e}")
        sys.exit()
    except openai.error.APIConnectionError as e:
        # Handle connection error here
        print(f"Failed to connect to OpenAI API: {e}")
        sys,exit()
    except openai.error.RateLimitError as e:
        # Handle rate limit error (we recommend using exponential backoff)
        print(f"OpenAI API request exceeded rate limit: {e}")
        sys.exit()
    except openai.error.OpenAIError as e:
        print(f"OpenAI API returned an error: {e}")
        sys.exit()
    return response
    
def get_chat(messages):
    openai.api_key = config.get_api_key()
    engine = config.get_model()

    if len(messages) == 0:
        print("Prompt is empty. Please enter a prompt.")

    try:
        response = openai.ChatCompletion.create(
            model=engine,
            messages=messages
        ).choices[0].message.content
    except openai.error.APIError as e:
        # Handle API error here, e.g. retry or log
        print(f"OpenAI API returned an API Error: {e}")
        sys.exit()
    except openai.error.APIConnectionError as e:
        # Handle connection error here
        print(f"Failed to connect to OpenAI API: {e}")
        sys,exit()
    except openai.error.RateLimitError as e:
        # Handle rate limit error (we recommend using exponential backoff)
        print(f"OpenAI API request exceeded rate limit: {e}")
        sys.exit()
    except openai.error.OpenAIError as e:
        print(f"OpenAI API returned an error: {e}")
        sys.exit()
    return response
    
def get_models():
    openai.api_key = config.get_api_key()
#    engine = config.get_model()

    try:
        response = openai.Model.list(
        ).data
    except openai.error.APIError as e:
        # Handle API error here, e.g. retry or log
        print(f"OpenAI API returned an API Error: {e}")
        sys.exit()
    except openai.error.APIConnectionError as e:
        # Handle connection error here
        print(f"Failed to connect to OpenAI API: {e}")
        sys,exit()
    except openai.error.RateLimitError as e:
        # Handle rate limit error (we recommend using exponential backoff)
        print(f"OpenAI API request exceeded rate limit: {e}")
        sys.exit()
    except openai.error.OpenAIError as e:
        print(f"OpenAI API returned an error: {e}")
        sys.exit()
    return response
