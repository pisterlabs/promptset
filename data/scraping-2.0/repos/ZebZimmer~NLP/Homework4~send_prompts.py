import openai
import os
import json
import math


def set_api_key() -> None:
    """Read the API key from openai_key.txt and set it."""

    # the api key should be stored as a single line in the text file
    file_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "openai_key.txt")
    )

    with open(file_path) as f:
        openai.api_key = f.read().strip()


def completion_request(
    prompt: str,
    model: str = "text-davinci-003",
    temperature: int = 0,
    top_p: float = 1,
    frequency_penalty: float = 0.5,
    presence_penalty: float = 0,
) -> str:
    """Make an API call for Completion and return the response"""
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    )
    # print(f'Response from {model}: {json.dumps(response, indent=4)}')
    return response['choices'][0]['text']

def chat_completion_request(
    message: str,
    model: str = "gpt-3.5-turbo",
    temperature: int = 1,
    top_p: float = 1,
    frequency_penalty: float = 0,
    presence_penalty: float = 0,
) -> str:
    """Make an API call for ChatCompletion and return the response"""
    # for message formatting: https://platform.openai.com/docs/api-reference/chat/create
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "user", "content": message}
        ],
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )
    return response.choices[0]["message"]["content"]

def get_response(message):
    
    response = openai.ChatCompletion.create(
        model = 'gpt-3.5-turbo',
        temperature = 1,
        messages = [
            {"role": "user", "content": message}
        ]
    )
    return response.choices[0]["message"]["content"]


# example prompt for make_completion_request (text-davinci-003 model)
if __name__ == '__main__':
    set_api_key()

    prompt = "Decide whether a Tweet's sentiment is positive, neutral, or negative.\n\nTweet: \"I loved the new Batman movie!\"\nSentiment:"
    # response = completion_request(prompt)
    # print(f'response: {response}\n\n')

    prompt = """Q: Minneapolis and Saint Paul are known as the Twin Cities in Minnesota. Which one is the capital of Minnesota? 
    A: Saint Paul.

    Q: New York is a large city in New York. What's the capital of New York?
    A: Albany.

    Q: Kansas City is another large city. Kansas City is the capital of which state?"""
    response = chat_completion_request(prompt)
    print(f'response: {response}')

