import argparse

import requests
from personal_keys import openai_api_key

# REF: https://platform.openai.com/docs/api-reference/completions


def calculator_prompt(prompt):
    response = requests.post(
        url="https://api.openai.com/v1/completions",  # api endpoint
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}",
        },
        json={
            "model": "text-davinci-003",  # ML model to use
            "prompt": f"Write python script to {prompt}",
            "max_tokens": 1000,  #
            "temperature": 0.2,  # "creativeness"
        },
    )
    assert response.status_code == 200
    return response.json()["choices"][0]["text"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", help="The prompt to send to the OpenAI API")
    args = parser.parse_args()
    print(calculator_prompt(args.prompt))
