import requests
import json
from src.constants import OPENAI_API_KEY
from openai import OpenAI


client = OpenAI()
client.api_key = OPENAI_API_KEY


def execute_with_ollama(query):
    payload = {
        "model": "mistral",
        "format": "json",
        "stream": False,
        "messages": [{"role": "user", "content": query}],
    }

    payload_json = json.dumps(payload)
    url = "http://localhost:11434/api/chat"

    try:
        response = requests.post(url, data=payload_json)

        if response.status_code == 200:
            response_data = response.json()
            response_data = json.loads(response_data["message"]["content"])
            print(response_data)
            return response_data
        else:
            print(f"LLM Request failed with status code {response.status_code}")
            return None
    except requests.RequestException as e:
        print(f"Request exception: {e}")
        return None


def execute_with_openai(query):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": query},
        ],
        temperature=0.7,
        max_tokens=64,
        top_p=1,
        response_format={"type": "json_object"},
    )
    response = completion.choices[0].message.content
    print(response)
    response = json.loads(response)
    return response
