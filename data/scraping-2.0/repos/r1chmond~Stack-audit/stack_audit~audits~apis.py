import json
import os

from audits.forms import SmartContractForm
import openai
import requests
import dotenv


openai.api_key = dotenv.read_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

def get_source_from_hiro_api(contract: str):
    url = f"https://api.mainnet.hiro.so/extended/v1/contract/{contract}"

    payload = {}
    headers = {
        'Accept': 'application/json'
    }
    response = requests.request("GET", url, headers=headers, data=payload)
    contents = response.content.decode('utf-8')
    return json.loads(contents)["source_code"]


def get_response_from_openai(source: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"Explain this clarity contract in lay-man terms:{source}"}
        ]
    )
    return response['choices'][0]['message']['content']
