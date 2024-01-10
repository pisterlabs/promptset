import json

import requests
from openai.types.chat import ChatCompletionToolParam


def fetch_pokemon_data_impl(name: str, **kwargs):
    url = f"https://pokeapi.co/api/v2/pokemon/{name}"
    response = requests.get(url)
    if response.status_code == 200:
        data = {
            "wight": response.json()["weight"] * 0.1,
            "height": response.json()["height"] * 0.1,
        }
        return {"message": json.dumps(data), "file": None}
    else:
        return {"message": "not found.", "file": None}


fetch_pokemon_data_tool: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "fetch_pokemon_data",
        "description": "Fetch pokemon data by pokemon name.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Pokemon name, e.g. pikachu",
                },
            },
            "required": ["name"],
        },
    },
}
