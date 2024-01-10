import argparse
import json
import os
import re
import traceback
from pathlib import Path

import openai
from dotenv import load_dotenv
from googleapiclient.discovery import build as build_service
from rich.console import Console

import numpy as np

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORGANIZATION = os.getenv("OPENAI_ORGANIZATION")
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

openai.api_key = OPENAI_API_KEY
openai.organization = OPENAI_ORGANIZATION


def json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif np.isscalar(obj):
        return obj.item()
    raise TypeError(f"Cannot serialize object of type {type(obj)}")


def try_parse_json(s: str):
    match = re.search(r"```json(.*)```", s, re.DOTALL)
    if match is None:
        obj = json.loads(s)
    else:
        obj = json.loads(match.group(1))
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prompt", type=Path, required=True)
    parser.add_argument("-m", "--model", type=str, default="gpt-3.5-turbo")
    args = parser.parse_args()

    if not args.model.startswith(("gpt-4", "gpt-3.5-turbo")):
        raise ValueError("This script only supports chat models.")

    console = Console()

    service = build_service("customsearch", "v1", developerKey=GOOGLE_SEARCH_API_KEY)
    cse = service.cse()

    query = console.input("[cyan]Enter query:[/cyan] ")

    with open(args.prompt / "system.txt") as fid:
        system_prompt = fid.read().strip()

    def get_response(messages) -> str:
        # console.print("Sending prompt:", style="bold")
        # console.print(messages, style="bright_black", soft_wrap=True, markup=False)

        response = openai.ChatCompletion.create(
            model=args.model, messages=messages, temperature=0
        )
        resp_content = response.choices[0].message.content
        console.print("Response:", style="bold")
        console.print(resp_content, style="bright_black", soft_wrap=True, markup=False)
        console.print()
        return resp_content

    with open(args.prompt / "data_search.txt") as fid:
        prompt = fid.read().strip()
        prompt = prompt.format(query=query)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    resp_content = get_response(messages)
    obj = try_parse_json(resp_content)

    while True:
        if obj["action"] == "done":
            break
        elif obj["action"] == "select":
            selected_data_sources = obj["data_sources"]
            console.print(
                f"Selected data sources: {selected_data_sources}",
                style="bright_black",
            )
            console.input("Press Enter to continue...")
            messages += [
                {"role": "assistant", "content": resp_content},
                {"role": "user", "content": "Continue"},
            ]
            resp_content = get_response(messages)
            obj = try_parse_json(resp_content)
            continue

        search_query = obj["search_query"]
        res = cse.list(q=search_query, cx=GOOGLE_CSE_ID, num=5).execute()
        res = res.get("items", [])
        if len(res) == 0:
            search_response = "No results found."
        else:
            data_description = "\n".join(
                [f"{i + 1}. {x['title']}: {x['snippet']}" for i, x in enumerate(res)]
            )
            search_response = f"Here are the top 5 results:\n{data_description}"
        console.print("Search response:", style="bold")
        console.print(search_response, style="bright_black", soft_wrap=True, markup=False)
        console.input("Press Enter to continue...")
        messages += [
            {"role": "assistant", "content": resp_content},
            {"role": "user", "content": search_response},
        ]
        resp_content = get_response(messages)
        obj = try_parse_json(resp_content)


if __name__ == "__main__":
    main()
