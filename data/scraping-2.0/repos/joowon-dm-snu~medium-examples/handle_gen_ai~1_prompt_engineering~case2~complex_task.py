import os
from enum import Enum
from typing import Dict, Optional

import jinja2
import openai
from templates import PROMPT_V1, PROMPT_V2, PROMPT_V3

openai.api_key = os.environ.get("OPENAI_API_KEY") or "YOUR-API-KEY"

DEFAULT_PARAMS = {
    "engine": "text-davinci-003",
    "temperature": 0,
    "max_tokens": 256,
    "top_p": 1,
    "n": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "stop": ["###"],
}

DEFAULT_INPUT1 = "A close-up, black & white studio photographic portrait of SUBJECT, dramatic backlighting, 1973 photo from Life Magazine."
DEFAULT_INPUT2 = "A vibrant photograph of SUBJECT, wide shot, outdoors, sunset photo at golden hour, wide-angle lens, soft focus, shot on iPhone 6, on Flickr in 2007"
DEFAULT_INPUT3 = "A vibrant photograph of SUBJECT, wide shot, outdoors, sunset photo at golden hour, wide-angle lens, soft focus, shot on iPhone 6, on Flickr in 2007"


class PromptTable(Enum):
    v1 = [PROMPT_V1, DEFAULT_INPUT1]
    v2 = [PROMPT_V2, DEFAULT_INPUT2]
    v3 = [PROMPT_V3, DEFAULT_INPUT3]


def send_gpt3_request(prompt: str, params: Optional[Dict] = None) -> str:
    params = params or DEFAULT_PARAMS
    res = openai.Completion.create(prompt=prompt, **params)
    return res.choices[0].text


def run_task(prompt_version: str, user_input: Optional[str] = None) -> str:
    try:
        prompt, default_input = PromptTable[prompt_version].value
        user_input = user_input or default_input
        prompt = jinja2.Template(prompt).render(user_input=user_input)

    except IndexError:
        raise ValueError("Invalid prompt version")

    print("Used prompt:")
    print(prompt)
    return send_gpt3_request(prompt)


if __name__ == "__main__":
    prompt_version = "v" + input("Prompt Version: ")
    user_input = input("Input (you can skip this step): ")
    print("=====================")
    output = run_task(prompt_version, user_input)
    print("=====================")
    print(output)
