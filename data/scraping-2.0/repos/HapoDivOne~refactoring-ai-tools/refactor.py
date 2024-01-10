"""
Refactoring tool using ChatGPT from Vue 2 to Vue 3
Example:
$ python refactor.py MyView.ts
"""

import os
import re
import sys

import openai
from dotenv import load_dotenv

load_dotenv()


REFACTOR_PROMPT = """
You are an assistant design to help developper for migrating their code from Vue 2 to Vue 3 using Typescript with Composition API. Here is a set of rules you must absolutely follow:

1. Rewrite the <script lang="ts"> to <script setup lang="ts">
2. The content of the script tag must be a valid Typescript code
3. The component must be flattened into the script setup
4. Use the `useRoute` approach instead of $route. Same for $router.
5. Store is using vuex.
6. Do not use Ref is the type can be infered from the value pass into ref()
7. Do not put all the methods and properties into a global const object
8. Prefer using global "const router = useRouter()" instead of live instanciation when needed
"""

def refactor(filename, model):
    with open(filename, "r", encoding="utf8") as f:
        content = f.read()

    # ask for refactoring
    response = openai.ChatCompletion.create(
        model=model,
        stream=True,
        temperature=0,
        messages=[
            {"role": "system", "content": REFACTOR_PROMPT},
            {"role": "user", "content": content},
        ]
    )

    # get the refactored script
    for entry in response:
        choice = entry["choices"][0]
        if choice["finish_reason"] == "stop":
            break

        if choice["finish_reason"] is not None:
            print("ERR: Unexpected finish_reason", choice["finish_reason"])
            sys.exit(1)

        delta_content = choice["delta"].get("content")
        if delta_content is not None:
            print(delta_content, end="")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-16k")
    parser.add_argument("file", type=str)
    args = parser.parse_args()

    openai.api_key = os.getenv('OPENAPI_APIKEY')
    refactor(args.file, args.model)
