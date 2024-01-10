import os
import os.path as osp
import json
import openai
import re
from datetime import date

openai.api_key = os.getenv("OPENAI_API_KEY")

PROMPT_CHAT_LIBLINE = """The {lib} library in Python exposes the following API: {api}
"""

PROMPT_CHAT_IMPORTLINE = """import {lib}
"""

PROMPT_CHAT = """
{liblines}
Write a few real-world idiomatic (frequent) usage examples that use ALL the above mentioned APIs.
Note: Each example should use all the APIs mentioned above.
Write only code and don't print the outputs. Mark the start of each and every example with a comment # <<Example>> and end with a comment # <</Example>>.
```
{importlines}
"""


def fill_template(template, libs, apis):
    """Fill the template with the given libraroes and APIs"""
    liblines = "".join([PROMPT_CHAT_LIBLINE.format(lib=lib, api=api) for lib, api in zip(libs, apis)])
    importlines = "".join([PROMPT_CHAT_IMPORTLINE.format(lib=lib) for lib in set(libs)])

    return template.format(liblines=liblines, importlines=importlines)


def encode_question_chat(template, libs, apis):
    """Encode the prompt instructions into a conversation for chat models.
    Reference: https://github.com/ShishirPatil/gorilla/blob/main/eval/get_llm_responses.py
    """
    prompt = fill_template(template, libs, apis)

    prompts = [
        {"role": "system", "content": "You are a helpful API assistant who can write idiomatic API usage examples given the API name."},
        {"role": "user", "content": prompt},
    ]
    return prompts


def get_llm_response(libs, apis):
    PROMPT_TEMPLATE = PROMPT_CHAT

    responses = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=encode_question_chat(PROMPT_TEMPLATE, libs, apis),
        temperature=0,
        max_tokens=1024,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        # stop=['"""', "```"],
    )
    return responses["choices"][0]["message"]["content"]


def parse_response(response):
    """Parse the response to get the code snippets."""
    regex = r"# (?:<Example>|<<Example>>)\n(.*?)# (?:</Example>|<</Example>>)"
    idioms = re.findall(regex, response, re.DOTALL)
    return idioms


if __name__ == "__main__":
    alias_to_lib = {
        "plt": "matplotlib.pyplot",
        "pd": "pandas",
        "df": "pandas",
        "np": "numpy",
        "nn": "torch.nn",
    }

    with open("../multibench.json") as f:
        benchmarks = json.load(f)

    for type in benchmarks:
        for apis in benchmarks[type]:
            query = ";".join(apis)

            libs = []
            for api in apis:
                alias = api.split(".")[0]
                if alias in alias_to_lib:
                    libs.append(alias_to_lib[alias])
                else:
                    libs.append(alias)

            result_dir = f"./results/{date.today()}/{type}/{query}/"

            if not osp.exists(result_dir):
                os.makedirs(result_dir)

            print(f"EVALUATING [{libs}] [{apis}]")
            print("=====================================")

            response = get_llm_response(libs=libs, apis=apis)
            idioms = parse_response(response)

            if idioms == []:
                print(f"Warning: No idioms parsed for [{libs}] [{apis}]; but the response is:\n{response}")

            for i, idiom in enumerate(idioms):
                with open(f"{result_dir}/idiom_{i}.py", "w") as f:
                    f.write(idiom)

            print("=====================================")
