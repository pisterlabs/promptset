import openai
from typing import Optional
import numpy as np
import tiktoken
import os


enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
openai.api_key = os.environ[
    "OPENAI_API_KEY"
]  # "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"
# OPENAI_MODEL = "gpt-4"
OPENAI_MODEL = "gpt-3.5-turbo-16k"
TEMPERATURE = 0
TOP_P = 1

PROMPT_BASE = (
    "Write a Python script that uses {library} to connect to a {device} {category}. Only respond with code or direct explanations"
    " of the code provided, nothing else."
)

LIBRARY_DESC_PROMPT = (
    "Write a rich, helpful description of {library} Python library and then provide a list of popular, commonly used instruments from the library. Give thorough explanations and insightful information about the library"
    " nothing else than words."
)

PROMPT_DOCS = "Given the following documentation\n\n```{docstring}```\n\n{base_prompt}"


def query_chatgpt(
    library: str, device: str, category: str, docstring: Optional[str]
) -> str:
    prompt = PROMPT_BASE.format(library=library, device=device, category=category)
    if docstring and docstring is not np.nan:
        tokens = enc.encode(docstring)
        if len(tokens) > 14_000:
            docstring = enc.decode(tokens[:14_000]) + "..."
        prompt = PROMPT_DOCS.format(docstring=docstring, base_prompt=prompt)

    messages = [
        {
            "role": "system",
            "content": f"You are a Python hardware engineer building code examples for {library}",
        },
        {"role": "user", "content": prompt},
    ]
    response = openai.ChatCompletion.create(
        model=OPENAI_MODEL, temperature=TEMPERATURE, top_p=TOP_P, messages=messages
    )
    return response.choices[0].message.content


def query_python_lib_desc(library: str):
    prompt = LIBRARY_DESC_PROMPT.format(library=library)
    messages = [
        {
            "role": "system",
            "content": f"You are a Python hardware engineer experienced with instruments from {library} and have made insane amount of contributions to Python libraries.",
        },
        {"role": "user", "content": prompt},
    ]

    response = openai.ChatCompletion.create(
        model=OPENAI_MODEL, temperature=TEMPERATURE, top_p=TOP_P, messages=messages
    )
    return response.choices[0].message.content
