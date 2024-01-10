import os
import sys
from typing import Optional

import openai

OPENAI_SECRETS_SEARCH_PATHS: list[str] = [
    "~/.openai_keys",
]

def load_openai_secrets(
    secrets_file: Optional[str] = None,
) -> None:
    """Load secrets into openai client."""
    if secrets_file is None:
        for path in OPENAI_SECRETS_SEARCH_PATHS:
            path = os.path.expanduser(path)
            if os.path.exists(path):
                secrets_file = path
                break

    if secrets_file:
        path = os.path.expanduser(secrets_file)

        with open(path) as f:
            secrets = dict(
                sline.split("=")
                for line in f.readlines()
                for sline in [line.strip()]
                if sline
            )

        openai.organization = secrets["OPENAI_ORGANIZATION"]  # noqa
        openai.api_key = secrets["OPENAI_API_KEY"]  # noqa


def _chat_completion(
    prefix: str,
    query: str,
) -> str:
    result = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            dict(role="system", content=prefix),
            dict(role="user", content=query),
        ],
    )
    return result["choices"][0]["message"]["content"]

def _davinci_completion(
    prefix: str,
    query: str,
) -> str:
    p=f"{prefix}\nInput:\n{query}\n\nResult:\n"
    result = openai.Completion.create(
        model="text-davinci-003",
        prompt=p,
	max_tokens=4096 - len(p),
    )
    return result["choices"][0]["text"]

def completion(
    prefix: str,
    query: str,
    *,
    _davinci: bool = False,
) -> str:
    if _davinci:
    	return _davinci_completion(prefix=prefix, query=query)
    else:
    	return _chat_completion(prefix=prefix, query=query)
