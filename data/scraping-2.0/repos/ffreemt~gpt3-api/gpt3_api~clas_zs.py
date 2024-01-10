"""Classify via zero-shot.

response = openai.Completion.create(
  engine="davinci",
  prompt="Text: I want to visit Mars.\nCategory: Wanderlust\n###\nText: I like it here.\nCategory: Content\n###\nText: I'd love to visit São Paulo.\nCategory: Wanderlust",
  temperature=0,
  max_tokens=64,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0,
  stop=["\n"]
)

"""
from typing import List, Tuple

import os
import sys

import re
from textwrap import dedent
import dotenv
import openai

import logzero
from logzero import logger
from .gtp3_api import assemble_prompt
from set_loglevel import set_loglevel

logzero.loglevel(set_loglevel())


_ = "OPENAI_API_KEY"
OPENAI_API_KEY = os.environ.get(_) or dotenv.dotenv_values().get(_)
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY is not set, you can set in .env or system environ")
else:
    _ = f"{OPENAI_API_KEY[:5]}...{OPENAI_API_KEY[-5:]}"
    logger.debug("OPENAI_API_KEY: %s", _)
    openai.api_key = OPENAI_API_KEY


# fmt: off
def clas_zs(
        query: str,
        examples: List[Tuple[str, str]],
        # engine: str = "davinci-instruct-beta-v3"
        engine: str = "davinci",
        temperature: float = 0.11,
        max_tokens: int = 32,
        top_p: int = 1,
        frequency_penalty: float = 0,  # -2..2, positive penalty
        presence_penalty: float = 0,
        stop: List[str] = ["###", "\n\n"],
        echo: bool = False,
) -> str:
    # fmt: on
    """Classify via zero-shot.

    engine: str = "davinci"
    temperature: float = 0.11
    max_tokens: int = 32
    top_p: int = 1
    frequency_penalty: float = 0
    presence_penalty: float = 0
    stop: List[str] = ["###", "\n\n"]
    stop = None
    echo: bool = True
    """
    # replace ##{n} with #-{n}
    query = re.sub(r"(?<=#)#", "-", str(query))

    # remove \n
    query = re.sub("\n+", " ", query)

    prompt = assemble_prompt(query, examples=examples, prefixes=("Text: ", "Category: "), suffixes=("\n", "\n###\n")).strip()

    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
        echo=echo
    )

    logger.debug(response.choices[0].text)

    return response.choices[0].text
