import os
from itertools import chain
from random import choices

import openai
import yaml
from more_itertools import always_iterable

comp_keys = ["measure", "dimension", "restriction"]
key_order = ["question", "parts"] + comp_keys


openai.api_key = os.getenv("OPENAI_API_KEY")


def read_decompositions(dc_path):
    """Read the YAML file of completions for decompositions and update the 'parts' key"""

    recs = []

    for e in yaml.safe_load(dc_path.open()):
        d = {k: e.get(k) for k in key_order}

        d["parts"] = list(
            chain(
                *[
                    always_iterable(e.get(k, []))
                    for k in ["measure", "dimension", "restriction"]
                ]
            )
        )

        for ck in comp_keys:
            d[ck] = list(always_iterable(d[ck])) if d[ck] else []

        recs.append(d)

    return recs


def write_prompt(decomps, question):
    """Write the prompt for the OpenAI completions interface"""

    dc = choices(decomps, k=4)

    prompt = ""

    for d in dc:
        prompt += f"""
Decompose this question into parts and return the parts as YAML: {d['question']}

{yaml.safe_dump(d, sort_keys=False)}

"""

    prompt += f"\nQuestion: {question} \n"

    return prompt


def decompose_question(decomps, question):
    """Call the OpenAI completions interface to re-write the extra path for a census variable
    into an English statement that can be used to describe the variable."""

    openai.api_key = os.getenv("OPENAI_API_KEY")

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=write_prompt(decomps, question),
        temperature=0.7,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    return response
