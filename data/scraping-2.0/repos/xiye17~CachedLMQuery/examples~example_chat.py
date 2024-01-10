import os
import openai
import json

from cachedllm.query_tools import *
from cachedllm.prompt_tools import *
from cachedllm.openai_engine import *

def read_json(fname):
    with open(fname) as f:
        return json.load(f)

def dump_json(obj, fname, indent=None):
    with open(fname, 'w', encoding='utf-8') as f:
        return json.dump(obj, f, indent=indent)

def example_openai_chat():
    data = read_json("demo_data.json")
    prompt_tpl = OAIChatPromptTemplate.from_file("demo_chat_template.tpl")
    messages = [prompt_tpl.render(d) for d in data]

    query_interface = CachedQueryInterface(
        OpenAIChatEngine("gpt-3.5-turbo"),
        cache_dir="gsm.sqlite",
    )

    print(messages[0])
    responses = query_interface.complete_batch(
        messages,
        max_tokens=128,
        temperature=0.5,
        n=1,
        logprobs=None,
        echo_prompt=False,
    )

    for r in responses:
        print(json.dumps(r["completions"][0]["message"]["content"], indent=2))

example_openai_chat()
