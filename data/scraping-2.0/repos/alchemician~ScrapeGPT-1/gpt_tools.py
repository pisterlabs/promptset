import os
import re

import backoff
import streamlit as st
from dotenv import load_dotenv

from prompts import (compress_knowledge_base_prompt, debug_error_prompt,
                     extract_fn_name_prompt, look_for_clues_prompt,
                     system_prompt, write_scraper_prompt)

load_dotenv()

if "PROMPTLAYER_API_KEY" in os.environ:
    import promptlayer

    promptlayer.api_key = os.environ.get("PROMPTLAYER_API_KEY")
    openai = promptlayer.openai
else:
    import openai

openai.api_key = os.environ.get("OPENAI_API_KEY")
from openai.error import RateLimitError


@backoff.on_exception(backoff.expo, RateLimitError)
def chat_gpt_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


@st.cache_data(persist=True)
def generate_function_name(url, task):
    completion = chat_gpt_with_backoff(
        model="gpt-3.5-turbo",
        temperature=0.0,
        max_tokens=50,
        messages=[
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": extract_fn_name_prompt(url, task)},
        ],
    )
    return completion["choices"][0]["message"]["content"]


@st.cache_data(persist=True)
def write_scraper(url, task, function_name, instructions):
    completion = chat_gpt_with_backoff(
        model="gpt-4",
        temperature=0.0,
        max_tokens=4000,
        messages=[
            {"role": "system", "content": system_prompt()},
            {
                "role": "user",
                "content": write_scraper_prompt(url, task, instructions, function_name),
            },
        ],
    )
    response = completion["choices"][0]["message"]["content"]
    try:
        if "```python" in response:
            code = re.search("```python([^`]*)```", response).group(1)
        elif "```py" in response:
            code = re.search("```py([^`]*)```", response).group(1)
        else:
            code = response
    except:
        code = response
    return code


@st.cache_data(persist=True)
def try_debug(code, log, function_name, stdout, stderr, model="gpt-4"):
    prompt = debug_error_prompt(
        function_name=function_name, stdout=stdout, stderr=stderr
    )
    log.append({"role": "user", "content": prompt})
    completion = chat_gpt_with_backoff(
        model=model, temperature=0.3, max_tokens=1000, messages=log
    )
    response = completion["choices"][0]["message"]["content"]
    log.append({"role": "assistant", "content": response})
    if "```python" in response:
        code = re.search("```python([^`]*)```", response).group(1)
    else:
        code = re.search("```py([^`]*)```", response).group(1)
    return code, log, response


@st.cache_data(persist=True)
def look_for_clues(chunk, url, idx, num_chunks, task):
    prompt = look_for_clues_prompt(url, task, idx, num_chunks, chunk)

    completion = chat_gpt_with_backoff(
        model="gpt-3.5-turbo",
        temperature=0.0,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": prompt},
        ],
    )
    return completion["choices"][0]["message"]["content"]


@st.cache_data(persist=True)
def compress_knowledge_base(kb, url, task):
    prompt = compress_knowledge_base_prompt(kb, url, task)
    completion = chat_gpt_with_backoff(
        model="gpt-4",
        temperature=0.0,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": prompt},
        ],
    )
    return completion["choices"][0]["message"]["content"]
