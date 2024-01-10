#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os import environ

import together
from dotenv import load_dotenv
from openai import OpenAI

# Load the API keys from .env file
load_dotenv()

client = OpenAI()
together.api_key = environ["TOGETHER_API_KEY"]

system_prompt = """
You are a helpful assistant that extracts information from clinical reports.
The user will give you texts containing measurements of lesions.  Format your
answers as JSON with the following structure:

[
    {
        "lesion type": ("malignment" or "benign"),
        "lesion location": (the location where the lesion is observed),
        "lesion size": (the measurement reported in the text, written as <number> <unit>)
    },
    ...
]
"""

user_prompt = """
Clinical report (chest x-ray): Two new lesions found in the left upper lobe, 1
cm and 8mm respectively.  Lesion right inferior lobe, 7mm (was 4 millimeters).
One cyst right upper lobe, half a centimeter.
"""

# OpenAI structure (see https://platform.openai.com/docs/api-reference/chat/create)
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]

# Structure of response: see https://platform.openai.com/docs/api-reference/chat/create
response = client.chat.completions.create(model="gpt-4", messages=messages)

print(f"Generated response message: {response.choices[0].message.content}")

# If using e.g. TogetherAI endpoint, slightly different
# Prompt structure is [INST]<<SYS>> system prompt here <</SYS>> user message here [/INST]
# after which generated message will follow.

llama2_system_prompt = f"<<SYS>>{system_prompt}<</SYS>>"
llama2_user_prompt = f"[INST]{llama2_system_prompt}\n{user_prompt}[/INST]"
llama2_response = together.Complete.create(
    prompt=llama2_user_prompt,
    model="togethercomputer/llama-2-70b-chat",
    max_tokens=1024,
)

generated_message = llama2_response["output"]["choices"][0]["text"]
print(f"Generated response message: {generated_message}")
