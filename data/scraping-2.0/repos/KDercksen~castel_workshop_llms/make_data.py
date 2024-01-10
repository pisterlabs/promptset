#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import trange

# Load the API keys from .env file
load_dotenv()

openai_client = OpenAI()
model = "gpt-3.5-turbo"
output_file = "data.json"
prompt = "Write a short story with an unexpected ending."
num_samples = 5

# Collect `num_samples` short stories from ChatGPT and store them in `output_file`.

results = []
for _ in trange(num_samples, desc="Collecting sample stories"):
    response = openai_client.chat.completions.create(
        model=model, temperature=1, messages=[{"role": "user", "content": prompt}]
    )
    results.append(
        {"prompt": prompt, "completion": response.choices[0].message.content}
    )

with open(output_file, "w") as f:
    json.dump(results, f, indent=4)
