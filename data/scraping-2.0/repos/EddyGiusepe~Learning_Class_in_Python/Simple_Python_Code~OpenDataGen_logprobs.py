#!/usr/bin/env python3
"""
Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

Link --> https://github.com/thoddnn/open-datagen/blob/main/opendatagen/examples/never-seen-eval/factuality/Generate_never_seen_factual_dataset.ipynb

"""
from openai import OpenAI

# Substitua sua chave de API OpenAI:
import openai
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key  = os.environ['OPENAI_API_KEY']

from opendatagen.model import OpenAIChatModel



messages = [
    {"role": "system", "content": "Determine o sentimento Positivo ou Negativo na seguinte sentença"},
    {"role": "user", "content": "Eu amo este filme!"}
           ]

model = OpenAIChatModel(name="gpt-3.5-turbo-1106", logprobs=True, max_tokens=50, temperature=[0])
answer = model.ask(messages=messages)

print(model.confidence_score)
