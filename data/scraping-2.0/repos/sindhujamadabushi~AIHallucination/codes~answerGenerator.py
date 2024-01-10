import json
import os
from openai import OpenAI
import argparse


def generate_answers(context, question, openai_key):
    client = OpenAI(api_key = openai_key)
    context_part = f"Context: {context}\n" if context else "Context: \n"
    prompt = f"Based on the following context, answer the given question: {context_part}Question: {question}"   
    response = client.completions.create(model="text-davinci-002",
    prompt=prompt,
    max_tokens=150)
    answer = response.choices[0].text.strip()
    return answer