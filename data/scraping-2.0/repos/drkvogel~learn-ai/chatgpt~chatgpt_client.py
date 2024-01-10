#!/usr/bin/env python

import openai

openai.api_key = "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"
model_engine = "text-davinci-002"

while True:
    prompt = input("Input: ")
    completion = openai.Completion.create(engine=model_engine, prompt=prompt, max_tokens=1024, n=1, stop=None, temperature=0.7)
    message = completion.choices[0].text
    print(message)
    print()
