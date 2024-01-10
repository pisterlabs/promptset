import os

import openai

OPENAI_API_KEY = "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"
openai.api_key = OPENAI_API_KEY


def query(words):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"Hi, my friend! Now I am so {words}! What kind of activities you can can recommend me with my mind?",
        temperature=0.9,
        max_tokens=300,
        frequency_penalty=0.5,
        presence_penalty=-0.5,
        best_of=2,
    )
    return response.choices[0].text
