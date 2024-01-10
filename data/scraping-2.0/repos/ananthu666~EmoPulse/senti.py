import os
import openai
from sentipipe import scoring
openai.api_key = "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"


def analyse(text):
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text},
    ],
    temperature=0,
    max_tokens=256
    )

    scor=scoring(text)
    res=response['choices'][0]['message']['content']
    
    return scor,res

