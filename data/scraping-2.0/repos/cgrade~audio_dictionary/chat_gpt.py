#!/usr/bin/env python3
""" A module that uses ChatGPT API to generate response
    from the prompts fed into it by transcribe.
        Transcribe:
            This is a transcription from DeepGram
            using deepgram API to convert speech to text
"""


# import required dependencies
import openai
import asyncio
from transcribe import record_stream1
from key import GPT_KEY, ORG_KEY
import typing


# getting the transcribed text from record_stream func
prompts = asyncio.run(record_stream1())


# Setting the OpenAI API & Organization Keys
openai.api_key = GPT_KEY
openai.organization = ORG_KEY


def generate_response(prompt: str) -> str:
    """ A function that generate response using OpenAI
        Completion API to process the prompts fed into it
        Params:
            prompt: result from transcribed audio
    """
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=60,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response.choices[0].text.strip()


response = generate_response(prompts)
print(prompts)
print(response)
