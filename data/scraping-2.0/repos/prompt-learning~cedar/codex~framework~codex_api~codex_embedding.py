import openai
import time
import logging
import backoff
import itertools

import urllib3
import os
from timer import timer

logging.getLogger('backoff').addHandler(logging.StreamHandler())

"""
# https://pypi.org/project/backoff/
pip install backoff
"""

OPENAI_API_KEYS = [
    os.getenv("OPENAI_API_KEY")
]

# STOP_STR = "==================="
STOP_STR = "END_OF_DEMO"

class CodexAPI:
    def __init__(self):
        seq = OPENAI_API_KEYS
        self.round_robin = itertools.cycle(seq)

    @backoff.on_exception(backoff.expo,
                          (openai.error.RateLimitError,
                           openai.error.APIConnectionError))
    @timer
    def get_suggestions(self,
                        input_prompt,
                        number_of_suggestions=1, max_tokens=1000, temperature=0, frequency_penalty=0):
        openai.api_key = next(self.round_robin)

        response = openai.Completion.create(
            model="code-davinci-002",
            prompt=input_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            n=number_of_suggestions,
            frequency_penalty=frequency_penalty,
            presence_penalty=0,
            stop=STOP_STR,
        )
        # suggestions = response['choices']

        result = ""
        if 'choices' in response:
            x = response['choices']
            if len(x) > 0:
                for i in range(0, len(x)):
                    result = x[i]['text']
            else:
                result = ''
        return result

    @backoff.on_exception(backoff.expo,
                          (openai.error.RateLimitError,
                           openai.error.APIConnectionError))
    @timer
    def get_embedding(self,
                      input_prompt,
                      number_of_suggestions=1, max_tokens=1000, temperature=0, frequency_penalty=0):
        pass
