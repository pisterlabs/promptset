import time
import openai
import logging
import backoff
import itertools
from timer import timer
import os

logging.getLogger('backoff').addHandler(logging.StreamHandler())
logging.getLogger('backoff').setLevel(logging.ERROR)


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
                           openai.error.APIConnectionError,
                           openai.error.ServiceUnavailableError))
    @timer
    def get_suggestions(self,
                        input_prompt,
                        number_of_suggestions=1, max_tokens=1000, temperature=0, frequency_penalty=0):
        openai.api_key = next(self.round_robin)

        start_time = time.perf_counter()
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

        end_time = time.perf_counter()
        run_time = end_time - start_time

        # response_completion_tokens = response["usage"]["completion_tokens"]
        if "completion_tokens" in response["usage"]:
            response_completion_tokens = response["usage"]["completion_tokens"]
        else:
            response_completion_tokens = "N/A"

        if "prompt_tokens" in response["usage"]:
            response_prompt_tokens = response["usage"]["prompt_tokens"]
        else:
            response_prompt_tokens = "N/A"

        if "total_tokens" in response["usage"]:
            response_total_tokens = response["usage"]["total_tokens"]
        else:
            response_total_tokens = "N/A"

        return round(run_time, 3), result, response_completion_tokens, response_prompt_tokens, response_total_tokens


    @backoff.on_exception(backoff.expo,
                          (openai.error.RateLimitError,
                           openai.error.APIConnectionError,
                           openai.error.ServiceUnavailableError))
    @timer
    def get_suggestions_with_any_stop_token(self,
                        input_prompt,
                        stop_token,
                        number_of_suggestions=1, max_tokens=1000, temperature=0, frequency_penalty=0):
        openai.api_key = next(self.round_robin)

        start_time = time.perf_counter()
        response = openai.Completion.create(
            model="code-davinci-002",
            prompt=input_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            n=number_of_suggestions,
            frequency_penalty=frequency_penalty,
            presence_penalty=0,
            stop=stop_token,
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

        end_time = time.perf_counter()
        run_time = end_time - start_time

        response_completion_tokens = response["usage"]["completion_tokens"]
        response_prompt_tokens = response["usage"]["prompt_tokens"]
        response_total_tokens = response["usage"]["total_tokens"]

        return round(run_time, 3), result, response_completion_tokens, response_prompt_tokens, response_total_tokens
