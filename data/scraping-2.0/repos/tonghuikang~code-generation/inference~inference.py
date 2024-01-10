import json
from datetime import datetime

import openai

from config import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY


def get_tz_time():
    return datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')


def save_request_info(kwargs, response):
    tz = get_tz_time()
    with open(f"openai-api-request/{tz}.json", "w") as f:
        json.dump(kwargs, f, indent=4)
    with open(f"openai-api-response/{tz}.json", "w") as f:
        json.dump(dict(response), f, indent=4)


def request_completion_openai_api(
        prompt,
        suffix="",
        engine="code-davinci-002",
        temperature=0,
        frequency_penalty=-1,
        max_tokens=100,
        logprobs=5,  # take the maximum
        stop=["###"],
    ):
    # https://beta.openai.com/docs/api-reference/completions/create    
    # unused parameters but probably useful
    # logit_bias, presence_penalty, frequency_penalty, suffix
    
    # using codex model in private beta
    # https://beta.openai.com/docs/engines/codex-series-private-beta
    
    kwargs = {
        "prompt":prompt,
        "suffix":suffix,
        "engine":engine,
        "temperature":temperature,
        "max_tokens":max_tokens,
        "logprobs":logprobs,
    }

    response = openai.Completion.create(**kwargs)

    save_request_info(kwargs, response)

    return response

