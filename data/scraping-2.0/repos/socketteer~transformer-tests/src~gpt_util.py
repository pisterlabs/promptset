import os
import time
from collections import defaultdict
from types import SimpleNamespace

import openai
import math

from util import metadata

openai.api_key = os.environ["OPENAI_API_KEY"]


def normalize(probs):
    return [float(i) / sum(probs) for i in probs]


def logprobs_to_probs(probs):
    if isinstance(probs, list):
        return [math.exp(x) for x in probs]
    else:
        return math.exp(probs)


def dict_logprobs_to_probs(prob_dict):
    return {key: math.exp(prob_dict[key]) for key in prob_dict.keys()}


def total_logprob(response):
    logprobs = response['logprobs']['token_logprobs']
    logprobs = [i for i in logprobs if not math.isnan(i)]
    return sum(logprobs)


@metadata(usage_count=defaultdict(int), override=defaultdict(lambda: False))
def _request_limiter(engine):
    limits = {
        "ada": 5000,
        "babbage": 1000,
        "curie": 500,
        "davinci": 100,
    }
    _request_limiter.meta["usage_count"][engine] += 1

    if (not _request_limiter.meta["override"]["engine"]
            and _request_limiter.meta["usage_count"][engine] > limits.get(engine, 1000)):

        resp = input(f"{engine} has been used {_request_limiter.meta['usage_count'][engine]} times.\n"
                     f" Are you sure you want to continue and turn off limits? Type {engine} to resume:\n")
        if resp.lower().strip() == engine:
            _request_limiter.meta["override"]["engine"] = True
        else:
            print("STOPPING RESPONSE GENERATION. Quit the program or allow it to finish")
            raise PermissionError(f"{engine} has run too many times: {_request_limiter.meta['usage_count'][engine]}")


def query(prompt, engine="ada", temperature=0.0, attempts=3,
          delay=1, max_tokens=200, override_limits=False, stop=["\n"],
          **kwargs):
    try:
        _request_limiter(engine)
    except PermissionError as e:
        if not override_limits:
            return SimpleNamespace(choices=[defaultdict(int)])

    if attempts < 1:
        raise TimeoutError()
    try:
        params = dict(
            engine=engine,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
            stop=stop,
            timeout=15,
            logprobs=0
        )
        response = openai.Completion.create(**{**params, **kwargs})
        # Sort top_logprobs dict by logprobs
        for choice in response.choices:
            choice["logprobs"]["top_logprobs"] = [
                {key: lp[key] for key in sorted(lp, key=lambda k: -lp[k])}
                for lp in choice["logprobs"]["top_logprobs"]
            ]
        return response


    except Exception as e:
        print(f"Failed to query, {attempts} attempts remaining, delay={delay}")
        print(type(e), e)
        time.sleep(delay)
        return query(prompt, engine, attempts=attempts-1, delay=delay*2)


def query_yes_no(prompt, engine="ada", attempts=3, delay=1, max_tokens=1):
    if attempts < 1:
        raise TimeoutError
    try:

        mask = {'yes': 100,
                'no': 100,
                'YES': 100,
                'NO': 100,
                'Yes': 100,
                'No': 100}


        result = openai.Completion.create(
            engine=engine,
            prompt=prompt,
            temperature=0.0,
            logprobs=10,
            max_tokens=max_tokens,
            echo=True,
            top_p=1,
            n=1,
            stop=["\n"],
            timeout=15,
            logit_bias = logit_mask(mask)

        )
        print(result)

        last_token_logprobs = result['choices'][0]['logprobs']['top_logprobs'][-1]
        print(last_token_logprobs)
        last_token_probs = dict_logprobs_to_probs(last_token_logprobs)
        print(last_token_probs)

        best_prob = 0
        result = "na"
        for key, prob in last_token_probs.items():
            if prob > best_prob:
                if key.lower().replace(" ", "") == 'yes' or key.lower().replace(" ", "") == 'no':
                    best_prob = prob
                    result = key.lower().replace(" ", "")
        return result


    except Exception as e:
        print(f"Failed to query, {attempts} attempts remaining, delay={delay}")
        print(e)
        time.sleep(delay)
        return query(prompt, engine, attempts=attempts-1, delay=delay*2)

def main():
    pass
    # while True:
    #     print(query("", "ada").choices[0]["text"])



