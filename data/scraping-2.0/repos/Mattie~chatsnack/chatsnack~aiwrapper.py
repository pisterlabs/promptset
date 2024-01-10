import asyncio
import openai
import os
import json
import random
import time
from loguru import logger
from functools import wraps
from openai.error import *

openai.api_key = os.getenv("OPENAI_API_KEY")

if os.getenv("OPENAI_API_BASE") is not None:
    openai.api_base = os.getenv("OPENAI_API_BASE")

if os.getenv("OPENAI_API_VERSION") is not None:
    openai.api_version = os.getenv("OPENAI_API_VERSION")

if os.getenv("OPENAI_API_TYPE") is not None:
    # e.g. 'azure'
    openai.api_type = os.getenv("OPENAI_API_TYPE")

async def set_api_key(api_key):
    openai.api_key = api_key

openai_exceptions_for_retry=(RateLimitError,Timeout,ServiceUnavailableError,TryAgain,APIError)

def retryAPI_a(exceptions, tries=4, delay=3, backoff=2):
    """Retry calling the decorated function using an exponential backoff.
    :param Exception exceptions: the exceptions to check. may be a tuple of
        exceptions to check
    :param int tries: number of times to try (not retry) before giving up
    :param int delay: initial delay between retries in seconds
    :param int backoff: backoff multiplier e.g. value of 2 will double the
        delay each retry (but with a random factor that is 0.5x to 1.5x)
    :raises Exception: the last exception raised
    """
    def deco_retry(f):
        @wraps(f)
        async def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return await f(*args, **kwargs)
                except exceptions as e:
                    msg = "%s, Retrying in %d seconds..." % (str(e), mdelay)
                    logger.debug(msg)
                    await asyncio.sleep(mdelay)
                    mtries -= 1
                    mdelay *= (backoff * random.uniform(0.75, 1.25))
            return await f(*args, **kwargs)
        return f_retry
    return deco_retry

def retryAPI(exceptions, tries=4, delay=3, backoff=2):
    """Retry calling the decorated function using an exponential backoff.
    :param Exception exceptions: the exceptions to check. may be a tuple of
        exceptions to check
    :param int tries: number of times to try (not retry) before giving up
    :param int delay: initial delay between retries in seconds
    :param int backoff: backoff multiplier e.g. value of 2 will double the
        delay each retry (but with a random factor that is 0.5x to 1.5x)
    :raises Exception: the last exception raised
    """
    def deco_retry(f):
        @wraps(f)
        def f_retry(*args, **kwargs):
            mtries = tries
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except exceptions as e:
                    msg = "%s, Retrying in %d seconds..." % (str(e), delay)
                    logger.debug(msg)
                    time.sleep(delay)
                    mtries -= 1
                    delay *= (backoff * random.uniform(0.75, 1.25))
            return f(*args, **kwargs)
        return f_retry  # true decorator
    return deco_retry

# openai
@retryAPI_a(exceptions=openai_exceptions_for_retry, tries=8, delay=2, backoff=2)
async def _chatcompletion(prompt, engine="gpt-3.5-turbo", max_tokens=None, temperature=0.7, top_p=1, stop=None, presence_penalty=0, frequency_penalty=0, n=1, stream=False, user=None, deployment=None, api_type=None, api_base=None, api_version=None, api_key_env=None):
    if user is None:
        user = "_not_set"
    # prompt will be in JSON format, let us translate it to a python list
    # if the prompt is a list already, we will just use it as is
    if isinstance(prompt, list):
        messages = prompt
    else:
        messages = json.loads(prompt)
    logger.trace("""Chat Query:
    Prompt: {0}
    Model: {2}, Max Tokens: {3}, Stop: {5}, Temperature: {1}, Top-P: {4}, Presence Penalty {6}, Frequency Penalty: {7}, N: {8}, Stream: {9}, User: {10}
    """,prompt, temperature, engine, max_tokens, top_p, stop, presence_penalty, frequency_penalty, n, stream, user)

    additional_args = {}
    if deployment is not None:
        additional_args["deployment_id"] = deployment
    if api_key_env is not None:
        additional_args["api_key"] = os.getenv(api_key_env)           
    if stream is None:
        stream = False

    response = await openai.ChatCompletion.acreate(model=engine,
                                            messages=messages,
                                            max_tokens=max_tokens,
                                            temperature=temperature,
                                            top_p=top_p,
                                            presence_penalty=presence_penalty,
                                            frequency_penalty=frequency_penalty,
                                            stop=stop,
                                            n=n,
                                            stream=stream,
                                            user=user,
                                            # NOTE: It's not documented, but the openai library allows you 
                                            #       to pass these api_ parameters rather than depending on
                                            #       the environment variables
                                            api_type=api_type, api_base=api_base, api_version=api_version, 
                                            **additional_args)
    logger.trace("OpenAI Completion Result: {0}".format(response))
    return response

@retryAPI(exceptions=openai_exceptions_for_retry, tries=8, delay=2, backoff=2)
def _chatcompletion_s(prompt, engine="gpt-3.5-turbo", max_tokens=None, temperature=0.7, top_p=1, stop=None, presence_penalty=0, frequency_penalty=0, n=1, stream=False, user=None, deployment=None, api_type=None, api_base=None, api_version=None, api_key_env=None):
    if user is None:
        user = "_not_set"
    # prompt will be in JSON format, let us translate it to a python list
    # if the prompt is a list already, we will just use it as is
    if isinstance(prompt, list):
        messages = prompt
    else:
        messages = json.loads(prompt)
    logger.trace("""Chat Query:
    Prompt: {0}
    Model: {2}, Max Tokens: {3}, Stop: {5}, Temperature: {1}, Top-P: {4}, Presence Penalty {6}, Frequency Penalty: {7}, N: {8}, Stream: {9}, User: {10}
    """,prompt, temperature, engine, max_tokens, top_p, stop, presence_penalty, frequency_penalty, n, stream, user)
    additional_args = {}
    if deployment is not None:
        additional_args["deployment_id"] = deployment
    if api_key_env is not None:
        additional_args["api_key"] = os.getenv(api_key_env)   
    if stream is None:
        stream = False
    response = openai.ChatCompletion.create(model=engine,
                                            messages=messages,
                                            max_tokens=max_tokens,
                                            temperature=temperature,
                                            top_p=top_p,
                                            presence_penalty=presence_penalty,
                                            frequency_penalty=frequency_penalty,
                                            stop=stop,
                                            n=n,
                                            stream=stream,
                                            user=user, 
                                            # NOTE: It's not documented, but the openai library allows you 
                                            #       to pass these api_ parameters rather than depending on
                                            #       the environment variables
                                            api_type=api_type, api_base=api_base, api_version=api_version, 
                                            **additional_args)
    # revert them back to what they were
    logger.trace("OpenAI Completion Result: {0}".format(response))
    return response

def _trimmed_fetch_chat_response(resp, n):
    if n == 1:
        return resp.choices[0].message.content.strip()
    else:
        logger.trace('_trimmed_fetch_response :: returning {0} responses from ChatGPT'.format(n))
        texts = []
        for idx in range(0, n):
            texts.append(resp.choices[idx].message.content.strip())
        return texts

# ChatGPT
async def cleaned_chat_completion(prompt, engine="gpt-3.5-turbo", max_tokens=None, temperature=0.7, top_p=1, stop=None, presence_penalty=0, frequency_penalty=0, n=1, stream=False, user=None, **additional_args):
    '''
    Wrapper for OpenAI API chat completion. Returns whitespace trimmed result from ChatGPT.
    '''
    # ignore any additional_args which are None
    additional_args = {k: v for k, v in additional_args.items() if v is not None}
    resp = await _chatcompletion(prompt,
                            engine=engine,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            presence_penalty=presence_penalty,
                            frequency_penalty=frequency_penalty,
                            stop=stop,
                            n=n,
                            stream=stream,
                            user=user, **additional_args)

    return _trimmed_fetch_chat_response(resp, n)

# TODO: Add back support for content classification (i.e. is this text NSFW?)
# TODO: Consider adding support for other local language models

# Structure of this code is based on some methods from github.com/OthersideAI/chronology
# licensed under this MIT License:
######
# MIT License
#
# Copyright (c) 2020 OthersideAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#######