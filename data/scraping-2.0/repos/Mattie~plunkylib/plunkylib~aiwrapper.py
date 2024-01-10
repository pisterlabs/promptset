# Reasonable portion of this code is taken from github.com/OthersideAI/chronology licensed under this MIT License:
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

import time
import asyncio
import openai
import os
import json
from loguru import logger
from functools import wraps

openai.api_key = os.getenv("OPENAI_API_KEY")

async def set_api_key(api_key):
    openai.api_key = api_key

# decorator to retry API calls
def retryAPI(exception, tries=4, delay=3, backoff=2):
    """Retry calling the decorated function using an exponential backoff.
    :param Exception exception: the exception to check. may be a tuple of
        exceptions to check
    :param int tries: number of times to try (not retry) before giving up
    :param int delay: initial delay between retries in seconds
    :param int backoff: backoff multiplier e.g. value of 2 will double the
        delay each retry
    :raises Exception: the last exception raised
    """
    def deco_retry(f):
        @wraps(f)
        async def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return await f(*args, **kwargs)
                except exception as e:
                    msg = "%s, Retrying in %d seconds..." % (str(e), mdelay)
                    logger.debug(msg)
                    await asyncio.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return await f(*args, **kwargs)
        return f_retry  # true decorator
    return deco_retry

# openai
async def _completion(prompt, engine="ada", max_tokens=64, temperature=0.7, top_p=1, stop=None, presence_penalty=0, frequency_penalty=0, echo=False, n=1, stream=False, logprobs=None, best_of=1, logit_bias={}, user=None):
    if user is None:
        user = "_not_set"
    logger.debug("""CONFIG:
    Prompt: {0}
    Engine: {2}
    Max Tokens: {3}, Stop: {5}
    Temperature: {1}, Top-P: {4}
    Presence Penalty {6}, Frequency Penalty: {7}
    Echo: {8},  N: {9}, Stream: {10}, Log-Probs: {11}, Best Of: {12}, Logit Bias: {13}
    User: {14}""".format(prompt, temperature, engine, max_tokens, top_p, stop, presence_penalty, frequency_penalty, echo, n, stream, logprobs, best_of, logit_bias, user))
    response = await openai.Completion.acreate(engine=engine,
                                                prompt=prompt,
                                                max_tokens=max_tokens,
                                                temperature=temperature,
                                                top_p=top_p,
                                                presence_penalty=presence_penalty,
                                                frequency_penalty=frequency_penalty,
                                                echo=echo,
                                                stop=stop,
                                                n=n,
                                                stream=stream,
                                                logprobs=logprobs,
                                                best_of=best_of,
                                                logit_bias=logit_bias,
                                                user=user)
    logger.debug("OpenAI Completion Result: {0}".format(response))
    return response

# oai
async def _chatcompletion(prompt, engine="gpt-3.5-turbo", max_tokens=64, temperature=0.7, top_p=1, stop=None, presence_penalty=0, frequency_penalty=0, n=1, stream=False, logit_bias={}, user=None):
    if user is None:
        user = "_not_set"
    # prompt will be in JSON format, let us translate it to a python list
    # if the prompt is a list already, we will just use it as is
    if isinstance(prompt, list):
        messages = prompt
    else:
        messages = json.loads(prompt)
    logger.debug("""CONFIG:
    Prompt: {0}
    Model: {2}
    Max Tokens: {3}, Stop: {5}
    Temperature: {1}, Top-P: {4}
    Presence Penalty {6}, Frequency Penalty: {7}
    N: {8}, Stream: {9}, Logit Bias: {10}
    User: {11}""".format(prompt, temperature, engine, max_tokens, top_p, stop, presence_penalty, frequency_penalty, n, stream, logit_bias, user))
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
                                            logit_bias=logit_bias,
                                            user=user)
    logger.debug("OpenAI Completion Result: {0}".format(response))
    return response

def _fetch_response(resp, n):
    if n == 1:
        return resp.choices[0].text
    else:
        logger.debug('_fetch_response :: returning {0} responses from GPT-3'.format(n))
        texts = []
        for idx in range(0, n):
            texts.append(resp.choices[idx].text)
        return texts

def _trimmed_fetch_response(resp, n):
    if n == 1:
        return resp.choices[0].text.strip()
    else:
        logger.debug('_trimmed_fetch_response :: returning {0} responses from GPT-3'.format(n))
        texts = []
        for idx in range(0, n):
            texts.append(resp.choices[idx].text.strip())
        return texts

def _trimmed_fetch_chat_response(resp, n):
    if n == 1:
        #response[‘choices’][0][‘message’][‘content’]
        return resp.choices[0].message.content.strip()
    else:
        logger.debug('_trimmed_fetch_response :: returning {0} responses from ChatGPT3'.format(n))
        texts = []
        for idx in range(0, n):
            texts.append(resp.choices[idx].message.content.strip())
        return texts

def prepend_prompt(new_stuff, prompt):
    '''
    Add new content to the start of a string.
    '''
    return "{0}{1}".format(new_stuff, prompt)

def append_prompt(new_stuff, prompt):
    '''
    Add new content to the end of a string.
    '''
    return "{1}{0}".format(new_stuff, prompt)

def add_new_lines_end(prompt, count):
    '''
    Add N new lines to the end of a string.
    '''
    return "{0}{1}".format(prompt, "\n"*count)

def add_new_lines_start(prompt, count):
    '''
    Add N new lines to the start of a string.
    '''
    return "{1}{0}".format(prompt, "\n"*count)

async def gather(*args):
    '''
    Run methods in parallel (they don't need to wait for each other to finish).
    Requires method argumets to be async.
    Example: await gather(fetch_max_search_doc(query_1, docs), fetch_max_search_doc(query_2, docs))
    '''
    return await asyncio.gather(*args)

# Wrappers

async def cleaned_completion(prompt, engine="ada", max_tokens=64, temperature=0.7, top_p=1, stop=None, presence_penalty=0, frequency_penalty=0, echo=False, n=1, stream=False, logprobs=None, best_of=1, logit_bias={}, user=None):
    '''
    Wrapper for OpenAI API completion. Returns whitespace trimmed result from GPT-3.
    '''
    resp = await _completion(prompt,
                             engine=engine,
                             max_tokens=max_tokens,
                             temperature=temperature,
                             top_p=top_p,
                             presence_penalty=presence_penalty,
                             frequency_penalty=frequency_penalty,
                             echo=echo,
                             stop=stop,
                             n=n,
                             stream=stream,
                             logprobs=logprobs,
                             best_of=best_of,
                             logit_bias=logit_bias,
                             user=user)
    return _trimmed_fetch_response(resp, n)


async def raw_completion(prompt, engine="ada", max_tokens=64, temperature=0.7, top_p=1, stop=None, presence_penalty=0, frequency_penalty=0, echo=False, n=1, stream=False, logprobs=None, best_of=1, logit_bias={}):
    '''
    Wrapper for OpenAI API completion. Returns raw result from GPT-3.
    '''
    resp = await _completion(prompt,
                             engine=engine,
                             max_tokens=max_tokens,
                             temperature=temperature,
                             top_p=top_p,
                             presence_penalty=presence_penalty,
                             frequency_penalty=frequency_penalty,
                             echo=echo,
                             stop=stop,
                             n=n,
                             stream=stream,
                             logprobs=logprobs,
                             best_of=best_of,
                             logit_bias=logit_bias)
    return _fetch_response(resp, n)


# ChatGPT
@retryAPI(openai.error.RateLimitError, tries=3, delay=2, backoff=2)
async def cleaned_chat_completion(prompt, engine="gpt-3.5-turbo", max_tokens=64, temperature=0.7, top_p=1, stop=None, presence_penalty=0, frequency_penalty=0, n=1, stream=False, logprobs=None, logit_bias={}, user=None):
    '''
    Wrapper for OpenAI API chat completion. Returns whitespace trimmed result from ChatGPT3.
    '''
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
                            logit_bias=logit_bias,
                            user=user)

    return _trimmed_fetch_chat_response(resp, n)

# Jurassic

def _j_trimmed_fetch_response(resp, n):
    # resp.json()['completions'][0]['data']['text']
    if n == 1:
        return resp.json()['completions'][0]['data']['text'].strip()
    else:
        texts = []
        for idx in range(0, n):
            texts.append(resp.json()['completions'][idx]['data']['text'].strip())
        return texts

# attempt at making an ai21 jurassic query that mimics the chronological/gpt3 query
async def jurassic_cleaned_completion(prompt, engine="j1-grande", max_tokens=64, temperature=0.7, top_p=1, stop=None, presence_penalty=0, frequency_penalty=0, echo=False, n=1, stream=False, logprobs=None, best_of=1, logit_bias={}, count_penalty=0):
    import requests
    import os
    apikey = os.getenv("AI21_API_KEY")
    resp = requests.post("https://api.ai21.com/studio/v1/" + engine + "/complete",
    headers={"Authorization": "Bearer " + apikey},
    json={
        "prompt": prompt,
        "numResults": n,
        "maxTokens": max_tokens,
        "temperature": temperature,
        "topKReturn": 0,
        "topP":top_p,
        "stopSequences":stop,
        "countPenalty": {
                "scale": count_penalty,
                "applyToNumbers": False,
                "applyToPunctuations": False,
                "applyToStopwords": False,
                "applyToWhitespaces": False,
                "applyToEmojis": True
            },
            "frequencyPenalty": {
                "scale": frequency_penalty,
                "applyToNumbers": False,
                "applyToPunctuations": False,
                "applyToStopwords": False,
                "applyToWhitespaces": False,
                "applyToEmojis": True
            },
            "presencePenalty": {
                "scale": presence_penalty,
                "applyToNumbers": False,
                "applyToPunctuations": False,
                "applyToStopwords": False,
                "applyToWhitespaces": False,
                "applyToEmojis": True
            },
        })
    if resp.status_code != 200:
        return False
    print(resp)
    return _j_trimmed_fetch_response(resp, n)

# method that takes any number of keyword arguments to wrap cleaned_completion, but don't list the named arguments
async def cleaned_completion_wrapper(*args, **kwargs):
    # if the keyword argument "engine" was included
    if "engine" in kwargs:
        # if engine begins with "j1" use the jurassic_cleaned_completion method
        if kwargs["engine"].startswith("j1"):
            # jurassic doesn't support the user params so let us remove it
            if "user" in kwargs:
                del kwargs["user"]
            return await jurassic_cleaned_completion(*args, **kwargs)
        elif kwargs["engine"].startswith("gpt-"):
            if "best_of" in kwargs:
                del kwargs["best_of"]
            return await cleaned_chat_completion(*args, **kwargs)
        # otherwise use the cleaned_completion method
        else:
            return await cleaned_completion(*args, **kwargs)

async def content_classification(content_to_classify):
    """ Uses OpenAI's content filter API to classify content as suitable for viewing. 2 = bad/filtered, 1 = sensitive, 0 = good/unfiltered. 
        See https://beta.openai.com/docs/models/content-filter
    """
    if len(content_to_classify) > 1500:
        # truncate to the last 1500 characters
        content_to_classify = content_to_classify[-1500:]
    # async def cleaned_completion(prompt, engine="ada", max_tokens=64, temperature=0.7, top_p=1, stop=None, presence_penalty=0, frequency_penalty=0, echo=False, n=1, stream=False, logprobs=None, best_of=1, logit_bias={}):    
    prompt = "<|endoftext|>"+content_to_classify+"\n--\nLabel:"
    response = await _completion(prompt, engine="content-filter-alpha", max_tokens=1, temperature=0, top_p=0, logprobs=10)

    output_label = response["choices"][0]["text"]
    # trim it
    output_label = output_label.strip()

    # This is the probability at which we evaluate that a "2" is likely real
    # vs. should be discarded as a false positive
    toxic_threshold = -0.355

    if output_label == "2":
        # If the model returns "2", return its confidence in 2 or other output-labels
        logprobs = response["choices"][0]["logprobs"]["top_logprobs"][0]

        # If the model is not sufficiently confident in "2",
        # choose the most probable of "0" or "1"
        # Guaranteed to have a confidence for 2 since this was the selected token.
        if logprobs["2"] < toxic_threshold:
            logprob_0 = logprobs.get("0", None)
            logprob_1 = logprobs.get("1", None)

            # If both "0" and "1" have probabilities, set the output label
            # to whichever is most probable
            if logprob_0 is not None and logprob_1 is not None:
                if logprob_0 >= logprob_1:
                    output_label = "0"
                else:
                    output_label = "1"
            # If only one of them is found, set output label to that one
            elif logprob_0 is not None:
                output_label = "0"
            elif logprob_1 is not None:
                output_label = "1"

            # If neither "0" or "1" are available, stick with "2"
            # by leaving output_label unchanged.

    # if the most probable token is none of "0", "1", or "2"
    # this should be set as unsafe
    if output_label not in ["0", "1", "2"]:
        output_label = "2"

    return output_label

async def search_query(*args, **kwargs):
    # if the keyword argument "engine" was included, we check for pinecone and then use the pinecone_vectorsearch_query method
    if "engine" in kwargs:
        if kwargs["engine"] == "pinecone":
            return await pinecone_vectorsearch_query(*args, **kwargs)
        else:
            # ain't no other alternative right now
            return None

# attempt at making an ai21 jurassic query that mimics the chronological/gpt3 query
async def pinecone_vectorsearch_query(query, index="default", engine="pinecone", embedding_model="text-embedding-ada-002", top_k=1, include_metadata=True, include_values=False, filter=None, result_format="---\n{result}\n"):
    import pinecone
    pinecone.init(api_key=os.getenv("PINECONE_API_KEY"),
                  environment=os.getenv("PINECONE_PROJ_ENVIRONMENT"))

    # get the index
    index = pinecone.Index(index)

    # convert the query to a vector use the OpenAI embedding model -- one day we'll support other embedding models perhaps
    axq = await openai.Embedding.acreate(input=query, engine=embedding_model)
    #TODO error handling
    xq = axq['data'][0]['embedding']

    res = index.query(xq, top_k=top_k, include_metadata=include_metadata, include_values=include_values, filter=filter)
    output = ""
    for match in res['matches']:
        content = match['metadata']['text'] if 'metadata' in match and 'text' in match['metadata'] else ""
        sourceinfo = match['metadata']['source'] if 'metadata' in match and 'source' in match['metadata'] else ""
        output += result_format.format(result=content, score=match['score'], source=sourceinfo)
    return output
