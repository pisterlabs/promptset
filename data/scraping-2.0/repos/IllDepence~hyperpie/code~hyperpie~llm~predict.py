import datetime
import re
import openai
from collections import OrderedDict
from tenacity import (
    retry, stop_after_attempt, wait_random_exponential,
    retry_if_not_exception_type
)
from hyperpie import settings
from hyperpie.data import llm_cache


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(9),
    retry=retry_if_not_exception_type(openai.error.InvalidRequestError)
)
def completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


def openai_api(para, prompt, params=None, verbose=False):
    """ Get LLM completion for `prompt` using OpenAI API.

        If the prompt has been used before, the completion will be
        loaded from cache.

        Returns:
        <completion_dict>, <from_cache>
    """

    if params is None:
        params = settings.gpt_default_params

    # check cache
    cached_completion = llm_cache.llm_completion_cache_load(
        params, prompt
    )
    if cached_completion is not None:
        if verbose:
            print('loading completion from cache')
        return cached_completion, True
    if verbose:
        print('completion not in cache')

    # not in cache, get from API
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    completion_dict = OrderedDict({
        "timestamp": timestamp,
        "params": params,
        "paragraph": para,
        "prompt": prompt,
        "completion": None
    })

    try:
        completion = completion_with_backoff(prompt=prompt, **params)
        completion_dict['completion'] = completion
    except openai.error.InvalidRequestError as e:
        # assume requested too many tokens
        token_error_patt = re.compile(
            r"This model\'s maximum context length is (\d+) tokens, however "
            r"you requested \d+ tokens \((\d+) in your prompt; \d+ for "
            r"the completion\)"
        )
        match = token_error_patt.search(e.error.message)
        if match is None:
            # not a token limit error, raise whatever it is
            raise e
        # retry with maximum possible tokens left after accounting for prompt
        max_tokens = int(match.group(1))
        req_prompt = int(match.group(2))
        max_for_completion = max_tokens - req_prompt
        print(
            f'token limit error, retrying with max_tokens={max_for_completion}'
        )
        params_mod = params.copy()
        params_mod['max_tokens'] = max_for_completion
        # first check cache for prompt with reduced max_tokens
        cached_completion = llm_cache.llm_completion_cache_load(
            params_mod, prompt
        )
        if cached_completion is not None:
            if verbose:
                print('loading completion from cache during retry')
            completion_dict = cached_completion
            # dont return here, we still need to save to cache
        else:
            # not in cache, get from API
            completion = completion_with_backoff(prompt=prompt, **params_mod)
            completion_dict['completion'] = completion
            # completion_dict['params'] = params_mod  # we don’t save the
            # completion with modified params, because we would then make the
            # API call causing the error again when re-running the same prompt

    # save to cache
    llm_cache.llm_completion_cache_save(
        params, prompt, completion_dict
    )
    if verbose:
        print('saving completion to cache')

    return completion_dict, False


def get_completion_text(completion):
    """ Get completion text from completion dict.
    """

    return completion['completion']['choices'][0]['text']
