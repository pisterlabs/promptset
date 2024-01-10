"""
A collection of wrapper utilities around language model APIs like the one offered by OpenAI.
"""
import collections
import datetime
import os
import time
from typing import Optional, List, Union, Dict, Deque, Tuple

import attrs
import openai
import transformers

from databutler.utils import paths
from databutler.utils.logging import logger


@attrs.define(eq=False, repr=False)
class OpenAIKeyManager:
    """
    An internal manager that cycles between OPENAI keys to enable a net-increase in rate=limits.
    """

    keys: List[str]
    _cur_key: str = attrs.field(init=False, default=None)
    _next_key_idx: int = attrs.field(init=False, default=0)

    def __attrs_post_init__(self):
        #  Setup the first key so it's ready to use.
        self.set_next_key()

    def set_next_key(self) -> None:
        """
        Cycles between available API keys. This method should be called before any request.
        """
        self._cur_key = cur_key = self.keys[self._next_key_idx]
        openai.api_key = cur_key
        self._next_key_idx = (self._next_key_idx + 1) % len(self.keys)

    def set_key(self, key: str) -> None:
        """
        Set the key to the provided key.
        """
        self._cur_key = key
        openai.api_key = key
        self._next_key_idx = (self.keys.index(key) + 1) % len(self.keys)

    def get_current_key(self) -> str:
        return self._cur_key


_DEFAULT_KEY_MANAGER: Optional[OpenAIKeyManager] = None


def _setup_openai_key() -> None:
    """
    Loads the OpenAI api_key from the designated path, or from the environment variable OPENAI_KEY if it exists.

    :raises RuntimeError: if neither the path exists, nor the environment variable is supplied.
    """
    global _DEFAULT_KEY_MANAGER

    if _DEFAULT_KEY_MANAGER is not None:
        return

    path_key = os.path.join(
        paths.get_user_home_dir_path(), ".databutler", "openai_key.txt"
    )
    if os.path.exists(path_key):
        #  Great just load it from the file.
        with open(path_key, "r") as f:
            key_text = f.read().strip()

        if "\n" in key_text:
            keys = key_text.split("\n")
        else:
            keys = [key_text]

        _DEFAULT_KEY_MANAGER = OpenAIKeyManager(keys=keys)

    else:
        #  Check if the environment variable OPENAI_KEY is set.
        key_text = os.getenv("OPENAI_KEY")
        if key_text is not None:
            #  Interpret
            if key_text.startswith("["):
                #  Interpret as a list
                try:
                    keys = eval(key_text)
                except:
                    raise ValueError(
                        "Could not understand environment variable OPENAI_KEY"
                    )

            elif "," in key_text:
                keys = key_text.split(",")

            else:
                keys = key_text.split()

            #  First, create the cache file.
            os.makedirs(os.path.dirname(path_key), exist_ok=True)
            with open(path_key, "w") as f:
                f.write("\n".join(keys))

            _DEFAULT_KEY_MANAGER = OpenAIKeyManager(keys=keys)

        else:
            raise RuntimeError(
                f"Neither the environment variable OPENAI_KEY is set, nor the file {path_key} exists."
            )


def _get_default_key_manager() -> OpenAIKeyManager:
    """
    Sets up the default key manager if necessary, and returns it.
    """
    if _DEFAULT_KEY_MANAGER is None:
        _setup_openai_key()

    return _DEFAULT_KEY_MANAGER


def get_available_keys() -> List[str]:
    """
    Returns:
        (List[str]): A list of strings corresponding to available keys
    """
    if _DEFAULT_KEY_MANAGER is None:
        _setup_openai_key()

    return _DEFAULT_KEY_MANAGER.keys


@attrs.define(eq=False, repr=False)
class RateLimiter:
    tokens_per_min: int = 150000
    req_per_min: int = 20

    _window: Deque[Dict] = attrs.field(init=False, default=None)
    _num_tokens_rolling_sum: int = attrs.field(init=False, default=0)

    def __attrs_post_init__(self):
        self._window = collections.deque()
        self._num_tokens_rolling_sum = 0

    def _update_window(self):
        cur_time = datetime.datetime.now()

        while (
            len(self._window) > 0
            and (cur_time - self._window[0]["time"]).total_seconds() > 60
        ):
            item = self._window.popleft()
            self._num_tokens_rolling_sum -= item["num_tokens"]

    def record_request(self, num_tokens: int):
        cur_time = datetime.datetime.now()
        self._num_tokens_rolling_sum += num_tokens
        self._window.append({"num_tokens": num_tokens, "time": cur_time})

        self._update_window()

    def get_wait_time(self, num_tokens: int) -> int:
        if num_tokens > self.tokens_per_min:
            raise ValueError(f"Number of tokens cannot exceed {self.tokens_per_min}")
        if len(self._window) == 0:
            return 0

        self._update_window()
        cur_time = datetime.datetime.now()

        in_conflict_items: List[Dict] = []
        total_tokens_so_far = self._num_tokens_rolling_sum
        while ((total_tokens_so_far + num_tokens) / 60) > (self.tokens_per_min / 60):
            item = self._window[len(in_conflict_items)]
            in_conflict_items.append(item)
            total_tokens_so_far -= item["num_tokens"]

        if len(in_conflict_items) == 0 and len(self._window) == self.req_per_min:
            in_conflict_items.append(self._window[0])

        if len(in_conflict_items) == 0:
            return 0

        latest_in_conflict_time: datetime.datetime = in_conflict_items[-1]["time"]
        return int(
            max(
                (
                    (latest_in_conflict_time + datetime.timedelta(seconds=60))
                    - cur_time
                ).total_seconds(),
                0,
            )
        )


@attrs.define
class OpenAICompletion:
    text: str
    top_logprobs: Optional[List[Dict[str, float]]]
    finish_reason: str


@attrs.define
class OpenAICompletionResponse:
    completions: List[OpenAICompletion]
    timestamp: str
    model: str
    id: str


_CODEX_TOKENIZER: Optional[transformers.GPT2Tokenizer] = None


def codex_tokenize(text: str) -> Dict[str, Union[List[int], List[str]]]:
    global _CODEX_TOKENIZER
    if _CODEX_TOKENIZER is None:
        _CODEX_TOKENIZER = transformers.GPT2Tokenizer.from_pretrained(
            "SaulLu/codex-like-tokenizer"
        )

    return {
        "token_ids": _CODEX_TOKENIZER(text)["input_ids"],
        "token_strs": [
            _CODEX_TOKENIZER.convert_tokens_to_string([i])
            for i in _CODEX_TOKENIZER.tokenize(text)
        ],
    }


_GPT3_TOKENIZER: Optional[transformers.GPT2TokenizerFast] = None


def gpt3_tokenize(text: str) -> Dict[str, Union[List[int], List[str]]]:
    global _GPT3_TOKENIZER
    if _GPT3_TOKENIZER is None:
        _GPT3_TOKENIZER = transformers.GPT2TokenizerFast.from_pretrained("gpt2")

    return {
        "token_ids": _GPT3_TOKENIZER(text)["input_ids"],
        "token_strs": _GPT3_TOKENIZER.tokenize(text),
    }


def tokenize(text: str, engine: str) -> Dict[str, Union[List[int], List[str]]]:
    if engine.startswith("code-"):
        return codex_tokenize(text)
    else:
        return gpt3_tokenize(text)


_RATE_LIMITERS_DICT: Dict[str, RateLimiter] = {}


def _get_rate_limiter_for_api_key(api_key: str):
    if api_key not in _RATE_LIMITERS_DICT:
        _RATE_LIMITERS_DICT[api_key] = RateLimiter()

    return _RATE_LIMITERS_DICT[api_key]


def _retrieve_top_tokens(completion_resp: Dict) -> Optional[List[Dict[str, float]]]:
    if len(completion_resp["logprobs"]["top_logprobs"]) != len(
        completion_resp["logprobs"]["tokens"]
    ):
        logger.warning(f"Unexpected top_logprobs value in response: {completion_resp}")
        return None

    logprob_tokens = completion_resp["logprobs"]["tokens"]
    text = completion_resp["text"]

    start_idx: int = 0
    while start_idx < len(logprob_tokens):
        if text not in "".join(logprob_tokens[start_idx:]):
            start_idx -= 1
            break
        start_idx += 1

    if start_idx < 0:
        logger.warning(f"Could not retrieve top tokens for response: {completion_resp}")
        return None

    end_idx: int = len(logprob_tokens)
    while end_idx > start_idx:
        if "".join(logprob_tokens[start_idx:end_idx]).endswith(text):
            break

        end_idx -= 1

    if end_idx < start_idx:
        logger.warning(f"Could not retrieve top tokens for response: {completion_resp}")
        return None

    return completion_resp["logprobs"]["top_logprobs"][start_idx:end_idx]


def openai_completion(
    engine: str,
    prompt: Optional[str] = None,
    prompts: Optional[List[str]] = None,
    temperature: float = 0,
    num_completions: int = 1,
    max_tokens: int = 64,
    stop: Optional[Union[str, List[str]]] = None,
    retrieve_top_tokens: bool = False,
    *,
    retry_wait_duration: int = 30,
    max_retries: int = 5,
    key_manager: Optional[OpenAIKeyManager] = None,
    min_latency: Optional[int] = None,
    **completion_kwargs,
) -> Union[OpenAICompletionResponse, List[OpenAICompletionResponse]]:
    """
    Wraps the OpenAI completion API, primarily for handling errors in a retry loop.

    :param engine: A string corresponding to the name of the engine to use.
                   Refer to https://beta.openai.com/docs/engines for a list of available engines.
    :param prompt: A string representing the prompt to complete. Must be specified (not specified) if prompts
        is not specified (specified).
    :param prompts: A list of strings corresponding to prompts to complete. Must be specified (not specified) if prompt
        is not specified (specified).
    :param prompts: A string representing the prompt to complete.
    :param temperature: A float for the temperature.
                        Defaults to 0.0 where the model gives the most well-defined answer.
                        A value of 1.0 means the model is the most creative.
    :param num_completions: An integer representing the number of completions to return. Note that this makes sense
                            only when the temperature is non-zero, as duplicates will be returned otherwise.
    :param max_tokens: An integer representing the maximum number of tokens in a valid completion.
    :param stop: An optional string or list of strings corresponding to the stop token(s) for a completion.
    :param retrieve_top_tokens: Whether to retrieve the most likely tokens for every token chosen. The result, if any,
        will be stored in the `top_logprobs` field of the completion object.
    :param retry_wait_duration: An integer representing the time in seconds between retry attempts.
    :param max_retries: An integer representing the maximum number of retries.
    :param key_manager: A key manager to use instead of the default one. Useful for limiting keys or using a specific
        key.
    :param min_latency: An integer representing the desired observed latency. If the request completes faster, the
        process will sleep to achieve the target latency. This can be used to achieve better steady utilization amidst
        rate limits.
    :param completion_kwargs: Other keyword arguments accepted by openai.Completion.create.
                              See https://beta.openai.com/docs/api-reference/completions/create for a full list.

    :return: An OpenAICompletionResponse object containing the completions along with metadata.
    :raises InvalidRequestError: If the request is invalid. This can happen if the wrong model is specified, or invalid
        values for max_tokens, temperature, stop etc. are provided.
    """
    if (prompt is None and prompts is None) or (
        prompt is not None and prompts is not None
    ):
        raise ValueError("Exactly one of prompt and prompts must be specified")

    if prompts is None:
        #  Satisfy the type checker.
        assert prompt is not None
        is_parallel = False
        prompts = [prompt]
    else:
        is_parallel = True

    if key_manager is None:
        #  If the default is initialized, the API key is guaranteed to be assigned as well.
        key_manager = _get_default_key_manager()

    num_keys_tried = 0
    num_retries = 0

    num_tokens = sum(
        (len(tokenize(prompt, engine=engine)["token_ids"]) + max_tokens)
        * num_completions
        for prompt in prompts
    )

    start_time = time.time()
    cur_best = (None, None)
    for key in key_manager.keys:
        rl = _get_rate_limiter_for_api_key(key)
        wait_time = rl.get_wait_time(num_tokens)
        if cur_best[1] is None:
            cur_best = (key, wait_time)
        elif wait_time < cur_best[1]:
            cur_best = (key, wait_time)

    key_to_use, wait_time = cur_best
    # print("KEY", key_to_use)
    key_manager.set_key(key_to_use)

    while num_retries <= max_retries:
        try:
            cur_key = key_manager.get_current_key()
            rate_limiter = _get_rate_limiter_for_api_key(cur_key)
            wait_time = rate_limiter.get_wait_time(num_tokens)
            if wait_time > 0:
                # print(f"Waiting for {wait_time} seconds")
                time.sleep(wait_time)

            openai_response = openai.Completion.create(
                engine=engine,
                prompt=prompts,
                temperature=temperature,
                n=num_completions,
                max_tokens=max_tokens,
                stop=stop,
                logprobs=5 if retrieve_top_tokens else None,
                **completion_kwargs,
            )

        except openai.error.InvalidRequestError:
            raise

        except openai.error.OpenAIError:
            if num_keys_tried < len(key_manager.keys):
                #  Try with another key first.
                num_keys_tried += 1
                key_manager.set_next_key()

            else:
                logger.warning(f"Forced to sleep for {retry_wait_duration} seconds")
                time.sleep(retry_wait_duration)
                num_keys_tried = 0
                num_retries += 1

        else:
            rate_limiter.record_request(num_tokens)
            processed_responses: List[OpenAICompletionResponse] = []
            timestamp = openai_response["created"]
            model = openai_response["model"]
            resp_id = openai_response["id"]
            for idx in range(0, len(prompts)):
                choices = openai_response["choices"][
                    idx * num_completions : (idx + 1) * num_completions
                ]
                completions = [
                    OpenAICompletion(
                        text=choice["text"],
                        top_logprobs=None,
                        finish_reason=choice["finish_reason"],
                    )
                    for choice in choices
                ]

                if retrieve_top_tokens and all(
                    "logprobs" in choice for choice in choices
                ):
                    for completion, choice in zip(completions, choices):
                        completion.top_logprobs = _retrieve_top_tokens(choice)

                processed_responses.append(
                    OpenAICompletionResponse(
                        timestamp=timestamp,
                        model=model,
                        id=resp_id,
                        completions=completions,
                    )
                )

            cur_time = time.time()
            if min_latency is not None and cur_time - start_time < min_latency:
                time.sleep(min_latency - (cur_time - start_time))

            if not is_parallel:
                return processed_responses[0]

            return processed_responses
