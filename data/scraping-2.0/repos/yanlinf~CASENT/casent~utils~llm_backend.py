from collections import Counter
import os
from typing import Dict, Any, List, Union
import openai
import random
import time

# adapter from https://github.com/reasoning-machines/prompt-lib/tree/main/prompt_lib/backends

openai.api_key = os.getenv("OPENAI_API_KEY")


# check if orgainization is set

# if os.getenv("OPENAI_ORG") is not None:
#     openai.organization = os.getenv("OPENAI_ORG")


# from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
def retry_with_exponential_backoff(
        func,
        initial_delay: float = 1,
        exponential_base: float = 2,
        jitter: bool = True,
        max_retries: int = 10,
        errors: tuple = (Exception,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:

                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                print(e)
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(f"Maximum number of retries ({max_retries}) exceeded.")

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


class OpenaiTextAPIWrapper:
    @staticmethod
    @retry_with_exponential_backoff
    def call(
            prompt: Union[str, List[str]],
            max_tokens: int,
            engine: str,
            top_p: float,
            stop_token: str,
            temperature: float,
            num_completions: int = 1
    ) -> dict:
        assert num_completions == 1
        response = openai.Completion.create(
            model=engine,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=[stop_token],
            n=num_completions,
        )
        return response

    @classmethod
    def batch_call(cls,
                   prompts: List[str],
                   max_tokens: int,
                   engine: str,
                   top_p: float,
                   stop_token: str,
                   temperature: float,
                   num_completions: int = 1
                   ) -> dict:
        return cls.call(prompts, max_tokens, engine, top_p, stop_token, temperature, num_completions)

    @staticmethod
    def get_first_response(response) -> Dict[str, Any]:
        """Returns the first response from the list of responses."""
        text = response["choices"][0]["text"]
        return text

    @staticmethod
    def get_majority_answer(response) -> Dict[str, Any]:
        """Returns the majority answer from the list of responses."""
        answers = [choice["text"] for choice in response["choices"]]
        answers = Counter(answers)
        # if there is a tie, return the first answer
        if answers.most_common(1)[0][1] == answers.most_common(2)[1][1]:
            return OpenaiTextAPIWrapper.get_first_response(response)

        return answers.most_common(1)[0][0]

    @staticmethod
    def get_all_responses(response) -> Dict[str, Any]:
        """Returns the list of responses."""
        length = max(choice["index"] for choice in response["choices"]) + 1
        res = [None] * length
        for choice in response["choices"]:
            if res[choice["index"]] is not None:
                raise ValueError('Response contains duplicates')
            res[choice["index"]] = choice["text"]
        return res


class OpenaiChatAPIWrapper:
    @staticmethod
    @retry_with_exponential_backoff
    def call(
            prompt: str,
            max_tokens: int,
            engine: str,
            top_p: float,
            stop_token: str,
            temperature: float,
            num_completions: int = 1,
            system_prompt: str = "You are a helpful assistant."
    ) -> dict:
        assert num_completions == 1
        response = openai.ChatCompletion.create(
            model=engine,
            messages=[
                {
                    'role': 'system',
                    'content': system_prompt
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=[stop_token],
            n=num_completions,
        )
        return response

    @staticmethod
    def get_first_response(response) -> Dict[str, Any]:
        """Returns the first response from the list of responses."""
        text = response["choices"][0]["message"]["content"]
        return text


BACKEND_MAP = {
    "openai-text": OpenaiTextAPIWrapper,
    "openai-chat": OpenaiChatAPIWrapper,
}

ENGINE_TO_BACKEND = {
    "text-curie-001": "openai-text",
    "text-davinci-001": "openai-text",
    "text-davinci-002": "openai-text",
    "text-davinci-003": "openai-text",
    "code-davinci-002": "openai-text",
    "code-cushman-001": "openai-text",
    "gpt-3.5-turbo": "openai-chat",
    "gpt-3.5-turbo-0301": "openai-chat",
}


def few_shot_query(prompt: str, engine: str, **kwargs):
    backend_name = ENGINE_TO_BACKEND[engine]
    api = BACKEND_MAP[backend_name]
    output = api.call(prompt=prompt, engine=engine, **kwargs)
    return api.get_first_response(output)


def batch_few_shot_query(prompts: List[str], engine: str, **kwargs):
    backend_name = ENGINE_TO_BACKEND[engine]
    api = BACKEND_MAP[backend_name]
    if not hasattr(api, "batch_call"):
        raise NotImplementedError(f"Batch call not implemented for {backend_name} backend.")
    output = api.batch_call(prompts=prompts, engine=engine, **kwargs)
    return api.get_all_responses(output)
