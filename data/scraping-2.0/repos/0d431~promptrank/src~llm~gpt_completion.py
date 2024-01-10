from typing import List
from tenacity import (
    retry,
    wait_random_exponential,
    wait_fixed,
    stop_after_attempt,
    retry_if_not_exception_type,
)
import openai


def get_gpt_completion(
    prompt: str, model="text-davinci-003", temperature=0.0, max_tokens=50, stop=None
) -> str:
    """Run a prompt completion with OpenAI, retrying with backoff in failure case. Supports batching."""
    return get_gpt_completions([prompt], model, temperature, max_tokens, stop)[0]


@retry(
    wait=wait_random_exponential(min=0.5, max=20),
    stop=stop_after_attempt(5),
    retry=retry_if_not_exception_type(openai.error.InvalidRequestError),
)
def get_gpt_completions(
    prompts: List[str],
    model="text-davinci-003",
    temperature=0.0,
    max_tokens=50,
    stop=None,
) -> List[str]:
    """Run a batched prompt completion with OpenAI, retrying with backoff in failure case."""
    response = openai.Completion.create(
        engine=model,
        prompt=prompts,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop,
    )

    # unravel response
    completions = [""] * len(prompts)
    for choice in response.choices:
        completions[choice.index] = choice.text.strip(" \n")

    return completions


@retry(
    wait=wait_fixed(10),
    stop=stop_after_attempt(5),
    retry=retry_if_not_exception_type(openai.error.InvalidRequestError),
)
def get_gpt_chat_completion(
    system: str = "", prompt: str = "", temperature=0.0, max_tokens=50, model="gpt-4"
) -> str:
    """Run a prompt completion with OpenAI chat, retrying with backoff in failure case."""
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response["choices"][0]["message"]["content"].strip(" \n")
