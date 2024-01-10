import json
import random
import time
from json import JSONDecodeError
from typing import Optional

import openai


def extract_codeblock(output: str):
    if "```" not in output:
        return output

    lines = output.split("\n")

    back_quotes_index = [
        index for index, line in enumerate(lines) if line.startswith("```")
    ]
    start_index = back_quotes_index[-2] + 1
    end_index = back_quotes_index[-1]

    return "\n".join(lines[start_index:end_index])


def correct_json(text: str, model: str):
    prompt = f"""
    以下のJSONをパース可能に修正してください。修正点は記載せず、JSONだけを返してください。
    ```json
    {text}
    ```
    """

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant, which returns answer in JSON format.",
            },
            {"role": "user", "content": prompt[1:-1]},
        ],
        max_tokens=1024,
        temperature=0,
    )

    return response["choices"][0]["message"]["content"]


def parse_llm_output_json(output: str, model: str = "gpt-3.5-turbo"):
    code_block = extract_codeblock(output)

    # まずはそのままパースしてみる
    try:
        return json.loads(code_block)
    except JSONDecodeError:
        pass
    except:
        return None

    # evalでパースしてみる
    try:
        ans = eval(code_block)
        if type(ans) == dict:
            return ans
    except:
        pass

    # それでもダメだったらLLMに直してもらう
    code_block = extract_codeblock(correct_json(code_block, model))
    try:
        return json.loads(code_block)
    except:
        raise ValueError(output)


def execute_openai(system_str: str, prompt: str, model: str = "gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system_str},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1024,
        temperature=0,  # 生成する応答の多様性,
    )

    return response.choices[0]["message"]["content"]


def execute_openai_for_json(system_str: str, prompt: str, model: str = "gpt-3.5-turbo"):
    llm_result = execute_openai(system_str, prompt, model)
    result = parse_llm_output_json(llm_result, model="gpt-4")
    return result


# define a retry decorator
# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
def retry_with_exponential_backoff(
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: Optional[int] = 10,
    errors: tuple = (openai.error.RateLimitError,),
):
    """Retry a function with exponential backoff."""

    def decorator(func):
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
                    # Increment retries
                    num_retries += 1

                    # Check if max retries has been reached
                    if max_retries and num_retries > max_retries:
                        raise Exception(
                            f"Maximum number of retries ({max_retries}) exceeded."
                        )

                    # Increment the delay
                    delay *= exponential_base * (1 + jitter * random.random())

                    # Sleep for the delay
                    time.sleep(delay)

                # Raise exceptions for any errors not specified
                except Exception as e:
                    raise e

        return wrapper

    return decorator


def trim_prompt(
    prompt_base: str, adjusting_text: str, max_tokens: int, model: str = "gpt-3.5-turbo"
):
    import tiktoken

    enc = tiktoken.encoding_for_model(model)
    prompt = prompt_base.format(adjusting_text).strip()
    tokens = enc.encode(prompt)

    # そもそもmax_tokensより小さい場合はそのまま返す
    if len(tokens) < max_tokens:
        return prompt, len(adjusting_text) - 1

    for i in range(len(adjusting_text) - 1, -1, -1):
        prompt = prompt_base.format(adjusting_text[:i]).strip()
        tokens = enc.encode(prompt)
        if len(tokens) < max_tokens:
            return prompt, i

    raise ValueError("prompt_base larger than max_tokens")
