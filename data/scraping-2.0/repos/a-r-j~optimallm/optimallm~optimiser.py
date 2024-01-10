import inspect
import os
from typing import Callable, Optional

import openai
from rich.console import Console
from rich.markdown import Markdown

DEFAULT_PROMPT: str = """
Here's a Python function:

$code

Optimize the function $name for better performance and readability without changing its output. Make the function as fast as possible. Enclose code in triple backticks.
"""


def format_prompt(prompt: str, function_code: str, function_name: str):
    prompt = prompt.replace("$code", function_code)
    return prompt.replace("$name", function_name)


def opt(function: Callable, prompt_template: Optional[str] = None, max_tokens: int = 300, n: int = 1, temperature: float = 0.5) -> Optional[str]:
    """
    Optimize a Python function using GPT-3.5-turbo.

    :param function: The Python function to be optimized.
    :param prompt_template: A template for the GPT-4 prompt.
    :param max_tokens: Maximum tokens to generate.
    :param n: Number of responses to generate.
    :param temperature: Temperature of the sampling distribution.

    :return: The optimized function code.
    """
    if openai.api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        openai.api_key = api_key

    function_name = function.__name__
    function_code = inspect.getsource(function).strip()

    if prompt_template is None:
        prompt_template = DEFAULT_PROMPT

    prompt = format_prompt(prompt_template, function_code, function_name)

    messages = [
        {"role": "system", "content": "You are an AI language model trained to optimize Python functions for better performance and readability without changing their output."},
        {"role": "user", "content": prompt}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=max_tokens,
        n=n,
        temperature=temperature,
    )

    optimized_code = response.choices[0]["message"]["content"]  # .text.strip()
    if optimized_code:
        console = Console()
        md = Markdown(optimized_code)
        console.print(md)
