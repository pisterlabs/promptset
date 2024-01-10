"""
This module contains the code for the code generation model codex.
"""

from time import sleep
import random

import openai
from apikey import OPENAI_KEY
from f import get_samples, write_jsonl_in_folder

openai.api_key = OPENAI_KEY
NUM_SAMPLES_PER_TASK = 1


def generate(prompt):
    """
    The generate function takes a prompt as input and returns the generated code.
    The prompt is a string of text that will be used to generate code. The function
    returns the generated code as a string.
    """

    try:
        sleep(3)

        response = openai.Completion.create(
            # model="code-cushman-001",
            model="code-davinci-002",
            prompt=prompt,
            temperature=0.2,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\nclass", "\ndef"],
        )
        sleep(20)
        code = response.choices[0].text
        print("".join([prompt, code]))
        return code

    except Exception as error:  # pylint: disable=broad-except
        print(error)
        print("Error")
        return ""


def main() -> None:
    """
    This function is the main function for the code generation model codegeex.
    """
    samples = get_samples(
        num_samples_per_task=NUM_SAMPLES_PER_TASK, _get_code_from_api=generate
    )
    print(samples)
    write_jsonl_in_folder("cd", samples)


if __name__ == "__main__":
    main()
