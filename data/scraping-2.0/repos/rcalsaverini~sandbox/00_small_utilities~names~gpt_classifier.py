import openai
from entities import Prompt
from typing import Tuple

openai.api_key = "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"

def get_response_and_append_it(prompt: Prompt, max_tokens:int=285, temperature:float=0.1) -> Tuple[str, Prompt]:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompt,
        max_tokens=max_tokens,
        temperature=temperature
    )
    message = response["choices"][0]["message"]["content"]
    prompt.append({"role": "user", "content": message})
    return (message, prompt)