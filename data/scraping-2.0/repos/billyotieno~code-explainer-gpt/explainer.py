import os
from functools import partial

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")


def send_question(question: str) -> dict:
    # Generate doc for this function
    return openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a developer of 30 years experience"},
            {"role": "user", "content": question},
        ],
    )


def retrieve_ai_answer(response: dict) -> str:
    return response["choices"][0]["message"]["content"]


def get_code_info(question: str, code: str) -> str:
    resp = send_question(f"{question}\n\n{code}")
    return retrieve_ai_answer(resp)


retrieve_code_language = partial(
    get_code_info, question="Explain in 1 word what language this code is written in."
)

retrieve_code_explanation = partial(
    get_code_info, question="Explain to me what this code does in one paragraph."
)
