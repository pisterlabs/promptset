import os
from functools import partial
from typing import Dict, Any

import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

openai.api_key = os.getenv("OPENAI_API_KEY")

@retry(wait=wait_random_exponential(min=2, max=30), stop=stop_after_attempt(5))
def send_question(question: str) -> Dict[Any, Any]:
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a medical doctor"},
            {"role": "user", "content": question},
        ],
    )

def retrieve_ai_answer(response: Dict[Any, Any]) -> str:
    return response["choices"][0]["message"]["content"]

def get_text_info(question: str, text: str) -> str:
    resp = send_question(f"{question}\n\n{text}")
    return retrieve_ai_answer(resp)

retrieve_text_explanation = partial(
    get_text_info,
    question=(
        "Can you summarize this conversation with a patient into a clinical report? "
        "Use appropriate medical terms"
    ),
)