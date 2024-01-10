from django.conf import settings
from .prompts import PROMPT_LIBRARY

import openai

openai.api_key = settings.OPENAI_API_KEY


def generate_request(content, endpoint):
    if endpoint not in PROMPT_LIBRARY.keys():
        raise Exception("Invalid endpoint")

    prompt = PROMPT_LIBRARY[endpoint].format(sentence=content)

    return openai.Completion.create(
        prompt=prompt,
        # Model
        model=settings.MODEL,
        # Parameters
        temperature=settings.TEMPERATURE,
        max_tokens=settings.MAX_TOKENS,
        top_p=settings.TOP_P,
        frequency_penalty=settings.FREQUENCY_PENALTY,
        presence_penalty=settings.PRESENCE_PENALTY,
        n=settings.SAMPLES,
    )
