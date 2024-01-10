import logging
import random

import openai
from openai.error import OpenAIError

from constants import OPENAI_API_KEY
from utils.env_handler import get_env_var


def generate_compliment(name, adjectives, attributes):
    """Generate a compliment using OpenAI API."""
    attribute = random.choice(attributes)
    adjective = random.choice(adjectives)
    prompt = (
        f"Craft a compliment for {name}, highlighting their {adjective} {attribute}."
    )

    openai.api_key = get_env_var(OPENAI_API_KEY)
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0.9,
            max_tokens=100,
        )
    except OpenAIError as e:
        logging.error(f"Failed to generate compliment: {str(e)}")
        return None  # or raise the exception, depends on your use case

    if response and response.choices:
        return response.choices[0].text.strip()  # type: ignore
    else:
        logging.warning("No compliment generated from the OpenAI API response.")
        return None
