from enum import Enum
from typing import Optional, Union, Any

import openai
from dotenv import load_dotenv
from openai import Completion
import os


class Models(str, Enum):
    MODEL_ENGINE = "gpt-3.5-turbo"
    MODEL_ENGINE_LARGE = "gpt-4"


class ModelIntHyperParams(int, Enum):
    MAX_TOKENS = 500
    MODEL_THRESHOLD = 3000


class ModelFloatHyperParams(float, Enum):
    TEMPERATURE = 0.01


load_dotenv()

openai.api_key = os.environ.get("OPENAI_KEY")


def call_model_endpoint(
    prompt: str,
    model: Optional[str] = None,
    max_tokens: int = 500,
):
    if not model:
        model = (
            Models.MODEL_ENGINE_LARGE
            if (len(prompt.split()) + max_tokens) > ModelIntHyperParams.MODEL_THRESHOLD
            else Models.MODEL_ENGINE
        )
    try:
        completion: Completion = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            n=1,
            max_tokens=max_tokens,
            temperature=ModelFloatHyperParams.TEMPERATURE,
        )
        saved_text = (
            completion.choices[0]
            .message.content.replace("• ", "* ")
            .replace("- ", "* ")
        )
    except openai.error.InvalidRequestError as exception:
        saved_text = exception.user_message
    return saved_text
