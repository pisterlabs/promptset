"""
* Pizza delivery prompt example
* run example by writing `python example/pizza.py` in your console
"""
import re

from openai import OpenAI, OpenAIError
from prompt_toolkit.validation import ValidationError

from .prompts import PROMPTS

OPENAI_MODELS = [
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k" "gpt-4-1106-preview",
    "gpt-4",
    "gpt-4-32k",
]


def validate_openai_key(api_key):
    regex = re.compile(r"^sk-[a-zA-Z0-9]+$")
    if not regex.match(api_key):
        raise ValidationError(
            message="Please enter a key in the format sk-XXXXXXXXXXXXXXXX",
            cursor_position=len(api_key),
        )

    openai_client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=api_key
    )

    try:
        test_messages = [
            {
                "role": "system",
                "content": "Test message",
            }
        ]
        chat_completion = openai_client.chat.completions.create(
            messages=test_messages,
            model="gpt-3.5-turbo",
            stream=True,
        )
        for chunk in chat_completion:
            if chunk:
                break

    except OpenAIError as e:
        raise ValidationError(
            message=f"Please enter a valid key encountered error: {e}",
            cursor_position=len(api_key),
        ) from e

    return True


def get_settings_fields(defaults=None):
    """Get the configuration fields."""
    defaults = defaults or {}
    questions = [
        {
            "type": "password",
            "name": "openai_api_key",
            "message": "Insert the OpenAI API key",
            "validate": validate_openai_key,
            "default": defaults.get("openai_api_key", ""),
        },
        {
            "type": "list",
            "name": "openai_model",
            "message": "Select the model you want to use",
            "choices": [
                {
                    "name": f"{model_name}",
                    "value": model_name,
                }
                for model_name in OPENAI_MODELS
            ],
            "default": defaults.get("model", "gpt-3.5-turbo"),
        },
        {
            "type": "list",
            "name": "prompt",
            "message": "Select the prompt you want to use",
            "choices": [
                {
                    "key": prompt_name.lower()[0],
                    "name": f"{prompt_name}",
                    "value": prompt_name,
                    "description": prompt,
                }
                for prompt_name, prompt in PROMPTS.items()
            ],
            "default": defaults.get("prompt", "generic"),
        },
        {
            "type": "input",
            "name": "user_icon",
            "message": "What should the user terminal block look like?",
            "default": defaults.get("user_icon", "🙍"),
        },
        {
            "type": "input",
            "name": "assistant_icon",
            "message": "What should the assistant terminal block look like?",
            "default": defaults.get("assistant_icon", "🤖"),
        },
        {
            "type": "confirm",
            "name": "verbose",
            "message": "Do you want to print debug information?",
            "default": defaults.get("verbose", False),
        },
    ]
    return questions
