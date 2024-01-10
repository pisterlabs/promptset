from collections import namedtuple
from typing import Tuple, Type

import openai
from aiogram.types import Message
from openai.openai_object import OpenAIObject

from bot_ai.handlers.cb_handler import standard_settings

StandardSettings: Type['StandardSettings'] = namedtuple('StandardSettings', ['model', 'temperature', 'max_tokens'])


# standard_settings: StandardSettings = StandardSettings(
#     model='text-davinci-003',
#     temperature=1,
#     max_tokens=3500
# )
#
#
# def standard_bot_model(message: Message) -> OpenAIObject:
#     response: OpenAIObject = openai.Completion.create(
#         model=standard_settings.model,
#         prompt=message.text,
#         temperature=standard_settings.temperature,
#         max_tokens=standard_settings.max_tokens,
#     )
#     return response


def standard_bot_model(message: Message) -> Tuple[OpenAIObject, str]:
    response: OpenAIObject = openai.Completion.create(
        model=standard_settings.get('model'),
        prompt=message.text,
        temperature=standard_settings.get('temperature'),
        max_tokens=standard_settings.get('max_tokens'),
    )
    msg: str = response.choices[0].text
    return response, msg


def artist_bot_model():
    ...


def coder_bot_model():
    ...


def companion_bot_model(message: Message) -> Tuple[OpenAIObject, str]:
    response: OpenAIObject = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"{message.text}"}],
    )
    msg: str = response.choices[0].message.content
    return response, msg


def translator_bot_model():
    ...


OPENAI_MODELS = {
    'set_artist': artist_bot_model,
    'set_coder': coder_bot_model,
    'set_translator': translator_bot_model,
    'set_companion': companion_bot_model,
    'set_standard': standard_bot_model
}
