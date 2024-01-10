import os

import pytest

from openfaker.bot import Bot
from openfaker.exceptions import OpenaiApiKeyNotFoundException
from openfaker.prompter import HelloEchoPrompter

dummy_message = 'hello'


def test_bot_with_echo():
    try:
        bot = Bot()
        response = bot.query(prompter=HelloEchoPrompter())
        assert response[:len(dummy_message)].lower() == dummy_message.lower(), "Echo response not correct."

    except OpenaiApiKeyNotFoundException:
        pytest.fail("OPENAI_API_KEY not retrieved correctly.")
