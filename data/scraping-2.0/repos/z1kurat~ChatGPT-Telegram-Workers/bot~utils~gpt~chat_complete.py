import os

from httpx import ReadTimeout
from openai_async import openai_async

from bot.parameters.gpt_parameters import MODEL, TEMPERATURE, MAX_VALUE_COUNT, TIME_OUT, STOP
from bot.structures.erorrs import TooManyRequests, SomethingWentWrong


async def chat_complete(user_dialog) -> [str, int]:
    """
    Getting a response from GPT
    :param user_dialog: users history
    :return: response and token
    """
    try:
        completion = await openai_async.chat_complete(
            os.getenv("OPENAI_KEY"),
            timeout=TIME_OUT,
            payload={
                "model": MODEL,
                "messages": user_dialog,
                "temperature": TEMPERATURE,
                "stop": STOP,
                "n": MAX_VALUE_COUNT
            }
        )

        response = completion.json()["choices"][0]["message"]["content"]
        token_usage = completion.json()['usage']['total_tokens']

        return response, token_usage

    except Exception as err:
        if completion.status_code == 429:
            raise TooManyRequests(err)

        if isinstance(err, ReadTimeout):
            raise

        raise SomethingWentWrong(err)
