import logging

from aiogram import types
from openai import APIError
from openai.error import RateLimitError, APIConnectionError, InvalidRequestError, AuthenticationError, Timeout, \
    ServiceUnavailableError

GPTErrors = (APIError, RateLimitError, APIConnectionError, InvalidRequestError, AuthenticationError,
             ServiceUnavailableError, Timeout)

async def handle_gpt_errors(e, msg_for_answer: types.Message):
    try:
        raise e
    except APIError:
        err_msg = f"Произошла ошибка API Error: {e}. Попробуйте задать вопрос снова"
    except Timeout:
        err_msg = f"Произошла ошибка Timeout Error: {e} Попробуйте задать вопрос снова"
    except RateLimitError:
        err_msg=f"Произошла ошибка Rate Limit Error: {e} Попробуйте задавать вопросы пореже"
    except APIConnectionError:
        err_msg=f"Произошла ошибка Connection Error: {e} Проверьте подключение к сети"
    except InvalidRequestError:
        err_msg=f"Произошла ошибка Invalid Request Error: {e} Программист что-то накосячил. Сообщите об этом ему"
    except AuthenticationError:
        err_msg=f"Произошла ошибка Authentication Error: {e} Программист накосячил с ключами. Сообщите об этом ему"
    except ServiceUnavailableError:
        err_msg=f"Произошла ошибка Service Unavailable Error: {e} Сервис OpenAI недоступен. Надо подождать."
    finally:
        await msg_for_answer.answer(err_msg)
        logging.error(err_msg)


