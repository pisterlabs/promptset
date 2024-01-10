import getpass
import openai
import os

"""
----------------------------------
MODEL_TURBO_16K = "gpt-3.5-turbo-16k"
MODEL_TURBO_0613 = "gpt-3.5-turbo-0613"
MODEL_GPT4 = "gpt-4-0613"
----------------------------------
"""


def get_key_OpenAI():
    openai.api_key = getpass.getpass(prompt='Введите секретный ключ для сервиса chatGPT: ')
    os.environ["OPENAI_API_KEY"] = openai.api_key


def about_gpt(data):
    pass


def experience_gpt(data):
    pass


def letter_gpt(data):
    pass
