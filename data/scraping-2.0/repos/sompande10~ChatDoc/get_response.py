from openai import OpenAI
import openai
import os
from dotenv import load_dotenv

load_dotenv()


def get_basic_response(message_list):
    client = OpenAI()

    chatresponse = client.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages = message_list,
    )

    response = str(chatresponse.choices[0].message.content)

    return response