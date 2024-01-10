import contextvars

from dotenv import load_dotenv
import openai
import os


class ChatGPTAPI(object):
    def __init__(self):
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.proxy = "http://127.0.0.1:7890"

    def chat(self, message):
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=message,
                user="test"
            )
        except openai.error.RateLimitError:
            print("访问过于频繁")
        message.append({"role": "assistant", "content": completion.choices[0].message.content})
        # 输出生成的内容
        return message, completion
