"""
用来连接openai
"""

import openai
from app.core.config import Settings


def send_msg_to_openai(prompt):
    errors = "超时错误，请重试！"
    try:
        openai.api_key = Settings.Openai["secret"]
        response = openai.ChatCompletion.create(
            model=Settings.Openai["model"],
            messages=prompt
        )
        res = response["choices"][0]["message"]["content"]
        if res != "":
            return res
        else:
            return errors
    except:
        return errors
