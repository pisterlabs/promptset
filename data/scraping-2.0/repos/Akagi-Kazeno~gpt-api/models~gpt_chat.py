import os

import openai
from dotenv import load_dotenv

import utils.json_utils
import utils.limit_utils

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


def chat(model: str, creative: float, content: str):
    # 限制接口请求次数
    utils.limit_utils.check_limit()
    # 请求openai的chat接口
    completion = openai.ChatCompletion.create(model=model,
                                              messages=[
                                                  # Change the prompt parameter to the messages parameter
                                                  {'role': 'user', 'content': content}
                                              ],
                                              # 0~2,default 1, 0:precision; 1:balance; 2:creative
                                              temperature=creative,
                                              stop=None)
    return completion
    # return chat_completion['choices'][0]['message']['content']


def chat_completion(model: str, message: list):
    # 限制接口请求次数
    utils.limit_utils.check_limit()
    # 请求openai的连续对话接口
    if type(message) != list:
        raise Exception('输入内容应如下格式:\neg:[{"role": "system", "content": "You are a helpful assistant."},\n{"role": '
                        '"user", "content": "Who won the world series in 2020?"},\n{"role": "assistant", "content": '
                        '"The Los Angeles Dodgers won the World Series in 2020."},\n{"role": "user", "content": '
                        '"Where was it played?"}]')
    for roles in message:
        if roles['role'] not in {"system", "user", "assistant"}:
            raise Exception('输入角色仅支持"system","user","assistant"')
    chat = openai.ChatCompletion.create(model=model,
                                        messages=message,
                                        stop=None)
    return chat


def use_chat_completion(model: str, message: str):
    # 限制接口请求次数
    utils.limit_utils.check_limit()
    message_list: list = utils.json_utils.create_user_chat_message_list(message)
    chat = openai.ChatCompletion.create(model=model,
                                        messages=message_list,
                                        stop=None)
    return chat
