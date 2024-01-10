import os

import openai
from dotenv import load_dotenv

import utils.limit_utils

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


def completions(model: str, creative: float, prompt: str):
    # 限制接口请求次数
    utils.limit_utils.check_limit()
    # 调用openai的completion接口
    completion = openai.Completion.create(model=model,
                                          prompt=prompt,
                                          # 0~2,default 1, 0:precision; 1:balance; 2:creative
                                          temperature=creative,
                                          stop=None)
    return completion
