#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai, yaml

"""
根据提示，生成代码
根据提示，生成单元测试代码
"""

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]


# gpt封装
def gpt35(prompt, model="text-davinci-002", temperature=0.4, max_tokens=1000, 
          top_p=1, stop=["\n\n", "\n\t\n", "\n    \n"]):
    response = openai.Completion.create(
        model=model,
        prompt = prompt,
        temperature = temperature,
        max_tokens = max_tokens,
        top_p = top_p,
        stop = stop
        )
    message = response["choices"][0]["text"]
    return message


# 按要求生成一段代码
def try_copilot():
    prompt = """
用Python写一个函数，进行时间格式化输出，比如：
输入  输出
1    1s
61    1min1s
要求仅需要格式化到小时(?h?min?s)，即可
"""
    response = gpt35(prompt)
    return response, prompt

"""
def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours > 0:
        return f"{hours}h{minutes}min{seconds}s"
    elif minutes > 0:
        return f"{minutes}min{seconds}s"
    else:
        return f"{seconds}s"
"""


# 按要求生成一段代码
def try_copilot_uint_test():
    prompt = """
可以用一下的Python代码实现时间格式化输出：

def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours > 0:
        return f"{hours}h{minutes}min{seconds}s"
    elif minutes > 0:
        return f"{minutes}min{seconds}s"
    else:
        return f"{seconds}s"

请为这个程序用pytest写一个单元测试
"""
    response = gpt35(prompt)
    return response, prompt


"""
import pytest

def test_format_time():
    assert format_time(1) == "1s"
    assert format_time(59) == "59s"
    assert format_time(60) == "1min0s"
    assert format_time(61) == "1min1s"
    assert format_time(3600) == "1h0min0s"
    assert format_time(3661) == "1h1min1s"
"""


if __name__ == '__main__':
    get_api_key()

    # 用chatgpt生成一段代码
    code = try_copilot()
    print(code)

    # 用chatgpt生成单元测试
    test_code = try_copilot_uint_test()
    print(test_code)

