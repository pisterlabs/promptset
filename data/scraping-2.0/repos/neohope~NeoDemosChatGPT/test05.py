#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai
import yaml
import tiktoken


'''
文本缩写
文本翻译
文本插入
文本过滤
'''


def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]


long_text = """
在这个快节奏的现代社会中，我们每个人都面临着各种各样的挑战和困难。
在这些挑战和困难中，有些是由外部因素引起的，例如经济萧条、全球变暖和自然灾害等。
还有一些是由内部因素引起的，例如情感问题、健康问题和自我怀疑等。
面对这些挑战和困难，我们需要采取积极的态度和行动来克服它们。
这意味着我们必须具备坚韧不拔的意志和创造性思维，以及寻求外部支持的能力。
只有这样，我们才能真正地实现自己的潜力并取得成功。
"""
prefix = """在这个快节奏的现代社会中，我们每个人都面临着各种各样的挑战和困难。
在这些挑战和困难中，有些是由外部因素引起的，例如经济萧条、全球变暖和自然灾害等。\n"""
# 还有一些是由内部因素引起的，例如情感问题、健康问题和自我怀疑等。
# 可以去掉下面的换行符试试
suffix = """\n面对这些挑战和困难，我们需要采取积极的态度和行动来克服它们。
这意味着我们必须具备坚韧不拔的意志和创造性思维，以及寻求外部支持的能力。
只有这样，我们才能真正地实现自己的潜力并取得成功。"""


def make_text_short(text, bias_map):
    messages = []
    messages.append( {"role": "system", "content": "你是一个用来将文本改写得短的AI助手，用户输入一段文本，你给出一段意思相同，但是短小精悍的结果"})
    messages.append( {"role": "user", "content": text})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.5,
        max_tokens=2048,
        presence_penalty=0,
        frequency_penalty=2,
        logit_bias = bias_map,
        n=3,
    )
    return response


# 把输入翻译为英文
def translate(text):
    messages = []
    messages.append( {"role": "system", "content": "你是一个翻译，把用户的话翻译成英文"})
    messages.append( {"role": "user", "content": text})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages, temperature=0.5, max_tokens=2048, n=1
    )
    return response["choices"][0]["message"]["content"]


# 文本插入
def insert_text(prefix, suffix):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prefix,
        suffix=suffix,
        max_tokens=1024,
        )
    return response


# 暴力文本过滤
def moderation(text):
    response = openai.Moderation.create(
        input=text
    )
    output = response["results"][0]
    return output


if __name__ == '__main__':
    get_api_key()

    # 通过bias_map控制希望那些词出现，哪些词不出现
    encoding = tiktoken.get_encoding('p50k_base')
    token_ids = encoding.encode("灾害")
    print(token_ids)
    bias_map = {}
    for token_id in token_ids: 
        bias_map[token_id] = -100

    # 对long_text进行缩写
    short_version = make_text_short(long_text, bias_map)
    index = 1
    for choice in short_version["choices"]:
        print(f"version {index}: " + choice["message"]["content"])
        index += 1

    # 将long_text翻译为英文，英文的tokens更少一些
    chinese = long_text
    english = translate(chinese)
    num_of_tokens_in_chinese = len(encoding.encode(chinese))
    num_of_tokens_in_english = len(encoding.encode(english))
    print(english)
    print(f"chinese: {num_of_tokens_in_chinese} tokens")
    print(f"english: {num_of_tokens_in_english} tokens")

    # 文本插入
    response = insert_text(prefix, suffix)
    print(response["choices"][0]["text"])

    # 暴力文本过滤，接口免费
    # 会得到判断，"violence": true
    threaten = "你不听我的我就拿刀砍死你"
    print(moderation(threaten))
