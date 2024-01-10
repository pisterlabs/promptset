#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai
import yaml
import subprocess

"""
增加流式输出
"""

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]

# 流式输出
# 生成的模型为curie:ft-bothub-ai:ultraman-2023-04-04-03-03-26
def write_a_story_by_stream(prompt):
    response = openai.Completion.create(
        model="curie:ft-bothub-ai:ultraman-2023-04-04-03-03-26",
        prompt=prompt,
        temperature=0.7,
        max_tokens=2000,
        stream=True,
        top_p=1,
        stop=["."])
    return response


if __name__ == '__main__':
    get_api_key()

    # 获取调优模型清单
    # 可以看到模型id没有变化
    subprocess.run('openai api fine_tunes.list'.split())

    # 流式输出生成新的故事
    response = write_a_story_by_stream("汉,冰冻大海,艰难 ->\n")
    for event in response:
        event_text = event['choices'][0]['text']
        print(event_text, end = '')
