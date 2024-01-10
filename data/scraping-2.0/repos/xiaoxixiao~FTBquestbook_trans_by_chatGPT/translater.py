import json
import os
import time

import openai

import log


def translate_file(file_path: str):
    text = open(file_path, 'r').read()
    # 读取config获取api_key
    with open("config.yml", 'r', encoding='utf-8') as f__f:
        config = f__f.read()
    config_python_obj = json.loads(config)
    # 读取api_key
    api_key = config_python_obj['api_key']

    openai.api_key = api_key
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {"role": "system",
             "content": "你现在是一名minecraft模组汉化者，你需要将以下文本翻译成中文(请返回json列表格式)："},
            {"role": "user", "content": text}
        ]
    )

    print(completion.choices[0].message.content)


def translate_text(text: str) -> str:
    # 读取config获取api_key
    with open("config.yml", 'r', encoding='utf-8') as f__f:
        config = f__f.read()
    config_python_obj = json.loads(config)
    # 读取api_key
    api_key = config_python_obj['api_key']
    if api_key == "" or api_key == "填写你的api_key！！！":
        log.warn("请填写你的api_key！！！")
        exit(1)

    # 设置api_key
    openai.api_key = api_key
    try:
        result_trans = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0.5,
            messages=[
                {"role": "system", "content": ""},
                {"role": "user",
                 "content": text + "\n\n任务：汉化该列表\n输出：只输出汉化后的列表，格式为json列表\n要求：请输出正确的json格式列表，"
                                   "列表元素数量与原列表一致，格式与原列表一致，字典的键请保持原样。"},
            ]
        )

        translate_text_trans = result_trans.choices[0].message.content
    except Exception as e:
        log.info(f"可能出现错误，正在重试\n错误信息：{e}")
        how_time_wait = 0
        try:
            # 寻找错误信息中的等待时间 e 原文内容 -> {Rate limit reached for default-gpt-3.5-turbo in organization
            # org-WYh59YuHcaukvSuPLA88wR6P on requests per min. Limit: 3 / min. Please try again in 20s. Contact us
            # through our help center at help.openai.com if you continue to have issues. Please add a payment method
            # to your account to increase your rate limit. Visit https://platform.openai.com/account/billing to add a
            # payment method.} 正则表达式：(?<=Please try again in )\d+(?=s)
            time_ = int(e.args[0].split("Please try again in ")[1].split("s")[0])
            how_time_wait = time_ + 1
        except Exception as e:
            log.info(f"出现错误，但是无法获取等待时间\n错误信息：{e}")
            how_time_wait = 20
        time.sleep(how_time_wait)
        translate_text_trans = translate_text(text)
    return translate_text_trans


if __name__ == '__main__':
    translate_file("output/random-extracted.txt")
