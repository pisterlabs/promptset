#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai
import yaml


'''
语音转文字
而且可以结果直接转换为英文
'''

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]


if __name__ == '__main__':
    get_api_key()

    # 语音转文字
    promot = """
    这是一段Onboard播客的内容，里面会聊到ChatGPT以及PALM这个大语言模型。这个模型也叫做Pathways Language Model。
    """
    audio_file= open("./data/podcast_clip.mp3", "rb")
    # response_format是输出格式，可以是JSON、TEXT、SRT、VTT等
    # language表示语言
    # temperature表示结果的随机性
    transcript = openai.Audio.transcribe("whisper-1", audio_file, response_format="json", promot=promot)
    print(transcript['text'])

    # 通过translate直接翻译结果
    translated_prompt="""This is a podcast discussing ChatGPT and PaLM model. 
    The full name of PaLM is Pathways Language Model."""
    transcript = openai.Audio.translate("whisper-1", audio_file, response_format="json", prompt=translated_prompt)
    print(transcript['text'])
