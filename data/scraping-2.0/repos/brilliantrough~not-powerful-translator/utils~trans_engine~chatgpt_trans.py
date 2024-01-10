"""
Author: brilliantrough pzyinnju@163.com
Date: 2023-06-29 18:37:40
LastEditors: brilliantrough pzyinnju@163.com
LastEditTime: 2023-07-10 20:15:29
FilePath: \not-powerful-translator\chatgpt_trans.py
Description: 定义 ChatGPT 翻译类，其中根据翻译类型定义了两个翻译方法，分别为中文翻译为英文和英文翻译为中文；并以此定义了两个翻译提示。
同时还安排了两种翻译方式，一种是普通的请求方式，一种是流的方式，流的方式可以实现实时翻译，默认使用此方式。

Copyright (c) 2023 by {brilliantrough pzyinnju@163.com}, All Rights Reserved. 
"""
import requests
from requests.exceptions import RequestException
import openai
import os

prompt_zh2en = "You should act as an English translator, spelling corrector and improver. The user will speak to you in any language and you will detect the language, translate it and answer in the corrected and improved version of my text, in English. You should only reply the correction, the improvements and nothing else, do not write explanations. Your goal is to ensure that the translation is as smooth and natural as possible, while not changing the meaning of the text."

prompt_en2zh = "You should act as a Chinese translator, spelling corrector and improver. The user will speak to you in any language and you will detect the language, translate it and answer in the corrected and improved version of my text, in Chinese. You should only reply the correction, the improvements and nothing else, do not write explanations. Your goal is to ensure that the translation is as smooth and natural as possible, while not changing the meaning of the text."

prompt_sst = """
You need to translate the English I provide you into Chinese. Note that the English translation I provide you is marked with each block separated by carriage returns. Please translate it for me by block, retaining the detailed structure, that is, retaining the original carriage returns.
In addition, some texts are recognized from screenshots, and there will be some recognition errors. For example, the dots in the screenshots are recognized as e, etc. Please distinguish and translate them properly, and still maintain the same structure.
"""


class ChatGPT:
    def __init__(self):
        self.headers = {"Authorization": f"Bearer {openai.api_key}"}
        self.proxies = {
            "https": "http://127.0.0.1:7890",
            "http": "http://127.0.0.1:7890",
        }
        self.url = f"{openai.api_base}/chat/completions"
        self.data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": ""},
            ],
        }
        self.retry_nums = 3
        
    def setProxy(self, address: str = "127.0.0.1", port: int = 7890, unset: bool = False):
        if unset:
            self.proxies = None
            return
        self.proxies = {"https": f"http://{address}:{port}", "http": f"http://{address}:{port}"}

    def chatgpt(self, prompt: str, text: str, stream: bool = False) -> tuple:
        """翻译文本

        Args:
            prompt (str): 提示 ChatGPT 翻译类型
            text (str): 待翻译文本
            stream (bool, optional): 是否采用流的方式输出翻译结果. Defaults to False.

        Returns:
            tuple: (翻译结果, 状态)
        """
        if stream:
            return self.chatgpt_stream(prompt, text)
        i = 0
        self.data["messages"][0]["content"] = prompt
        self.data["messages"][1]["content"] = text
        while i < self.retry_nums:
            try:
                response = requests.post(
                    self.url,
                    verify=False,
                    headers=self.headers,
                    proxies=self.proxies,
                    json=self.data,
                    timeout=40,
                )
                if response.status_code == 200:
                    result = response.json()["choices"][0]["message"]["content"]
                    return result, "成功"
                else:
                    print("连接失败，状态码为 ", response.status_code)
                    i += 1
            except RequestException as e:
                print("连接失败，错误信息为 ", e)
                i += 1
        return None, "失败"

    def chatgpt_stream(self, prompt: str, text: str) -> tuple:
        """用流的方式进行翻译

        Args:
            prompt (str): 提示 ChatGPT 翻译类型
            text (str): 待翻译文本

        Returns:
            tuple:  (翻译结果, 状态)
        """
        i = 0
        while i < self.retry_nums:
            try:
                openai.proxy = self.proxies['http'] if self.proxies else None
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": text},
                    ],
                    stream=True,
                    timeout=10,
                )
                return completion, "成功"
            except Exception as e:
                print("连接失败，错误信息为 ", e)
                i += 1
        return None, "失败"

    def chatgpt_zh2en(self, text: str, stream: bool = False, sst: bool = False) -> tuple:
        return self.chatgpt(prompt_zh2en, text, stream=stream)

    def chatgpt_en2zh(self, text: str, stream: bool = False, sst: bool = False) -> tuple:
        return self.chatgpt(prompt_sst, text) if sst else self.chatgpt(prompt_en2zh, text, stream=stream)
