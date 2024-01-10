# -*- coding: utf-8 -*-
# Date       : 2023/11/5
# Author     : Jiayi Zhang
# email      : didi4goooogle@gmail.com
# Description: LLM engine

import openai
import time
import json


class OpenAILLM:
    def __init__(self, model="gpt-3.5-turbo-1106", temperature=0.7, timeout=60):

        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def get_response(self, prompt: str, json_mode=False, max_tokens=500, retries=5):
        print(f"current prompt :{prompt}")
        print(f"current json mode: {json_mode}")
        """
        JSON_MODE开启之后直接返回JSON格式的结果，否则返回字符串
        """
        response_type = "text" if not json_mode else "json_object"
        for i in range(retries):
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    response_format={"type": response_type}
                )
                if json_mode:
                    result = response.choices[0].message.content
                    print(f"Json mode result: {result}")
                    result = json.loads(response.choices[0].message.content)
                else:
                    result = response.choices[0].message.content
                return result
            except openai.error.RateLimitError:
                print("Occur RateLimitError, sleep 20s")
                time.sleep(20)
                print("Rate limit retry")
            except openai.error.AuthenticationError:
                print("Please check your openai api key")
            except Exception as e:
                print(f"{__name__} occurs: {e}")


def prompt_load(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except FileNotFoundError:
        content = "File not found at the specified path."

    return content
