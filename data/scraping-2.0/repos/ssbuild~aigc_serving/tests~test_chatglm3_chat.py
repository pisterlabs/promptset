# coding=utf8
# @Time    : 2023/10/31 20:35
# @Author  : tk
# @FileName: test_chatglm3_tools
import copy
import json
import openai

openai.api_key = "EMPTY"
openai.api_base = "http://192.168.16.157:8081/v1"
model = "chatglm3-6b"


data = {
    "model": model,
    "adapter_name": None, # lora头
    "top_p": 0.8,
    "temperature": 1.0,
    "frequency_penalty": 1.01,
    "stream": False,
    "nchar": 1,# stream 字符
    "n": 1, # 返回 n 个choices
    # "max_tokens": 2048,
    "stop": ["Observation:"]
}

data["messages"] = [{"role": "user", "content": "今天天气怎么样"}]

completion = openai.ChatCompletion.create(**data)
print(completion.choices[0].message.content)