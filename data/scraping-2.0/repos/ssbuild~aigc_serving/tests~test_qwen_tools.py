# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/11/29 13:35

import json
import math

import openai
from openai.openai_object import OpenAIObject
from scipy import integrate


# 新版本
openai.api_key = "112233"
openai.api_base = "http://192.168.2.180:8081/v1"
openai.api_base = "http://106.12.147.243:8082/v1"
model = "chatglm2-6b-int4"
model = "qwen-7b-chat-int4"
model = "Qwen-14B-Chat"



query = '现在给我画个五彩斑斓的黑。'

functions = [
    {
        'name_for_human':
        '夸克搜索',
        'name_for_model':
        'quark_search',
        'description_for_model':
        '夸克搜索是一个通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。',
        'parameters': [{
            'name': 'search_query',
            'description': '搜索关键词或短语',
            'required': True,
            'schema': {
                'type': 'string'
            },
        }],
    },
    {
        'name_for_human':
        '通义万相',
        'name_for_model':
        'image_gen',
        'description_for_model':
        '通义万相是一个AI绘画（图像生成）服务，输入文本描述，返回根据文本作画得到的图片的URL',
        'parameters': [{
            'name': 'query',
            'description': '中文关键词，描述了希望图像具有什么内容',
            'required': True,
            'schema': {
                'type': 'string'
            },
        }],
    },
]


#第一步
messages = [{"role": "user","content": query}]
response = openai.ChatCompletion.create(
    model=model,
    messages=messages,
    functions=functions,
    temperature=0,
    # functions=self.functions,
    stop=["Observation:","Observation"]
)

messages.append({
    "role": "assistant",
    "content": response.choices[0].message.content
})

print(response.choices[0].message.content)



# 第二步

fake_result = '{"status_code": 200, "request_id": "3d894da2-0e26-9b7c-bd90-102e5250ae03", "code": null, "message": "", "output": {"task_id": "2befaa09-a8b3-4740-ada9-4d00c2758b05", "task_status": "SUCCEEDED", "results": [{"url": "https://dashscope-result-sh.oss-cn-shanghai.aliyuncs.com/1e5e2015/20230801/1509/6b26bb83-469e-4c70-bff4-a9edd1e584f3-1.png"}], "task_metrics": {"TOTAL": 1, "SUCCEEDED": 1, "FAILED": 0}}, "usage": {"image_count": 1}}'


messages.append({
    "role": "function",
    "content": fake_result,
})

response = openai.ChatCompletion.create(
    model=model,
    messages=messages,
    functions=functions,
    temperature=0,
    # functions=self.functions,
    stop=["Observation:","Observation"]
)
print(response.choices[0].message.content)

