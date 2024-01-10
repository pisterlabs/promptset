#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai, yaml
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMRequestsChain
from langchain.chains import TransformChain, SequentialChain

'''
通过langchain进行链式调用
LLMRequestsChain从互联网查询信息
LLMChain调用chatgpt找出信息
TransformChain将信息转化为json
'''

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]


# 通过query参数传入查询问题，查询结果会保存到requests_result中
# 然后通过chatgpt从查询结果中解析问题答案
def get_answer_from_google(question):
    inputs = {
        "query": question,
        "url": "https://www.google.com/search?q=" + question.replace(" ", "+")
    }
    result=requests_chain(inputs)
    print(result)
    print(result['output'])


# 解析天气信息，生成json
def parse_weather_info(weather_info: str) -> dict:
    # 将天气信息拆分成不同部分
    parts = weather_info.split('; ')

    # 解析天气
    weather = parts[0].strip()

    # 解析温度范围，并提取最小和最大温度
    temperature_range = parts[1].strip().replace('℃', '').split('～')
    temperature_min = int(temperature_range[0])
    temperature_max = int(temperature_range[1])

    # 解析风向和风力
    wind_parts = parts[2].split(' ')
    wind_direction = wind_parts[0].strip()
    wind_force = wind_parts[1].strip()

    # 返回解析后的天气信息字典
    weather_dict = {
        'weather': weather,
        'temperature_min': temperature_min,
        'temperature_max': temperature_max,
        'wind_direction': wind_direction,
        'wind_force': wind_force
    }

    return weather_dict


# 封装parse_weather_info，用于TransformChain
def transform_func(inputs: dict) -> dict:
    text = inputs["output"]
    return {"weather_info" : parse_weather_info(text)}


if __name__ == '__main__':
    get_api_key()

    template = """在 >>> 和 <<< 之间是来自Google的原始搜索结果.
    请把对于问题 '{query}' 的答案从里面提取出来，如果里面没有相关信息的话就说 "找不到"
    请使用如下格式：
    Extracted:<answer or "找不到">
    >>> {requests_result} <<<
    Extracted:"""

    PROMPT = PromptTemplate(
        input_variables=["query", "requests_result"],
        template=template,
    )
    llm_chain=LLMChain(llm=OpenAI(temperature=0), prompt=PROMPT)
    requests_chain = LLMRequestsChain(llm_chain)
    transformation_chain = TransformChain(input_variables=["output"], 
                                        output_variables=["weather_info"], transform=transform_func)
    final_chain = SequentialChain(chains=[requests_chain, transformation_chain], 
                                input_variables=["query", "url"], output_variables=["weather_info"])

    # 查看今天天气
    question = "今天上海的天气怎么样？"
    final_result = final_chain.run(question)
    print(final_result)
