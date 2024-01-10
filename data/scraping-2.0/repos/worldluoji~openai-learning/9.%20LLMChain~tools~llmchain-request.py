
import openai, os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, LLMRequestsChain, TransformChain, SequentialChain
from langchain.llms import OpenAI

openai.api_key = os.environ.get("OPENAI_API_KEY")

template = """在 >>> 和 <<< 直接是来自Google的原始搜索结果.
请把对于问题 '{query}' 的答案从里面提取出来，如果里面没有相关信息的话就说 "找不到"
请使用如下格式：
Extracted:<answer or "找不到">
>>> {requests_result} <<<
Extracted:"""

PROMPT = PromptTemplate(
    input_variables=["query", "requests_result"],
    template=template,
)
requests_chain = LLMRequestsChain(llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=PROMPT))
question = "今天成都的天气怎么样？"
inputs = {
    "query": question,
    "url": "https://www.google.com/search?q=" + question.replace(" ", "+")
}
result=requests_chain(inputs)
print(result)
print(result['output'])

import re
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

# 直接使用代码调用转化为JSON
weather_info = "小雨; 10℃～15℃; 东北风 风力4-5级"
weather_dict = parse_weather_info(weather_info)
print(weather_dict)

# 使用TransformChain转化为JSON
# def transform_func(inputs: dict) -> dict:
#     text = inputs["output"]
#     return {"weather_info" : parse_weather_info(text)}

# transformation_chain = TransformChain(input_variables=["output"], 
#                                       output_variables=["weather_info"], transform=transform_func)

# final_chain = SequentialChain(chains=[requests_chain, transformation_chain], 
#                               input_variables=["query", "url"], output_variables=["weather_info"])
# final_result = final_chain.run(inputs)
# print(final_result)