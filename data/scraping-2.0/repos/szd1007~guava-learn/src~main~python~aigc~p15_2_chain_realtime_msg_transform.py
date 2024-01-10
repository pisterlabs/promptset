import openai, os
from langchain.prompts import PromptTemplate
from langchain.llms import  OpenAI
from langchain.chains import LLMChain
from  langchain.chains import SequentialChain
from langchain.chains import LLMRequestsChain
openai.api_key = os.environ.get("OPENAI_API_KEY")
import re
import json
from langchain.chains import TransformChain, SequentialChain

#https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
#llm = OpenAI(model_name="gpt-4-1106-preview", max_tokens=2048, temperature=0.5)
llm = OpenAI(model_name="text-davinci-003", max_tokens=2048, temperature=0.5)

## gpt4代码生成
def extract_temperatures(text):
    # 使用正则表达式查找白天的最高气温和夜间的最低气温
    max_temperatures = re.findall(r'最高气温(\d+)℃', text)
    min_temperatures = re.findall(r'最低气温(-?\d+)℃', text)

    # 将找到的温度信息转换为字典格式
    temperature_data = {
        "max_temperature": [int(temp) for temp in max_temperatures],
        "min_temperature": [int(temp) for temp in min_temperatures]
    }
    return temperature_data
def transform_func(inputs: dict) -> dict:
    text = inputs["output"]
    print(text)
    out= {"weather_info": extract_temperatures(text)}
    print(out)
    return out

transform_chain = TransformChain(input_variables=["output"], output_variables=["weather_info"], transform=transform_func)


template = """在 >>> 和 <<< 直接是来自Google的原始搜索结果.
请把对于问题 '{query}' 的答案从里面提取出来， 如果没有相关信息的话就说 "找不到"
请使用如下格式：
Extracted:<answer or "找不到">
>>> {requests_result} <<<
Extracted:
"""
PROMPT = PromptTemplate(
    input_variables=['query', "requests_result"],
    template=template
)
requests_chain =  LLMRequestsChain(llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=PROMPT),  output_key="output")
question = "今天北京的天气怎么样"
inputs = {
    "query": question,
    "url": "https://www.google.com/search?q=" + question.replace(" ", "+")
}

final_chain = SequentialChain(chains=[requests_chain, transform_chain],
                              input_variables=["query", "url"],
                              output_variables=["weather_info"]
                              )


result = final_chain.run(inputs)
print("sdfsdflksdjlj")
print(result)
