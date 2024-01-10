
import getpass
import os
import openai
import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
from llm_utils.read_config import load_config
from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI


config = load_config() 
os.environ["https_proxy"] = "http://localhost:7890"
os.environ["OPENAI_API_BASE"] = config['openai_base']
os.environ["OPENAI_API_KEY"] = config['openai_api_key']



def llm_chat(text, format1):
    # 初始化包装器，temperature越高结果越随机
    openai.api_base = config['openai_base']
    openai.api_key = config['openai_api_key']

    # 定义提示模板
    template = "现在请你扮演一个格式转换器，输入的内容按照指定格式进行转换。你只用回答转换之后的文本内容，格式为：\n {format} \n，内容为：{text}  "
    prompt = PromptTemplate(template=template, input_variables=["format", "text"])

    # 实例化LLM
    llm = OpenAI()
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # 运行LLM
    result = llm_chain.run({"format": format1, "text": text})
    return result


def base_chat(format1, text):
    openai.api_key = config['openai_api_key']
    openai.api_base = config['openai_base']
    prompt = f"现在请你扮演一个格式转换器，输入的内容按照指定格式进行转换。你只用回答转换之后的文本内容，格式为：\n {format1} \n，内容为：{text}  "
    print(prompt)
    # gpt-3.5-turbo-0301     
    completion = openai.ChatCompletion.create(
      model='gpt-3.5-turbo-0301', 
      messages=[
        {"role": "user", "content":prompt}
      ]
    )

    #print(completion)    
    
    #print("completion.usage", completion.usage) 
    print(completion.choices[0].message['content'])  
    return completion.choices[0].message['content']