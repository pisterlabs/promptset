
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


def base_chat(text):
    openai.api_key = config['openai_api_key']
    openai.api_base = config['openai_base']
    prompt = f"请你扮演一个语文老师，精通中文语法句式表达，以下的句子可能是存在语病的，你必须思考和确认：1、细致的分析下面句子的每一个字、词、句式逻辑结构。2、有几个主语，几个宾语，动词能否搭配每个主语和宾语，主语、宾语是否清晰。3、关联词存不存在搭配错误，或者关联词的使用存不存在逻辑问题。4、句子前后是否矛盾或者重复，比如是和不是，全部和部分，基本和完全等组合不能同时出现。如果有语病则指出搭配上存在的错误：{text}  "
    print(prompt)
    # gpt-3.5-turbo-0301     
    completion = openai.ChatCompletion.create(
      model='gpt-3.5-turbo-16k', 
      messages=[
        {"role": "user", "content":prompt}
      ],
      temperature=0.4
    )

    #print(completion)    
    
    #print("completion.usage", completion.usage) 
    print(completion.choices[0].message['content'])  
    return completion.choices[0].message['content']