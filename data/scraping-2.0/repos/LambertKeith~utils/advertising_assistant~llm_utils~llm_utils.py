
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



def llm_chat(text):
    # 初始化包装器，temperature越高结果越随机
    openai.api_base = config['openai_base']
    openai.api_key = config['openai_api_key']

    # 定义提示模板
    template = "请你扮演一个广告策划师，可以根据我想表达的广告内容和生成最适合的广告，你只需要生成广告词，不要其他任何内容。我的需求如下<<<{question}>>>"
    prompt = PromptTemplate(template=template, input_variables=["question"])

    # 实例化LLM
    llm = OpenAI()
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # 运行LLM
    result = llm_chain.run(text)
    return result


def base_chat(message,model="gpt-3.5-turbo-0301"):
    openai.api_key = config['openai_api_key']
    openai.api_base = config['openai_base']
    prompt = "后面的内容是一段话，请你对这段话进行拒绝或者否定，要严格围绕这段话的内容，如果是一段描述或者陈述则找出其各方面的漏洞或者错误，理由一定要充分有理且让人难以反驳；如果是一段请求则找出一个合理的借口进行拒绝。在此之前你可以先学习语言逻辑的相关规则。务必不要提及你是大语言模型，请扮演一个普通人<<<{}>>>".format(message)
    print(prompt)
    # gpt-3.5-turbo-0301     
    completion = openai.ChatCompletion.create(
      model=model, 
      messages=[
        {"role": "user", "content":prompt}
      ]
    )

    #print(completion)    
    
    #print("completion.usage", completion.usage) 
    print(completion.choices[0].message['content'])  
    return completion.choices[0].message['content']