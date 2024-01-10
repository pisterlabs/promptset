
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
#os.environ["https_proxy"] = "http://localhost:7890"
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
    os.environ["http_proxy"] = "http://localhost:7890"
    os.environ["https_proxy"] = "http://localhost:7890"
    openai.api_key = config['openai_api_key']
    openai.api_base = config['openai_base']
    prompt = f"请你根据以下要求为我一个人说一段单口相声：结合{message[0]}，{message[1]}，{message[2]}，{message[3]}，{message[4]}，{message[5]}以及{message[6]}这几个数字，请你从祝贺、恭喜我发财的角度（也就是我还没有发财），结合这几个数字说一段吉利话，你的表演请直接从第一个数字开始，不要打招呼，你的观众只有我一个人，你只用称呼我一个人就行"
    print(prompt)
    # gpt-3.5-turbo-0301     
    completion = openai.ChatCompletion.create(
      model=model, 
      messages=[
        {"role": "user", "content":prompt}
      ]
    )

    print(completion.choices[0].message['content'])  
    return completion.choices[0].message['content']