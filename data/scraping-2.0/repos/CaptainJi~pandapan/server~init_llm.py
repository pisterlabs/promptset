#! /usr/bin/env python
"""
-*- coding: UTF-8 -*-
Project   : pandapan
Author    : Captain
Email     : qing.ji@extremevision.com.cn
Date      : 2023/11/4 7:54
FileName  : init_llm.py
Software  : PyCharm
Desc      : $END$
"""
import os

from dotenv import load_dotenv
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI


# LangChain相关导入


class SetupLangChain:
    '''
    用于初始化LangChain相关实例
    '''

    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    api_base = os.getenv('OPENAI_API_BASE')

    def __init__(self):
        pass

    def get_chat(self, temperature=0.5):
        chat = ChatOpenAI(temperature=temperature, openai_api_key=self.api_key, api_base=self.api_base)
        return chat

    def get_llm(self, temperature=0.5):
        llm = OpenAI(temperature=temperature, openai_api_key="", openai_api_base='http://192.168.1.93:7777/v1')
        return llm

    def get_chatglm(self, temperature=0.5):
        chat = OpenAI(temperature=temperature, openai_api_key="", openai_api_base='http://192.168.1.93:7777/v1',
                      model_name='chatglm2-6b-32k')
        return chat


# temperature用来设置大模型返回数据的随机性和创造性，较低的数值返回的数据就更贴近现实。
# chat = ChatOpenAI(temperature=0.9, openai_api_key=api_key, api_base=api_base)

class SetupAutoGen:
    '''
    用于初始化AutoGen相关实例
    '''

    def __init__(self):
        pass


if __name__ == '__main__':
    llm = SetupLangChain().get_chatglm()
