#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai, yaml
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory

'''
SummaryMemory通过总结之前的对话，提升对话效果
'''

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]


if __name__ == '__main__':
    get_api_key()

    llm = OpenAI(temperature=0)
    memory = ConversationSummaryMemory(llm=OpenAI())

    prompt_template = """你是一个中国厨师，用中文回答做菜的问题。你的回答需要满足以下要求:
    1. 你的回答必须是中文
    2. 回答限制在100个字以内

    {history}
    Human: {input}
    AI:"""
    prompt = PromptTemplate(
        input_variables=["history", "input"], template=prompt_template
    )
    # 开始了verbose模式
    conversation_with_summary = ConversationChain(
        llm=llm, 
        memory=memory,
        prompt=prompt,
        verbose=True
    )
    conversation_with_summary.predict(input="你好")
    conversation_with_summary.predict(input="鱼香肉丝怎么做？")
    memory.load_memory_variables({})
    conversation_with_summary.predict(input="那蚝油牛肉呢？")
