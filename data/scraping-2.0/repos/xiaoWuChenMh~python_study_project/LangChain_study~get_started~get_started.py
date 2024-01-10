#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 配置环境变量
import os
from LangChain_study.common import ChatParam

os.environ["OPENAI_API_KEY"] = ChatParam.OPENAI_API_KEY
os.environ["OPENAI_API_BASE"] = ChatParam.OPENAI_API_BASE

# 初始化LLM模型
import langchain
from langchain.llms import OpenAI
llm = OpenAI(temperature=0.9)

# ************** 1、以文本生成文本 *******************************
text = "What would be a good company name for a company that makes colorful socks?"
# print(llm(text))

# ************** 2、LangChain模板来管理用户输入 **********************

# 2.1 创建一个模板
from langchain.prompts import PromptTemplate
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)
# print(prompt.format(product="colorful socks"))
# print("--------------------------")

# 2.2 通过 LLMChain 将模板 和 llm组成起来
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)
# print(chain.run("colorful socks"))

# ************** 3、代理：基于用户输入的动态调用链 **********************
desc = """ 
  --- 概念说明：
    工具：执行特定任务的功能。这可以是：Google 搜索、数据库查找、Python REPL、其他链。工具的接口目前是一个函数，期望将字符串作为输入，将字符串作为输出。
    LLM：为代理提供支持的语言模型。
    代理：要使用的代理。这应该是一个引用支持代理类的字符串。
  --- 需要安装的工具：
    谷歌搜索： pip install google-search-results # 需要注册（https://serpapi.com/）并获取key

"""

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

# 3.1 谷歌搜索工具的key
os.environ["SERPAPI_API_KEY"] ="68147803e91a55189eb9966ca5681405385790ecee78d8bf3d5a396e8ecaccdd"



# 3.2 让我们加载一些工具来使用。 请注意，`llm-math` 工具使用 LLM，因此我们需要将其传入
tools = load_tools(["serpapi", "llm-math"], llm=llm)


# 3.3 让我们使用工具、语言模型和我们要使用的代理类型来初始化代理。
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# 最后测试一下
# agent.run("What was the high temperature in SF yesterday in Fahrenheit? What is that number raised to the .023 power?")
agent.run("中国北京市昨天的高温是多少摄氏度？ 这个数字的 .023 次方是多少？")