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
llm = OpenAI(model_name="text-davinci-002", n=2, best_of=2)

# 利用内存缓存返回结果:结果会缓存到内存中，在此问相同的问题会直接从内存中读取结果
from langchain.cache import InMemoryCache
langchain.llm_cache = InMemoryCache()
print(llm("Tell me a joke"))
print("----------- 在此提问 ————————————————")
print(llm("Tell me a joke"))

# 支持的缓存方式：SQLite 、Redis（支持语义缓存） 、GPTCache（精确匹配缓存或基于语义相似性缓存）、Momento缓存、SQLAlchemy 缓存