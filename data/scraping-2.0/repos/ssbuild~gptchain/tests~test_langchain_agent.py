# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/11/27 13:52


from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI






tool_names = ["serpapi"]
tools = load_tools(tool_names)

print(tools)