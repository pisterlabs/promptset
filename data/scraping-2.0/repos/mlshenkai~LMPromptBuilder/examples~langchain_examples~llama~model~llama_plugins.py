# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/5/19 6:37 PM
# @File: llama_plugins
# @Email: mlshenkai@163.com
import os

os.environ[
    "SERPAPI_API_KEY"
] = "c81cb4bdce372462ef4ad8f9fa77e2192ac142fb0b27ca44a295b09ac63dd3d7"
from examples.langchain_examples.llama.model.llama_model import Llama
from src.LMBuilder.LLM.models import LlamaArgs
from langchain.agents import load_tools, initialize_agent, AgentType

llm = Llama(
    args=LlamaArgs(temperature=0.),
    model_name="/code-online/resources/base_model_resources/chinese_llama/merge_chinese_alpaca_llama_lora_13b",
)

tools = load_tools(["serpapi"])

agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

agent.run("What's the date today? What great events have taken place today in history?")