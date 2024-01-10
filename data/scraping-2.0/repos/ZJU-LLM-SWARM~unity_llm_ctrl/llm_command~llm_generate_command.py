# -*- coding:utf-8 -*-
import os
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
#路径规划
import json
os.environ["OPENAI_API_KEY"] = "sk-vjUyMBvE9oKQI8ul7dBa8cBd6a3645AcAeB4B99dAdAa1c89"
openai_api_key = os.environ["OPENAI_API_KEY"]
os.environ['SERPAPI_API_KEY'] = "ecba56657bab25863cd52e445cd4f65214afe50b1507f2a10e423293877f0516"

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def gerate_langchain(input):
    # # 加载一些要使用的工具
    llm = OpenAI(model_name="gpt-4", openai_api_key=openai_api_key)

    tools = load_tools(["requests_all"], llm=llm)

    # 初始化 Agents
    agent = initialize_agent(tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True,handle_parsing_errors=True)
    result= agent.run(input)
    return result

prompt_data = read_json_file('json_data/prompt.json')
environment_data = read_json_file('json_data/environment.json')
agent_data = read_json_file('json_data/agentsettings.json')
library_data = read_json_file('json_data/library.json')

merged_data = [prompt_data, environment_data, agent_data, library_data]
generate_result = gerate_langchain(merged_data)
print(generate_result)
        
ret = generate_result

from cache import DiskCache
cache = DiskCache("cache",True)
content = ret.replace('```', '').replace('python', '').strip()
cache._save_to_disk('gpt-4',content)