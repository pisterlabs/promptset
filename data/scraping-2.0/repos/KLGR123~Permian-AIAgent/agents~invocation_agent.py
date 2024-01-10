#!/usr/bin/env python
# -*- coding: utf-8 -*-

from langchain import LLMChain
from langchain.agents import load_tools, Tool, tool, initialize_agent
from langchain.llms import OpenAI, OpenAIChat
from langchain.agents import ZeroShotAgent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool
import os, yaml

from utils.vecs import KapwingVectorStore
from utils.tools import start_driver, close_driver, open_project
from utils.tools import *


class KapwingAgent:
    def __init__(self, model_name="gpt-4", 
                       temperature=0):

        self.SEPERATE_TOKEN = '£'
        
        with open("config/config.yaml", 'r') as stream:
            config = yaml.safe_load(stream)

        os.environ["OPENAI_API_KEY"] = config['OPENAI_API_KEY']
        self.main_url = config['KAPWING_URL']
        self.project_name = config['KAPWING_PROJECT_NAME']
        self.user_data_dir = config['USER_DATA_DIR']

        start_driver(self.main_url, user_data_dir=self.user_data_dir)
        open_project(self.project_name)

        with open("config/prompts.yaml", 'r') as stream:
            prompts = yaml.safe_load(stream)

        self.checker_prompt = prompts['checker_prompt_fewshot']
        self.checker_description = prompts['checker_description']
        self.recommender_description = prompts['recommender_description']
        self.executor_description = prompts['executor_description']
        self.executor_prompt = prompts['executor_prompt']
        self.main_agent_prefix_prompt = prompts['main_agent_prefix_prompt']
        self.main_agent_suffix_prompt = prompts['main_agent_suffix_prompt']
        self.timestamp_query_prompt = prompts['timestamp_query_prompt']
        

        self.llm_ = OpenAIChat(model_name=model_name, temperature=temperature)
        self.vectorstore = KapwingVectorStore('data/tools_document.txt')
        self.check_tool_db, self.rec_tool_vec = self.vectorstore.get_faiss()
        
        def checker_(objective):
            checker_chain = LLMChain(llm=self.llm_, prompt=PromptTemplate(template=self.checker_prompt, input_variables=["context", "objective"]))
            relevant_funcs = self.check_tool_db.similarity_search(objective, k=1)
            inputs = [{"context": func.page_content, "objective": objective} for func in relevant_funcs]
            return checker_chain.apply(inputs)

        def executor_(scripts):
            try:
                exec(scripts)
                return "Execute successfully!"
            except Exception as e: 
                return self.executor_prompt.format(error=e)

        self.checker = Tool(name="checker", func=checker_, description=self.checker_description)
        self.recommender = Tool(name="recommender", func=self.rec_tool_vec.run, description=self.recommender_description)
        self.executor = Tool(name="executor", func=executor_, description=self.executor_description)   
    
        self.tools = [self.checker, self.recommender, self.executor]
        self.main_prompt = ZeroShotAgent.create_prompt(self.tools, prefix=self.main_agent_prefix_prompt, 
                                                       suffix=self.main_agent_suffix_prompt, 
                                                       input_variables=["input", "chat_history", "agent_scratchpad"]
                                                    )

        self.memory = ConversationBufferMemory(memory_key="chat_history")
        self.llm_chain = LLMChain(llm=self.llm_, prompt=self.main_prompt)
        self.agent = ZeroShotAgent(llm_chain=self.llm_chain, tools=self.tools, verbose=True)
        self.agent_chain = AgentExecutor.from_agent_and_tools(agent=self.agent, tools=self.tools, verbose=True, memory=self.memory)
    
    def run(self, query):
        if self.SEPERATE_TOKEN in query:
            content, timestamp = query[:query.index(self.SEPERATE_TOKEN)], query[query.index(self.SEPERATE_TOKEN)+1:]
            query = self.timestamp_query_prompt.format(content=content, timestamp=timestamp)

        if query == 'quit':
            res_url = close_driver()
            print(f"Exiting. Visit url {res_url}.")
        else:    
            self.agent_chain.run(input=query)

def main():
    agent = KapwingAgent(model_name="gpt-3.5-turbo")
    while True:
        query = input("QUERY: ")
        agent.run(query)
        if query == 'quit': break
            
if __name__ == "__main__":
    main()

