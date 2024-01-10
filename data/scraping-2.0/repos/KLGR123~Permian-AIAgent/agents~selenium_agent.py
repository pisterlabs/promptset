#!/usr/bin/env python
# -*- coding: utf-8 -*-

from langchain import LLMChain
from langchain.agents import load_tools, Tool, tool, initialize_agent
from langchain.agents import ZeroShotAgent, AgentExecutor, AgentType, LLMSingleActionAgent, AgentOutputParser
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, BaseChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from typing import List, Union
import pandas as pd
import os, yaml, re

from utils.vecs import KapwingVectorStore
from utils.tools import start_driver, close_driver, open_project
from utils.tools import *


class CustomPromptTemplate(BaseChatPromptTemplate):
    """Set up a prompt template."""
    template: str
    tools: List
    
    def format_messages(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""

        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "

        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])

        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]

class CustomOutputParser(AgentOutputParser):
    """parsing the LLM output into AgentAction and AgentFinish."""

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )

        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)

        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")

        action = match.group(1).strip()
        action_input = match.group(2)

        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

class FastAgent:
    """Agent that under 5 seconds. Only fast agent supports testing now."""

    def __init__(self, model_name="gpt-4", 
                       temperature=0,
                       vectorstore_type="gpt-index",
                       use_chromedriver=True):

        self.SEPERATE_TOKEN = '£'
        self.vectorstore = KapwingVectorStore(model_name=model_name,
                                                temperature=temperature,
                                                filepath='data/functions.txt'
                                            )
        self.vectorstore_type = vectorstore_type
        self.use_chromedriver = use_chromedriver

        with open("config/config.yaml", 'r') as stream:
            config = yaml.safe_load(stream)

        os.environ["OPENAI_API_KEY"] = config['OPENAI_API_KEY']
        self.main_url = config['KAPWING_URL']
        self.project_name = config['KAPWING_PROJECT_NAME']

        if self.use_chromedriver:
            start_driver(self.main_url)
            # open_project(self.project_name)

        if self.vectorstore_type == "faiss":
            self.vectorstore.get_faiss()
            self.vec_query = self.vectorstore.faiss_query
        if self.vectorstore_type == 'gpt-index':
            self.vectorstore.set_gpt_index()
            self.vec_query = self.vectorstore.gpt_index_funcs

        self.exec = False
            
    def run(self, query, url):
        if self.use_chromedriver:
            start_driver(url)

        if query == 'quit':
            if self.use_chromedriver:
                res_url = close_driver()
                print(f"Exiting. Visit url {res_url}.")
            else: print("Exiting.")

        else:  
            start_time = time.time()
            res = self.vec_query(query)
            end_time = time.time()
            latency = end_time - start_time

            res = str(res)
            func_list = [line for line in res.split("\n") if not line.startswith("Answer")]
            func_list = list(filter(bool, func_list))[:3]
        
            print("RECOMMEND FUNCTIONS ARE: \n", func_list)
            index = int(input("PLEASE SELECT YOUR FAVORITE OPTION. YOU SHOULD INPUT 1 / 2 / 3. IF NO SUITABLE OPTION, INPUT 0. \nYOUR CHOICE: ")) - 1

            if index != -1:
                func = func_list[index]
                i = 1
                while "<NULL>" in func:
                    content = input(f"PLACEHOLDER{i} SHOULD BE: ")
                    func = func.replace("<NULL>", content)
                    i += 1
                
                while '""' in func:
                    content = input(f"EMPTY PLACEHOLDER SHOULD BE: ")
                    func = func.replace('""', content)

                try: 
                    if self.use_chromedriver: 
                        exec(func)
                    else: print(func)
                    self.exec = True
                except Exception as e:
                    print("ERROR: ", e)
                    return latency, False, True, index+1, func_list

                is_exec = self.exec
                self.exec = False
                return latency, is_exec, True, index+1, func_list 
            else:
                return latency, False, False, 0, func_list

class FastAgent_Table:
    """Agent that under 5 seconds."""

    def __init__(self, model_name="gpt-4", 
                       temperature=0,
                       vectorstore_type="gpt-index",
                       use_chromedriver=True):

        self.SEPERATE_TOKEN = '£'
        self.vectorstore = KapwingVectorStore(model_name=model_name,
                                                temperature=temperature,
                                                filepath='data/queries.txt')
        self.vectorstore_type = vectorstore_type
        self.use_chromedriver = use_chromedriver

        with open("config/config.yaml", 'r') as stream:
            config = yaml.safe_load(stream)

        os.environ["OPENAI_API_KEY"] = config['OPENAI_API_KEY']
        self.main_url = config['KAPWING_URL']
        self.project_name = config['KAPWING_PROJECT_NAME']

        if self.use_chromedriver:
            start_driver(self.main_url)
            open_project(self.project_name)

        if self.vectorstore_type == "faiss":
            self.vectorstore.get_faiss()
            self.vec_query = self.vectorstore.faiss_query
        if self.vectorstore_type == 'gpt-index':
            self.vectorstore.set_gpt_index()
            self.vec_query = self.vectorstore.gpt_index_query

        self.df = pd.read_excel('data/query_scripts.xlsx')
        self.user_query = self.df['USER_QUERY']
        self.scripts = self.df['SCRIPTS']

        self.exec = False
            
    def run(self, query):
        start_time = time.time()
        res = self.vec_query(query)
        end_time = time.time()
        latency = end_time - start_time

        res = str(res)
        res = eval(res[res.find("["):])
        res_user_query = [res_dict["USER_QUERY"] for res_dict in res]
        res_masks = [res_dict["MASK"] for res_dict in res]

        print("RECOMMEND QUERIES ARE: \n", res_user_query)
        print("MASKS ARE: \n", res_masks)
        index = int(input("PLEASE SELECT YOUR FAVORITE OPTION. YOU SHOULD INPUT 1 / 2 / 3. \nYOUR CHOICE: ")) - 1

        objective_user_query = res_user_query[index]
        objective_masks = res_masks[index]

        if self.df['USER_QUERY'].isin([objective_user_query]).any():
            index_ = self.df['USER_QUERY'].where(self.df['USER_QUERY'] == objective_user_query).dropna().index.tolist()[0]

            if "" in objective_masks:
                masks = re.findall('<(.*?)>', objective_user_query)
                for mask in masks:
                    mask = "<"+mask+">"
                    mask_content = input(f"THE {mask} SHOULD BE:")
                    objective_masks[objective_masks.index("NONE")] = mask_content
            
            script_ = self.df['SCRIPTS'][index_]
            script_masks = re.findall('<(.*?)>', script_)

            for mask, obj_mask in zip(script_masks, objective_masks):
                mask = "<"+mask+">"
                script_ = script_.replace(mask, obj_mask)

            try: 
                if self.use_chromedriver: 
                    exec(script_)
                else: print(script_)
                self.exec = True
            except Exception as e:
                return "Execution Failed. Stop Now."
                
        else:
            return latency, False

        is_exec = self.exec
        self.exec = False
        return latency, is_exec

class RecommendAgent:
    """Agent recommending three queries."""

    def __init__(self, model_name="gpt-4", 
                       temperature=0,
                       vectorstore_type="gpt-index",
                       use_chromedriver=True):

        self.vectorstore = KapwingVectorStore(model_name=model_name,
                                                temperature=temperature,
                                                filepath='data/queries.txt')
        self.vectorstore_type = vectorstore_type
        self.use_chromedriver = use_chromedriver

        if self.vectorstore_type == "faiss":
            self.vectorstore.get_faiss()
            self.vec_query = self.vectorstore.faiss_query
        if self.vectorstore_type == 'gpt-index':
            self.vectorstore.set_gpt_index()
            self.vec_query = self.vectorstore.gpt_index_query
            
    def run(self, query):
        start_time = time.time()
        res = self.vec_query(query)
        end_time = time.time()
        latency = end_time - start_time

        res_list = eval(str(res))
        res_list = [s.strip() for s in res_list]
        index = int(input("PLEASE SELECT YOUR FAVORITE OPTION. YOU SHOULD INPUT 1 / 2 / 3. \nYOUR CHOICE: ")) - 1

        objective = res_list[index]

        if "MASK" in objective:
            masks = re.findall('<(.*?)>', objective)
            for mask in masks:
                mask = "<"+mask+">"
                mask_content = input(f"THE {mask} SHOULD BE:")
                objective = objective.replace(mask, mask_content)
        
        return objective, latency

class ExecuteAgent:
    """Agent executing scripts."""

    def __init__(self, model_name="gpt-4", 
                       temperature=0,
                       vectorstore_type="gpt-index",
                       use_chromedriver=True):

        self.SEPERATE_TOKEN = '£'
        self.vectorstore = KapwingVectorStore(model_name=model_name,
                                                temperature=temperature,
                                                filepath='data/instructions_scripts.txt')
        self.vectorstore_type = vectorstore_type
        self.use_chromedriver = use_chromedriver

        if self.vectorstore_type == "faiss":
            self.vectorstore.get_faiss()
            self.vec_query = self.vectorstore.faiss_query
        if self.vectorstore_type == 'gpt-index':
            self.vectorstore.set_gpt_index()
            self.vec_query = self.vectorstore.gpt_index_scripts_query
        
        self.count = 0
        self.exec = False

    def run(self, query):
        if self.SEPERATE_TOKEN in query:
            content, timestamp = query[:query.index(self.SEPERATE_TOKEN)], query[query.index(self.SEPERATE_TOKEN)+1:]
            query = self.timestamp_query_prompt.format(content=content, timestamp=timestamp)

        if query == 'quit':
            if self.use_chromedriver:
                res_url = close_driver()
                print(f"Exiting. Visit url {res_url}.")
            else: print("Exiting.")
        else:    
            query = self.vec_query(query)
            query = str(query)
            self.scripts = [line for line in query.split("\n") if not line.startswith("```")]
            print('\n')
            for script in self.scripts:
                try: 
                    if self.use_chromedriver: 
                        exec(script)
                    else: print(script)
                    self.exec = True
                except Exception as e:
                    if self.count < 3:
                        self.count += 1
                        return "Retry. The error is {error}".format(error=e)
                    else:
                        self.count = 0
                        return "Execution Failed. Stop Now."

            is_exec = self.exec
            self.exec = False
            return is_exec
        

class SimpleScriptAgent:
    """Simple agent operating video editing online site given user's objective."""

    def __init__(self, model_name="gpt-4", 
                       temperature=0,
                       vectorstore_type="faiss",
                       use_chromedriver=True):

        self.SEPERATE_TOKEN = '£'
        
        with open("config/config.yaml", 'r') as stream:
            config = yaml.safe_load(stream)

        os.environ["OPENAI_API_KEY"] = config['OPENAI_API_KEY']
        self.main_url = config['KAPWING_URL']
        self.project_name = config['KAPWING_PROJECT_NAME']

        self.use_chromedriver = use_chromedriver

        if self.use_chromedriver:
            start_driver(self.main_url)
            open_project(self.project_name)

        with open("config/prompts.yaml", 'r') as stream:
            prompts = yaml.safe_load(stream)

        self.main_agent_template_prompt = prompts['simple_script_main_agent_template_prompt']
        self.executor_description = prompts['simple_script_executor_description']
        self.executor_prompt = prompts['executor_prompt']

        if model_name in ["gpt-4", "gpt-3.5-turbo"]:
            self.llm_ = ChatOpenAI(model_name=model_name, temperature=temperature)
            self.agent_type = AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION
        else:
            self.llm_ = OpenAI(model_name=model_name, temperature=temperature)
            self.agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION

        self.vectorstore = KapwingVectorStore(filepath='data/instructions_scripts.txt')

        self.vectorstore_type = vectorstore_type
        if self.vectorstore_type == "faiss":
            self.vectorstore.get_faiss()
            self.vec_query = self.vectorstore.faiss_scripts_query
        if self.vectorstore_type == 'gpt-index':
            self.vectorstore.set_gpt_index()
            self.vec_query = self.vectorstore.gpt_index_scripts_query

        self.count = 0
        self.exec = False

        def executor_(query):
            query = self.vec_query(query)
            self.scripts = [line for line in query.split("\n") if not line.startswith("```")]

            print('\n')
            for script in self.scripts:
                print(script)
                try: 
                    if self.use_chromedriver: 
                        exec(script)
                    else: print('\n' + script)
                    self.exec = True
                except Exception as e:
                    if self.count < 3:
                        self.count += 1
                        return self.executor_prompt.format(error=e)
                    else:
                        self.count = 0
                        return "Execution Failed. Stop Now."
                    
        self.executor = Tool(name="executor", func=executor_, description=self.executor_description)

        self.tools = [self.executor]
        self.main_prompt = CustomPromptTemplate(template=self.main_agent_template_prompt, 
                                                tools=self.tools, 
                                                input_variables=["input", "intermediate_steps"]
                                            )
        self.output_parser = CustomOutputParser()

        self.llm_chain = LLMChain(llm=self.llm_, prompt=self.main_prompt)
        self.tool_names = [tool.name for tool in self.tools]
        self.agent = LLMSingleActionAgent(llm_chain=self.llm_chain, 
                                          output_parser=self.output_parser,
                                          stop=["\nObservation:"], 
                                          allowed_tools=self.tool_names
                                        )

        self.agent_executor = AgentExecutor.from_agent_and_tools(agent=self.agent, tools=self.tools, verbose=True)

    def get_executor_scripts(self):
        executor_script = "\n".join(self.scripts)
        return executor_script

    def run(self, query):
        if self.SEPERATE_TOKEN in query:
            content, timestamp = query[:query.index(self.SEPERATE_TOKEN)], query[query.index(self.SEPERATE_TOKEN)+1:]
            query = self.timestamp_query_prompt.format(content=content, timestamp=timestamp)

        if query == 'quit':
            if self.use_chromedriver:
                res_url = close_driver()
                print(f"Exiting. Visit url {res_url}.")
            else: print("Exiting.")
        else:    
            self.agent_executor.run(input=query)
            is_exec = self.exec
            self.exec = False
            return is_exec

            
class GPTScriptAgent:
    """Agent operating video editing online site given user's objective."""

    def __init__(self, model_name="gpt-4", 
                       temperature=0,
                       vectorstore_type="faiss",
                       use_chromedriver=True):

        self.SEPERATE_TOKEN = '£'
        
        with open("config/config.yaml", 'r') as stream:
            config = yaml.safe_load(stream)

        os.environ["OPENAI_API_KEY"] = config['OPENAI_API_KEY']
        self.main_url = config['KAPWING_URL']
        self.project_name = config['KAPWING_PROJECT_NAME']

        self.use_chromedriver = use_chromedriver

        if self.use_chromedriver:
            start_driver(self.main_url)
            open_project(self.project_name)

        with open("config/prompts.yaml", 'r') as stream:
            prompts = yaml.safe_load(stream)

        self.main_agent_template_prompt = prompts['script_main_agent_template_prompt']
        self.mask_prompt = prompts['mask_prompt']
        self.steps_generator_prompt = prompts['steps_generator_prompt']
        self.script_generator_prompt = prompts['script_generator_prompt']
        self.executor_description = prompts['script_executor_description']
        self.executor_prompt = prompts['executor_prompt']

        if model_name in ["gpt-4", "gpt-3.5-turbo"]:
            self.llm_ = ChatOpenAI(model_name=model_name, temperature=temperature)
            self.agent_type = AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION
        else:
            self.llm_ = OpenAI(model_name=model_name, temperature=temperature)
            self.agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION

        self.vectorstore_1 = KapwingVectorStore(filepath='data/query_instructions.txt')
        self.vectorstore_2 = KapwingVectorStore(filepath='data/instructions_scripts.txt')

        if vectorstore_type == "faiss":
            _, self.rec_tool_vec_1 = self.vectorstore_1.get_faiss()
            _, self.rec_tool_vec_2 = self.vectorstore_2.get_faiss()
        if vectorstore_type == 'gpt-index':
            self.vectorstore_1.set_gpt_index()
            self.vectorstore_2.set_gpt_index()

        self.count = 0
        self.exec = False

        def executor_(query):
            self.scripts = [line for line in query.split("\n") if not line.startswith("```")]
            for script in self.scripts:
                try: 
                    if self.use_chromedriver: exec(script)
                    else: print('\n' + script)
                    self.exec = True
                except Exception as e:
                    if self.count < 3:
                        self.count += 1
                        return self.executor_prompt.format(error=e)
                    else:
                        self.count = 0
                        return "Execution Failed. Stop Now."
                    
        self.steps_generator = Tool(name="steps_generator", func=self.rec_tool_vec_1.run, description=self.steps_generator_prompt+self.mask_prompt)
        self.script_generator = Tool(name="script_generator", func=self.rec_tool_vec_2.run, description=self.script_generator_prompt+self.mask_prompt)
        self.executor = Tool(name="executor", func=executor_, description=self.executor_description)
        self.human_tool = load_tools(["human"])[0]

        self.tools = [self.steps_generator, self.script_generator, self.executor, self.human_tool]
        self.main_prompt = CustomPromptTemplate(template=self.main_agent_template_prompt, 
                                                tools=self.tools, 
                                                input_variables=["input", "intermediate_steps"]
                                            )
        self.output_parser = CustomOutputParser()

        self.llm_chain = LLMChain(llm=self.llm_, prompt=self.main_prompt)
        self.tool_names = [tool.name for tool in self.tools]
        self.agent = LLMSingleActionAgent(llm_chain=self.llm_chain, 
                                          output_parser=self.output_parser,
                                          stop=["\nObservation:"], 
                                          allowed_tools=self.tool_names
                                        )

        self.agent_executor = AgentExecutor.from_agent_and_tools(agent=self.agent, tools=self.tools, verbose=True)

    def get_executor_scripts(self):
        executor_script = "\n".join(self.scripts)
        return executor_script

    def run(self, query):
        if self.SEPERATE_TOKEN in query:
            content, timestamp = query[:query.index(self.SEPERATE_TOKEN)], query[query.index(self.SEPERATE_TOKEN)+1:]
            query = self.timestamp_query_prompt.format(content=content, timestamp=timestamp)

        if query == 'quit':
            if self.use_chromedriver:
                res_url = close_driver()
                print(f"Exiting. Visit url {res_url}.")
            else: print("Exiting.")
        else:    
            self.agent_executor.run(input=query)
            is_exec = self.exec
            self.exec = False
            return is_exec


class MemoryGPTScriptAgent:
    """Agent operating video editing online site given user's objective with memory buffer."""

    def __init__(self, model_name="gpt-4", 
                       temperature=0,
                       vectorstore_type="faiss",
                       use_chromedriver=True):

        self.SEPERATE_TOKEN = '£'
        
        with open("config/config.yaml", 'r') as stream:
            config = yaml.safe_load(stream)

        os.environ["OPENAI_API_KEY"] = config['OPENAI_API_KEY']
        self.main_url = config['KAPWING_URL']
        self.project_name = config['KAPWING_PROJECT_NAME']

        self.use_chromedriver = use_chromedriver

        if self.use_chromedriver:
            start_driver(self.main_url)
            open_project(self.project_name)

        with open("config/prompts.yaml", 'r') as stream:
            prompts = yaml.safe_load(stream)

        self.main_agent_prefix_prompt = prompts['script_main_agent_prefix_prompt_memory']
        self.main_agent_suffix_prompt = prompts['script_main_agent_suffix_prompt_memory']
        self.mask_prompt = prompts['mask_prompt']
        self.steps_generator_prompt = prompts['steps_generator_prompt']
        self.script_generator_prompt = prompts['script_generator_prompt']
        self.executor_description = prompts['script_executor_description']
        self.executor_prompt = prompts['executor_prompt']

        if model_name in ["gpt-4", "gpt-3.5-turbo"]:
            self.llm_ = ChatOpenAI(model_name=model_name, temperature=temperature)
        else:
            self.llm_ = OpenAI(model_name=model_name, temperature=temperature)

        self.vectorstore_1 = KapwingVectorStore(filepath='data/query_instructions.txt')
        self.vectorstore_2 = KapwingVectorStore(filepath='data/instructions_scripts.txt')

        if vectorstore_type == "faiss":
            _, self.rec_tool_vec_1 = self.vectorstore_1.get_faiss()
            _, self.rec_tool_vec_2 = self.vectorstore_2.get_faiss()
        if vectorstore_type == 'gpt-index':
            _, self.rec_tool_vec_1 = self.vectorstore_1.get_faiss()
            _, self.rec_tool_vec_2 = self.vectorstore_2.get_faiss()

        self.count = 0
        self.exec = False

        def executor_(query):
            self.scripts = [line for line in query.split("\n") if not line.startswith("```")]
            for script in self.scripts:
                try: 
                    if self.use_chromedriver: exec(script)
                    else: print('\n' + script)
                    self.exec = True
                except Exception as e:
                    if self.count < 3:
                        self.count += 1
                        return self.executor_prompt.format(error=e)
                    else:
                        self.count = 0
                        return "Execution Failed. Stop Now."

        self.steps_generator = Tool(name="steps_generator", func=self.rec_tool_vec_1.run, description=self.steps_generator_prompt+self.mask_prompt)
        self.script_generator = Tool(name="script_generator", func=self.rec_tool_vec_2.run, description=self.script_generator_prompt+self.mask_prompt)
        self.executor = Tool(name="executor", func=executor_, description=self.executor_description)
        self.human_tool = load_tools(["human"])[0]

        self.tools = [self.steps_generator, self.script_generator, self.executor, self.human_tool] 
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        self.main_prompt = ZeroShotAgent.create_prompt(self.tools, 
                                                       prefix=self.main_agent_prefix_prompt, 
                                                       suffix=self.main_agent_suffix_prompt, 
                                                       input_variables=["input", "chat_history", "agent_scratchpad"]
                                                    )

        self.llm_chain = LLMChain(llm=self.llm_, prompt=self.main_prompt)
        self.agent = ZeroShotAgent(llm_chain=self.llm_chain, tools=self.tools, verbose=True)
        self.agent_chain = AgentExecutor.from_agent_and_tools(agent=self.agent, tools=self.tools, verbose=True, memory=self.memory)

    def get_executor_scripts(self):
        executor_script = "\n".join(self.scripts)
        return executor_script

    def run(self, query):
        if self.SEPERATE_TOKEN in query:
            content, timestamp = query[:query.index(self.SEPERATE_TOKEN)], query[query.index(self.SEPERATE_TOKEN)+1:]
            query = self.timestamp_query_prompt.format(content=content, timestamp=timestamp)

        if query == 'quit':
            if self.use_chromedriver:
                res_url = close_driver()
                print(f"Exiting. Visit url {res_url}.")
            else: print("Exiting.")
        else:    
            self.agent_executor.run(input=query)
            is_exec = self.exec
            self.exec = False
            return is_exec


def main():
    agent = FastAgent(model_name="gpt-4", temperature=0, use_chromedriver=False)
    # rec_agent = RecommendAgent(model_name="gpt-4", temperature=0)
    # exec_agent = ExecuteAgent(model_name="gpt-4", temperature=0, use_chromedriver=False)
    # agent = SimpleScriptAgent(model_name="gpt-4", temperature=0, use_chromedriver=False)

    while True:
        query = input("QUERY: ")
        if query == 'quit': break
        latency, is_exec, is_top3 = agent.run(query)
        print(latency, is_exec, is_top3)

        if not is_exec: print("TRY AGAIN MORE SPECIFIC.")
        # objective, lantency = rec_agent.run(query)
        # is_done = exec_agent.run(objective)
        # agent.run(objective)
        

if __name__ == "__main__":
    main()