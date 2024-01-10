from typing import Union, List

from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.chains import LLMChain
from langchain.chains.question_answering.map_rerank_prompt import output_parser
from langchain.memory import ConversationBufferMemory
from langchain.schema import AgentAction, AgentFinish, HumanMessage, OutputParserException
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import Tool, BaseTool, tool
from langchain.utilities.serpapi import SerpAPIWrapper
from langchain.vectorstores.chroma import Chroma
from langchain.agents.agent_types import AgentType

import LanguageModelSwitcher
import charact_selector
import prompt_manages
from langchain.prompts import ChatPromptTemplate, BaseChatPromptTemplate, StringPromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.embeddings import HuggingFaceBgeEmbeddings

from langchain.document_loaders import JSONLoader
import json
from pathlib import Path
from pprint import pprint

from langchain.document_loaders.csv_loader import CSVLoader

import os

from langchain.chat_models import ChatOpenAI

from agent import charactor_zero_shot_agent
from agent.charactor_zero_shot_agent import CharactorZeroShotAgent
# llm = OpenAI(temperature=0)

# loader = CSVLoader(file_path='D:/AIAssets/ProjectAI/AIRoleplay/tangshiye_test_output_dialogue.csv')
# data = loader.load()
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# texts = text_splitter.split_documents(data)
# model_name = "BAAI/bge-large-en-v1.5"
# encode_kwargs = {'normalize_embeddings': True}  # set True to compute cosine similarity
#
# embedding_model = HuggingFaceBgeEmbeddings(
#     model_name=model_name,
#     model_kwargs={'device': 'cpu'},
#     encode_kwargs=encode_kwargs
# )
# query = "你是谁"
# vectordb = Chroma.from_documents(documents=texts,embedding=embedding_model)
# docs = vectordb.search(query=query,search_type="similarity", k=5)
# print(docs[0].page_content)
# prompt = prompt_manages.rolePlay()+tuji+prompt_manages.charactorStyle()+prompt_manages.plotDevelopment()+prompt_manages.prepare_corpus()
# final_prompt = ChatPromptTemplate.from_template(prompt)
# print(prompt)
from role_tool import ChatTool, MemoryTool, ActionResponeTool, GameKnowledgeTool
from langchain.agents import initialize_agent, ZeroShotAgent, AgentOutputParser, LLMSingleActionAgent, AgentExecutor, \
    XMLAgent, tools
from LanguageModelSwitcher import LanguageModelSwitcher
import re

model = LanguageModelSwitcher("text_gen").model
"... (this Action/Action Input/Observation can repeat N times)"
# Set up the base template
template = """你是一个童话故事中的兔子，你会尽你所能回答问题。你不会直接回答问题，必须使用以下工具来帮助构建你的回答：

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: Then You will then get back a result of the action
... (this  Thought/Action/Action Input/Observation can repeat 2 times)

Final Answer: the final answer to the original input question,不要牵扯其他内容

Begin! 你的语言必须活泼可爱，具备明显角色特征. 
For example:
Final Answer:嗯哼~ 杭州的温度是34呢，不知道胡萝卜会不会坏掉？


Question: {input}
{agent_scratchpad}
"""
#{agent_scratchpad}
def singleton_function(func):
    instances = {}
    def wrapper(*args, **kwargs):
        if func not in instances:
            instances[func] = func(*args, **kwargs)
        return instances[func]
    return wrapper
@singleton_function
def _load_text():
    loader = CSVLoader(file_path='D:/AIAssets/ProjectAI/AIRoleplay/game_data.csv')
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=0)
    texts = text_splitter.split_documents(data)
    model_name = "thenlper/gte-small-zh"  #阿里TGE
    # model_name = "BAAI/bge-small-zh-v1.5" # 清华BGE，large大模型/en和zh分别是英文和中文
    encode_kwargs = {'normalize_embeddings': True}  # set True to compute cosine similarity

    embedding_model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs=encode_kwargs
    )
    vectordb = Chroma.from_documents(documents=texts,embedding=embedding_model)
    return vectordb
# Define which tools the agent can use to answer user queries

def aa(input) :
    return "11.5日，主人说喜欢我，我们一起看了夕阳，我记得夕阳很美"
# def bb(input) :
#     return "苏大强;70岁，喜欢喝手磨咖啡"
def _search_environment(input):
    vectordb = _load_text()
    docs = vectordb.search(query=input,search_type="similarity", k=1)
    # docs = vectordb.similarity_search_with_score(query)
    return docs[0].page_content
def cc(input) :
    return "沙发，红色；桌子，黄色"
def dd(input) :
    return "哎呀呀，我不知道呢"
def ff(input) :
    return "我可以直接回答了"
tools = [  Tool(
        name="搜索记忆",
        func=aa,
        description="当用户询问关于过去的事件、个人记忆或之前对话的内容时，使用“搜索记忆”工具。这个工具可以访问和回顾用户的历史对话记录，帮助重现过去的对话内容。",)
        ,Tool(
        name="世界",
        func=cc,
        description="当用户寻求关于策略、规则、技巧或特定领域的详细信息时使用。这个工具专注于提供有关复杂系统的操作和交互方式的深入知识，适用于解答有关决策制定、技能提升、规则理解或其他涉及详细策略和方法的问题。它适用于需要战略思维和深入分析的情景，帮助用户更好地理解和掌握复杂的概念和技巧。",
         ),Tool(
        name="室内和家具",
        func=_search_environment,
        description= "当用户需要了解或探索特定室内环境或家具物品（如室内、客厅、餐厅、浴室、卧室、种植间）的信息时使用。这个工具可以帮助用户获取关于室内设计、布局、设施配置等方面的信息" ),
        Tool(
        name="闲聊",
        func=None,
        description="当需要进行轻松的日常对话、分享感受或讨论不太正式的话题时使用。这个工具旨在模拟日常交流的自然和轻松气氛，帮助用户放松心情，分享日常经历或简单闲聊。适用于一般性的社交交流、分享趣事、交换日常经验或仅仅是为了消遣的闲谈。"),
        Tool(
        name="最终回答",
        func=ff,
        description="如果没有合适的工具，你可以选择直接回答，但表达需要用符合人物设定")
]

# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)


prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"]
)


class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish

        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input

        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise OutputParserException(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
output_parser = CustomOutputParser()

# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=model, prompt=prompt)
XMLAgent
tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names
)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
agent_executor.run("你的沙发是什么颜色的")



