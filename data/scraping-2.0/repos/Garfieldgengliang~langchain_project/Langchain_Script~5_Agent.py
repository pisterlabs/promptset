
import openai
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # 读取本地 .env 文件，里面定义了 OPENAI_API_KEY

openai.api_base = "https://api.fe8.cn/v1"
openai.api_key = os.getenv('OPENAI_API_KEY')


# 先定义一些工具Tools
from langchain import SerpAPIWrapper
from langchain.tools import Tool, tool

search = SerpAPIWrapper()
tools = [
    Tool.from_function(
        func=search.run,
        name="Search",
        description="useful for when you need to answer questions about current events"
    ),
]

import calendar
import dateutil.parser as parser
from datetime import date


@tool("weekday")
def weekday(date_str: str) -> str:
    """Convert date to weekday name"""
    d = parser.parse(date_str)
    return calendar.day_name[d.weekday()]

tools += [weekday]

# Agent类型一 ReAct
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.agents import AgentType
from langchain.agents import initialize_agent

llm = ChatOpenAI(model_name='gpt-4', temperature=0)

agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.run("周杰伦生日那天是星期几")

# Agent 类型二 通过OpenAI Funtion Call实现
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.agents import AgentType
from langchain.agents import initialize_agent

llm = ChatOpenAI(model_name='gpt-4-0613', temperature=0)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    max_iterations=2,
    early_stopping_method="generate",
)
agent.run("周杰伦生日那天是星期几")

# Agent 类型三 SelfAskWithSearch
from langchain import OpenAI, SerpAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

llm = ChatOpenAI(model_name='gpt-4-0613', temperature=0)
search = SerpAPIWrapper()
tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="useful for when you need to ask with search",
    )
]

self_ask_with_search = initialize_agent(
    tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True
)
self_ask_with_search.run(
    "吴京的老婆的主持过什么节目"
)

# Agent 类型四 Plan-And-Execute
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.agents import load_tools
from langchain import SerpAPIWrapper
from langchain.agents.tools import Tool
from langchain.tools import StructuredTool
from langchain.llms import OpenAI
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pydantic.v1 import BaseModel, Field

class TranslateArguments(BaseModel):
    text: str = Field(default="", description="source text to translate")
    language: str = Field(default="", description="target language to translate into")

llm = ChatOpenAI(model_name='gpt-4', temperature=0)

search = SerpAPIWrapper()

prompt = PromptTemplate(
    input_variables=["text","language"],
    template="{text}\n\nTranslate the above text into {language}. Generate output directly without comments or acknowledgements.",
)

translation_chain = LLMChain(llm=llm, prompt=prompt)

tools = [
    StructuredTool(
        name="Translate",
        func=translation_chain.run,
        description="useful for when you need to translate",
        args_schema=TranslateArguments,
    ),
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    )
]

planner = load_chat_planner(llm)
executor = load_agent_executor(llm, tools, verbose=True)
agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

agent.run("写一份关于开源LLM(large language model)的中文简报")
agent.run("写一份关于开源LLM(large language model)的中文简报")

