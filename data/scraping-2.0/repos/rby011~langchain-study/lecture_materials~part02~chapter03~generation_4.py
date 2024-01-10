import os

from dotenv import load_dotenv
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.agents.tools import Tool
from langchain.chat_models import ChatOpenAI
# Agent
from langchain_experimental.plan_and_execute import (
    PlanAndExecute, 
    load_agent_executor,
    load_chat_planner,
)
from langchain.llms import OpenAI

load_dotenv()

# https://serpapi.com/manage-api-key
# pip install google-search-results or poetry add google-search-results
llm = OpenAI(temperature=0)
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
search = SerpAPIWrapper(serpapi_api_key=os.getenv("SERPAPI_API_KEY"))
tools = [  # Agent 가 사용할 수 있느 도구들
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math",
    ),
]

model = ChatOpenAI(temperature=0)  # Agent 는 chatgpt 로 함
planner = load_chat_planner(model)
executor = load_agent_executor(model, tools, verbose=True)
agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

agent.run(
    "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?"
)
