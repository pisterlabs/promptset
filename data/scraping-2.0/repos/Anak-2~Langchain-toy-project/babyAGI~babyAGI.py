from langchain import OpenAI, SerpAPIWrapper, LLMChain
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
import faiss
import os
from collections import deque
from typing import Dict, List, Optional, Any

from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Field
from langchain.chains.base import Chain
from langchain_experimental.autonomous_agents import BabyAGI
from langchain.tools import GooglePlacesTool
from langchain.chains import LLMMathChain

import dotenv

from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore

dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv_file)

# Define embedding model
embeddings_model = OpenAIEmbeddings()

# Initialize the vectorstore as empty
embedding_size = 1536

index = faiss.IndexFlatL2(embedding_size)

vectorstore = FAISS(embeddings_model.embed_query,
                    index, InMemoryDocstore({}), {})


# Define Tools
todo_prompt = PromptTemplate.from_template(
    "You are a great travel planner. You should actively ask {objective} from customers how many days they would like to visit the country or city they want to visit, how much it will cost, and activities such as sightseeing, activities, shopping, food, etc."
)
todo_chain = LLMChain(llm=OpenAI(temperature=0.6), prompt=todo_prompt)

llm_math_chain = LLMMathChain(llm=OpenAI(temperature=0), verbose=True)

search = SerpAPIWrapper()


def trip_wrapper(input_text):
    search_results = search.run(f"site:trip.com {input_text}")
    return search_results


def naver_place_wrapper(input_text):
    search_results = search.run(
        f"site:tripadvisor.com {input_text}")
    return search_results


google_place_tool = GooglePlacesTool()

format_prompt = PromptTemplate.from_template("""
    Please organize things to do on each day and provide recommended places with information on Google Maps.
""")
format_chain = LLMChain(
    llm=OpenAI(temperature=0.7),
    prompt=format_prompt
)

tools = [
    Tool(
        name="Search trip.com",
        func=trip_wrapper,
        description="useful for when you need to answer restaurant places",
    ),
    Tool(
        name="Todo",
        func=todo_chain.run,
        description="Organizes information that must be provided to the questioneBr in order to plan a trip",
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="""useful for when you need to answer questions about calculate math. 
                     input: The input format must contain numbers."""
    ),
    Tool(
        name="Map",
        func=google_place_tool.run,
        description="This is useful when you need accurate information about a place.",
    ),
    # Tool(
    #     name="Formatter",
    #     func=format_chain.run,
    #     description="Tools to use when famatting your final answer"
    # )
]


prefix = """You are the agent who creates great travel plans. 
            The following is information about the place the questioner would like to plan a trip to: {objective}. 
            Take into account these previously completed tasks: {context}."""
suffix = """Question: {task}
{agent_scratchpad}"""
prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["objective", "task", "context", "agent_scratchpad"],
)

llm = OpenAI(temperature=0.5)
llm_chain = LLMChain(llm=llm, prompt=prompt)
tool_names = [tool.name for tool in tools]
agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)

# Logging of LLMChains
verbose = True

# If None, will keep on going forever
max_iterations: Optional[int] = 2

baby_agi = BabyAGI.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    task_execution_chain=agent_executor,
    verbose=verbose,
    max_iterations=max_iterations,
)

DESTINATION = input("여행지를 입력하세요: ")
FROM = input("출발 날짜를 입력하세요 (YYYY-MM-DD 형식): ")
TO = input("도착 날짜를 입력하세요 (YYYY-MM-DD 형식): ")
DATE = f"I'm planning to go from {FROM} to {TO}"
OBJECTIVE = f"""Please write an {DESTINATION} travel course. {DATE} and 
            Please provide with 
            accommodation information, tourist information, festival information, restaurant information, travel cost information, distance information, etc. that may be helpful in planning the trip.
"""
baby_agi({"objective": OBJECTIVE})
