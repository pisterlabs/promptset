from langchain import OpenAI, SerpAPIWrapper, LLMChain
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.tools import GooglePlacesTool
import faiss
import os
from langchain.chains import LLMMathChain
from collections import deque
from typing import Dict, List, Optional, Any
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.vectorstores.base import VectorStore
from langchain.chains.base import Chain
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain_experimental.autonomous_agents import BabyAGI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import BeautifulSoupTransformer
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.embeddings import OpenAIEmbeddings

import dotenv

from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore

dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv_file)

# Define your embedding model
embeddings_model = OpenAIEmbeddings()

# Initialize the vectorstore as empty
embedding_size = 1536

index = faiss.IndexFlatL2(embedding_size)

vectorstore = FAISS(embeddings_model.embed_query,
                    index, InMemoryDocstore({}), {})
# Define Memory
memory = ConversationSummaryBufferMemory(
    memory_key="chat_history",
    llm=OpenAI(temperature=0.5, model_name="gpt-3.5-turbo-0613"),
    max_token_limit=40,
    return_messages=True
)

# get travel duration

# Define Tools
todo_prompt = """
    You are a great travel planner. You should consider how many days they would like to visit the country or city they want to visit, how much it will cost, and todo list such as sightseeing, activities, shopping, food, etc.
"""
todo_chain = LLMChain.from_string(
    llm=OpenAI(temperature=0.6, model_name="gpt-3.5-turbo-0613"),
    template=todo_prompt
)
format_prompt = """
    Format the output as JSON with the following keys:
                day
                place_information
                price
                resoning
"""
format_chain = LLMChain.from_string(
    llm=OpenAI(temperature=0.5, model_name="gpt-3.5-turbo-0613"),
    template=format_prompt
)
llm_math_chain = LLMMathChain(llm=OpenAI(temperature=0), verbose=True)
google_place_tool = GooglePlacesTool()
search_tool = SerpAPIWrapper()

tools = [
    Tool(
        name="Search",
        func=search_tool.run,
        description="Useful for searching up-to-date information needed to answer and Search results must be abbreviated.",
    ),
    # Tool(
    #     name="Todo",
    #     func=todo_chain.run,
    #     description="useful for prompting",
    # ),
    Tool(
        name="Map",
        func=google_place_tool.run,
        description="This is useful when search information of places",
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about calculate math"
    ),
    # Tool(
    #     name="Formatter",
    #     func=format_chain.run,
    #     description="Tools to use when finalizing your answer"
    # )
]

llm = OpenAI(temperature=0.6, model_name="gpt-3.5-turbo-0613")

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    input_variables=["chat_history"],
    verbose=True,
    # 중간 과정도 출력하고 싶어서 사용한 속성이지만 오류 발생 ValueError: One output key expected, got dict_keys(['output', 'intermediate_steps'])
    # return_intermediate_steps=True
)

agent("""서울을 2023-09-27 부터 2023-09-30 여행할 동안 가볼만한 장소, 추천 레스토랑을 날짜마다 여행 계획을 작성해서 알려줘
                여행 계획은 다음과 같은 Key 를 가진 JSON 형식으로 작성해줘:
                day
                place_information
                price
                resoning
                """)
