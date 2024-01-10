from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain import OpenAI
from langchain.tools import StructuredTool
from v1.tools_math import OperacionesMatematicas
from tools.tools_search import BusquedaEnBd
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.agents.react.base import DocstoreExplorer
from langchain.agents import AgentType
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
import numpy as np
import os
import openai
from dotenv import load_dotenv
load_dotenv()

# imports tools

# configurar su api key
openai.api_key = os.environ.get("OPENAI_API_KEY", "")


# configurar las herramientas
tool = BusquedaEnBd()

# configurar la memoria
memory = ConversationBufferWindowMemory(
    memory_key="chat_history", return_messages=True)

# inicializar el agente
llm = ChatOpenAI(temperature=0.9, model="gpt-3.5-turbo")

agent_chain = initialize_agent(
    [tool], llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, memory=memory)


while True:
    # input
    query = input(">>> ")

    # run
    response = agent_chain.run(query)

    # output
    print(response)