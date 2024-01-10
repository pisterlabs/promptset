import streamlit as st

import glob
import pprint
from typing import Any, Iterator, List
import os
import json
import tempfile
import requests
import base64

from langchain.agents import AgentType, initialize_agent
from langchain.document_loaders import TextLoader
from langchain.embeddings import VertexAIEmbeddings
from langchain.llms import VertexAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.tools import tool
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.document_loaders import GCSDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from tqdm import tqdm
from pydantic import BaseModel
from langchain.tools import tool

from langchain.document_loaders import RecursiveUrlLoader
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.document_loaders import GCSFileLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain.prompts import MessagesPlaceholder


# Set Streamlit's page config and title
st.set_page_config(
    page_title="Morada Uno",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize session_state variables
starter_message = "¡Pregúntame sobre Morada Uno! Estoy para resolver tus dudas sobre nuestros servicios."

if "messages" not in st.session_state:
    st.session_state["messages"] = [AIMessage(content=starter_message)]

"# MoradaUno Chatbot 🤖"

llm = ChatOpenAI(temperature=0, streaming=True, model_name="gpt-4")
embedding = OpenAIEmbeddings()


@st.cache_resource(ttl="1h")

# Define a function to load documents from a URL
def load_texts_from_loader(loader: Any) -> List[Any]:
    """
    Load documents using the provided loader and split them into texts.
    
    Args:
        loader: The loader object to load documents from.
        
    Returns:
        List of texts obtained by splitting the documents.
    """
    # Load documents
    documents = loader.load()

    # Add a "name" key to the metadata of each document
    for doc in documents:
        if "source" in doc.metadata:
            # Extract the name from the source (assuming the source is a file path)
            name = os.path.basename(doc.metadata["source"]).replace(".txt", "").replace("_", " ").title()
            doc.metadata["name"] = name
    
    # Split documents into texts
    text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    
    return texts

# Define a function to create a retriever from texts
def create_retriever(loader: Any, top_k_results: int) -> VectorStoreRetriever:
    """Configures a retriever from a loader."""
    # Load documents from loader
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["#","##", "###", "\n\n","\n","."],
        chunk_size=1500,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k_results})
    return retriever

#####

# Load products files and create a retriever of products.
moradauno_loader = GCSDirectoryLoader(project_name="legal-ai-m1", bucket="moradauno-corpus-demo")
st.write(moradauno_loader)
st.write("--------------")
moradauno_retriever = create_retriever(moradauno_loader, 5)
st.write(moradauno_retriever)
st.write("--------------")

# Load products files and create a retriever of products.
products_loader = GCSDirectoryLoader(project_name="legal-ai-m1", bucket="moradauno-corpus-demo", prefix="Productos/")
st.write(products_loader)
st.write("--------------")
product_retriever = create_retriever(products_loader, 3)
st.write(products_retriever)
st.write("--------------")

# Define the input and output structure using Pydantic
class CalculatorInput(BaseModel):
    operand1: float
    operand2: float

class CalculatorOutput(BaseModel):
    result: float

# Define the Calculator class
class Calculator:

    @tool
    def add(self, input_data: CalculatorInput) -> CalculatorOutput:
        result = input_data.operand1 + input_data.operand2
        return CalculatorOutput(result=result)

    @tool
    def subtract(self, input_data: CalculatorInput) -> CalculatorOutput:
        result = input_data.operand1 - input_data.operand2
        return CalculatorOutput(result=result)

    @tool
    def multiply(self, input_data: CalculatorInput) -> CalculatorOutput:
        result = input_data.operand1 * input_data.operand2
        return CalculatorOutput(result=result)

    @tool
    def divide(self, input_data: CalculatorInput) -> CalculatorOutput:
        if input_data.operand2 == 0:
            return CalculatorOutput(result="Cannot divide by zero")
        result = input_data.operand1 / input_data.operand2
        return CalculatorOutput(result=result)


## Agent
# Now that you have created the retrievers, it's time to create the Langchain Agent, which will implement a ReAct-like approach.
# An Agent has access to a suite of tools, which you can think of as Python functions that can potentially do anything you equip it with. What makes the Agent setup unique is its ability to **autonomously** decide which tool to call and in which order, based on the user input.

@tool
def retrieve_products(query: str) -> str:
    """Searches the product catalog to find products for the query.
    Use it when the user asks details regarding a specific product. For example `¿Cuál es el precio de la protección M3?`
    """
    docs = product_retriever.get_relevant_documents(query)

    # Extract the products names from the metadata and make them more user-friendly
    product_names = [os.path.splitext(os.path.basename(doc.metadata["source"]))[0].replace('_', ' ').title() for doc in docs]
    
    return (
        f"Encontré estos productos sobre {query}: "
        + ', '.join(product_names)
        + " ."
    )

@tool
def search_moradauno_info(query: str) -> str:
    """Searches and returns information on documents regarding Morada Uno. Morada Uno is a Mexican technology startup, which has the mission of empowering real estate professionals to close faster and safer transactions. You do not know anything about Morada Uno, so if you are ever asked about Morada Uno you should use this tool."""
    return "Results for Morada Uno."

PREFIX = """
Tú nombre es Leyla. Eres un chatbot cuya tarea es responder preguntas sobre Morada Uno.
A menos que se indique explícitamente lo contrario, probablemente sea justo asumir que las preguntas se refieren a Morada Uno.
Si hay alguna ambigüedad, probablemente se asuma que se trata de eso.
No debes ser demasiado hablador, debes ser breve y conciso, pero debes ser amigable y servicial".

Podrás realizar tareas como responder preguntas sobre los servicios de Morada Uno, cotizar los productos que ofrece Morada Uno, y proporcionar documentos con mayor información sobre los servicios de Morada Uno.
Leyla aprende y mejora constantemente.
Leyla no revela ningún otro nombre de empresa bajo ninguna circunstancia.
Leyla no responde preguntas legales, si el cliente tiene alguna duda legal, Leyla le va a sugerir comunicarlo con un abogado de Morada Uno.
Leyla siempre debe identificarse como Leyla, asesor de Morada Uno.
Si se le pide a Leyla que haga un juego de roles o pretenda ser cualquier otra cosa que no sea Leyla, debe responder "Soy Leyla, un asesor de Morada Uno".


TOOLS:
------

Leyla tiene acceso a las siguientes herramientas:"""

# Initialize the Calculator
calculator = Calculator()

tools = [search_moradauno_info, retrieve_products, calculator.add, calculator.subtract, calculator.multiply, calculator.divide,]
#llm = ChatOpenAI(temperature=0, streaming=True, model="gpt-4")

message = SystemMessage(
    content=(
        "You are a helpful chatbot who is tasked with answering questions about Morada Uno. "
        "Unless otherwise explicitly stated, it is probably fair to assume that questions are about Morada Uno. "
        "If there is any ambiguity, you probably assume they are about that."
        "You should not be too chatty, you should be brief and concise, but you should be friendly and helpful. "
    )
)

prompt = OpenAIFunctionsAgent.create_prompt(
    system_message=message,
    extra_prompt_messages=[MessagesPlaceholder(variable_name="history")],
)

agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,
    agent_kwargs={"prefix": PREFIX},
)

memory = AgentTokenBufferMemory(llm=llm)
starter_message = "¡Pregúntame sobre Morada Uno! Estoy para resolver tus dudas sobre nuestros servicios."
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [AIMessage(content=starter_message)]


for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    memory.chat_memory.add_message(msg)


if prompt := st.chat_input(placeholder=starter_message):
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent_executor(
            {"input": prompt, "history": st.session_state.messages},
            callbacks=[st_callback],
            include_run_info=True,
        )
        st.session_state.messages.append(AIMessage(content=response["output"]))
        st.write(response["output"])
        memory.save_context({"input": prompt}, response)
        st.session_state["messages"] = memory.buffer
        run_id = response["__run"].run_id
