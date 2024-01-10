# API Keys
import os
os.environ["SERPAPI_API_KEY"] = "174af8beea028b1054380ab7dfc2b72836768a3de724db74a1f131ddf03c27f8"

# HuggingFace Image Generation
os.environ["IMAGE_PROVIDER"] = "sd"
os.environ["HUGGINGFACE_API_TOKEN"] = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Huggingface LLM Setup
from langchain import HuggingFaceHub
repo_id = "stabilityai/stablelm-tuned-alpha-7b"

llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0, "max_length":1024})

# Langchain Prereqs (BabyAGI)
from collections import deque
from typing import Dict, List, Optional, Any

from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Field
from langchain.chains.base import Chain
#from langchain.experimental import BabyAGI

# Langchain Prereqs (Agent Plugin Retrieval)
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
from langchain.agents.agent_toolkits import NLAToolkit
from langchain.tools.plugin import AIPlugin
import re
import plugnplai

####################################
# Langchain Memory Functionality
####################################

from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
#from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoModel

# Define your embedding model
embedding_model = AutoModel.from_pretrained(repo_id)
embedding_size = embedding_model.config.hidden_size
embeddings_model = HuggingFaceEmbeddings(model=embedding_model)

# Initialize the vectorstore as empty
import faiss

model = AutoModel.from_pretrained(repo_id)
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

####################################
# Langchain Tool Functionality
####################################

from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool

#  Get working plugins - only tested plugins (in progress)
urls = plugnplai.get_plugins(filter = 'working')
AI_PLUGINS = [AIPlugin.from_url(url + "/.well-known/ai-plugin.json") for url in urls]

docs = [
    Document(page_content=plugin.description_for_model, 
             metadata={"plugin_name": plugin.name_for_model}
            )
    for plugin in AI_PLUGINS
]
docs_vectorstore = FAISS.from_documents(docs, embeddings)
toolkits_dict = {plugin.name_for_model: 
                 NLAToolkit.from_llm_and_ai_plugin(llm, plugin) 
                 for plugin in AI_PLUGINS}
                 
retriever = docs_vectorstore.as_retriever()

def get_tools(query):
    # Get documents, which contain the Plugins to use
    docs = retriever.get_relevant_documents(query)
    
    # Get the toolkits, one for each plugin
    tool_kits = [toolkits_dict[d.metadata["plugin_name"]] for d in docs]
    
    # Get the tools: a separate NLAChain for each endpoint
    new_tools = []
    for tk in tool_kits:
        new_tools.extend(tk.nla_tools)
    tools = tools.append(new_tools)
    

search = SerpAPIWrapper()
tools = [
    Tool(
        name = "search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions"
    ),
    Tool(
        name = "plugin_retrieval",
        func = get_tools,
        description="useful for when there are no tools available to do what you want to do. You should ask a question relating to what kind of action you want to perform"
    ),
    WriteFileTool(),
    ReadFileTool(),
]



#from langchain.experimental import AutoGPT
#from langchain.chat_models import ChatOpenAI

####################################
# Main Agent
####################################

agent = AutoGPT.from_llm_and_tools(
    ai_name="Zora",
    ai_role="Assistant",
    tools=tools,
    llm=llm,
    memory=vectorstore.as_retriever()
)
# Set verbose to be true
agent.chain.verbose = True

agent.run(["write a weather report for SF today"])
