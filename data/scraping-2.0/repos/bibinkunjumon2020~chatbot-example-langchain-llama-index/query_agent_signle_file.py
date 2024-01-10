""" Using agent chatbot with subquery engine.
    Marvin Extractor for meta data augmentation in index vectors.



    Used single file - result was bad
"""

# https://gpt-index.readthedocs.io/en/latest/end_to_end_tutorials/chatbots/building_a_chatbot.html
# https://gpt-index.readthedocs.io/en/stable/examples/metadata_extraction/MarvinMetadataExtractorDemo.html


import openai
from index_handler import load_index

import os, logging
from pathlib import Path
from dotenv import load_dotenv

# Set base directory and load environment variables -- IT is must

BASE_DIR = Path(__file__).resolve().parent
dotenv_path = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path)
openai.api_key = os.environ["OPENAI_API_KEY"]

import nest_asyncio

nest_asyncio.apply()


from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index import ServiceContext
import json

from main5_meta_data_extractor import extract_my_metadata  # my module

# Using the LlamaDebugHandler to print the trace of the sub questions
# captured by the SUB_QUESTION callback event type

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])
service_context = ServiceContext.from_defaults(callback_manager=callback_manager)
cwd = os.getcwd()


file_repo_path = os.path.join(cwd, "file_repo") - #i used single file in this
index_repo_path = os.path.join(cwd, "index_repo1")

# documents = SimpleDirectoryReader(input_dir=file_repo_path).load_data()

# # initialize simple vector indices
from llama_index import VectorStoreIndex, ServiceContext, StorageContext

# nodes_doc = extract_my_metadata(file_repo=file_repo_path)

# #We build each index and save it to disk.
service_context = ServiceContext.from_defaults()

storage_context = StorageContext.from_defaults()
# cur_index = VectorStoreIndex(
#     nodes=nodes_doc,
#     service_context=service_context,
#     storage_context=storage_context,
# )
# storage_context.persist(persist_dir=f"{index_repo_path}")


# Load indices from disk

from llama_index import load_index_from_storage
from llama_index import VectorStoreIndex, ServiceContext, StorageContext


storage_context = StorageContext.from_defaults(persist_dir=f"{index_repo_path}")
doc_index = load_index_from_storage(storage_context, service_context=service_context)

# Setting up a Sub Question Query Engine

from llama_index.tools import QueryEngineTool, ToolMetadata

individual_query_engine_tools = [
    QueryEngineTool(
        query_engine=doc_index.as_query_engine(),
        metadata=ToolMetadata(
            name="immigration_service_documents",
            description=f"Information about the Botswana Government Immigration Services",
        ),
    )
]

# create the Sub Question Query Engine,

from llama_index.query_engine import SubQuestionQueryEngine

query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=individual_query_engine_tools,
    service_context=service_context,
)


# Setting up the Chatbot Agent

query_engine_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="immigration_service_documents",
        description=f"Information about the Botswana Government Immigration Services",
    ),
)

# we combine the Tools we defined above into a single list of tools for the agent

tools = individual_query_engine_tools + [query_engine_tool]

from llama_index.agent import OpenAIAgent

agent = OpenAIAgent.from_tools(tools, verbose=True)


while True:
    text_input = input("User: ")
    if text_input == "exit":
        break
    # response = agent.chat(f"System:{system},User:{text_input}")
    response = agent.chat(text_input)
    print(f"Agent: {response}")
