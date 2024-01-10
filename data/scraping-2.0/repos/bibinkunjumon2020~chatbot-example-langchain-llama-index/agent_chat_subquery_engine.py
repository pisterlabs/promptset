



""" Using agent chatbot with subquery engine."""

# https://gpt-index.readthedocs.io/en/latest/end_to_end_tutorials/chatbots/building_a_chatbot.html


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

# Using the LlamaDebugHandler to print the trace of the sub questions
# captured by the SUB_QUESTION callback event type
# llama_debug = LlamaDebugHandler(print_trace_on_end=True)
# callback_manager = CallbackManager([llama_debug])
# service_context = ServiceContext.from_defaults(callback_manager=callback_manager)
cwd = os.getcwd()


file_repo_path = os.path.join(cwd, "file_repo")
index_repo_path = os.path.join(cwd, "index_repo")
# documents = SimpleDirectoryReader(input_dir=file_repo_path).load_data()

# for document in documents:
#     # taking title from the string using json
#     json_string = document.get_text()
#     json_data = json.loads(json_string)
#     title = str(json_data["title"]).replace(" ", "_")
#     titles.append(title)

titles =["visa_application_employment_visa","visa_application_investment_visa","visa_application_visitors_visa"]

# all_docs = []
# doc_set = {}


# for title in titles:
#     title_docs = SimpleDirectoryReader(input_dir=f"{file_repo_path}/{title}").load_data()
#     for document in title_docs:
#         document.metadata = {"title": title}
#     doc_set[title] = title_docs
#     all_docs.extend(title_docs)

# # # initialize simple vector indices
# from llama_index import VectorStoreIndex, ServiceContext, StorageContext

# index_set = {}
# service_context = ServiceContext.from_defaults(chunk_size=512)

# #We build each index and save it to disk.
    
# for title in titles:
#     storage_context = StorageContext.from_defaults()
#     cur_index = VectorStoreIndex.from_documents(
#         documents=doc_set[title],
#         service_context=service_context,
#         storage_context=storage_context,
#     )
#     index_set[title] = cur_index
#     storage_context.persist(persist_dir=f"{index_repo_path}/{title}")



# Load indices from disk
from llama_index import load_index_from_storage
from llama_index import VectorStoreIndex, ServiceContext, StorageContext
service_context = ServiceContext.from_defaults(chunk_size=512)

index_set = {}
for title in titles:
    storage_context = StorageContext.from_defaults(
        persist_dir=f"{index_repo_path}/{title}"
    )
    cur_index = load_index_from_storage(
        storage_context, service_context=service_context
    )
    index_set[title] = cur_index

#Setting up a Sub Question Query Engine 

from llama_index.tools import QueryEngineTool, ToolMetadata

individual_query_engine_tools = [
    QueryEngineTool(
        query_engine=index_set[title].as_query_engine(),
        metadata=ToolMetadata(
            name=title,
            description=f"useful for when you want to answer queries about the {title} for Botswana Government Service",
        ),
    )
    for title in titles
]

#create the Sub Question Query Engine,

from llama_index.query_engine import SubQuestionQueryEngine

query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=individual_query_engine_tools,
    service_context=service_context,
)

#Setting up the Chatbot Agent

query_engine_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="sub_question_query_engine",
        description="useful for when you want to answer queries that require analyzing multiple service documents for Botswana Government Service",
    ),
)

#we combine the Tools we defined above into a single list of tools for the agent

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








