""" Using agent chatbot with subquery engine. main3.py+agent
    Marvin Extractor for meta data augmentation in index vectors.
"""

# https://gpt-index.readthedocs.io/en/latest/end_to_end_tutorials/chatbots/building_a_chatbot.html
# https://gpt-index.readthedocs.io/en/stable/examples/metadata_extraction/MarvinMetadataExtractorDemo.html
#https://gpt-index.readthedocs.io/en/latest/examples/query_engine/sub_question_query_engine.html


import openai
from index_handler import load_index

import os, logging
from pathlib import Path
from dotenv import load_dotenv
from access_title import access_title_docs

# Set base directory and load environment variables -- IT is must

BASE_DIR = Path(__file__).resolve().parent
dotenv_path = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path)
openai.api_key = os.environ["OPENAI_API_KEY"]

import nest_asyncio
import asyncio
nest_asyncio.apply()

from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index import ServiceContext
from llama_index.llms import OpenAI
from llama_index import load_index_from_storage
from llama_index import VectorStoreIndex, ServiceContext, StorageContext
from main5_meta_data_extractor import extract_my_metadata  # my module
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.agent import OpenAIAgent
from llama_index.callbacks.schema import CBEventType, EventPayload

# Using the LlamaDebugHandler to print the trace of the sub questions
# captured by the SUB_QUESTION callback event type

llm = OpenAI(model="gpt-3.5-turbo", temperature=0)

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])
service_context = ServiceContext.from_defaults(
    callback_manager=callback_manager, llm=llm
)


cwd = os.getcwd()


file_repo_path = os.path.join(cwd, "file_repo_dirs")
index_repo_path = os.path.join(cwd, "index_repo_dirs")

titles_of_docs = access_title_docs(root_directory=file_repo_path)

all_docs = []
node_set = {}
#====================== Index generation
# for title in titles_of_docs:
#     title_dir = f"{file_repo_path}/{title}"
#     nodes = asyncio.run(extract_my_metadata(file_repo=title_dir, title=title))
#     node_set[title] = nodes
#     all_docs.extend(nodes)

# print("#"*200,len(all_docs)) #node count in list
# # # initialize simple vector indices

# index_set = {}

# # #We build each index and save it to disk.
# for title in titles_of_docs:
#     storage_context = StorageContext.from_defaults()
#     cur_index = VectorStoreIndex(
#         nodes=node_set[title],
#         service_context=service_context,
#         storage_context=storage_context,
#     )
#     index_set[title] = cur_index
#     storage_context.persist(persist_dir=f"{index_repo_path}/{title}")
#========================================================================

# Load indices from disk


index_set = {}
for title in titles_of_docs:
    storage_context = StorageContext.from_defaults(
        persist_dir=f"{index_repo_path}/{title}"
    )
    cur_index = load_index_from_storage(
        storage_context, service_context=service_context
    )
    index_set[title] = cur_index

# Setting up a Sub Question Query Engine


individual_query_engine_tools = [
    QueryEngineTool(
        query_engine=index_set[title].as_query_engine(),
        metadata=ToolMetadata(
            # name=str(title).replace("_", " "),
            name=title,
            description=f"useful for when you want to answer queries about the {str(title).replace('_',' ')} service of Botswana Government",
        ),
    )
    for title in titles_of_docs
]
from pprint import pprint

pprint(individual_query_engine_tools)

# create the Sub Question Query Engine,


query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=individual_query_engine_tools,
    service_context=service_context,verbose=True,use_async=True
)

# Setting up the Chatbot Agent

query_engine_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="sub_question_query_engine",
        description="useful for when you want to answer queries that require analyzing multiple documents of Botswana Government Services",
    ),
)

# we combine the Tools we defined above into a single list of tools for the agent

tools = individual_query_engine_tools + [query_engine_tool]


agent = OpenAIAgent.from_tools(tools, verbose=True)

# async def print_trace_subquestion(): # Not working
#     # iterate through sub_question items captured in SUB_QUESTION event
#     for i, (start_event, end_event) in enumerate(
#         llama_debug.get_event_pairs(CBEventType.SUB_QUESTION)
#     ):
#         qa_pair = end_event.payload[EventPayload.SUB_QUESTION]
#         print("Sub Question " + str(i) + ": " + qa_pair.sub_q.sub_question.strip())
#         print("Answer: " + qa_pair.answer.strip())
#         print("====================================")


while True:
        text_input = input("User: ")
        if text_input == "exit":
            break
        # response = agent.chat(f"System:{system},User:{text_input}")
        response = agent.chat(text_input)
        print(f"Agent: {response}")
        print("====================================")
        # asyncio.run(print_trace_subquestion()) Not working

********************

from llama_index import SimpleDirectoryReader
from llama_index.indices.service_context import ServiceContext
from llama_index.llms import OpenAI
from llama_index.node_parser import SimpleNodeParser
from llama_index.node_parser.extractors import (
    MetadataExtractor,
    TitleExtractor,
    QuestionsAnsweredExtractor,
    SummaryExtractor,
)
from llama_index.text_splitter import TokenTextSplitter
import os, logging, openai
from pathlib import Path
from dotenv import load_dotenv
from pprint import pprint
from llama_index import set_global_service_context
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
import asyncio
# Set base directory and load environment variables -- IT is must

BASE_DIR = Path(__file__).resolve().parent
dotenv_path = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path)
openai.api_key = os.environ["OPENAI_API_KEY"]


llm = OpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=512)
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])
service_context = ServiceContext.from_defaults(
    callback_manager=callback_manager, llm=llm
)


async def extract_my_metadata(file_repo, title: str):
    # cwd = os.getcwd()

    # file_repo_path = os.path.join(cwd, "file_repo2/visa_application_visitors_visa") # individual folders
    file_repo_path = file_repo
    # index_repo_path = os.path.join(cwd, "index_repo")

    documents = SimpleDirectoryReader(input_dir=file_repo_path).load_data()

    # adding metadata into documents
    for d in documents:
        d.metadata = {"title":title.replace("_", " ")}

    # llm_model = "gpt-3.5-turbo"

    # llm = OpenAI(temperature=0.1, model_name=llm_model, max_tokens=512)
    # service_context = ServiceContext.from_defaults(llm=llm)
    set_global_service_context(service_context)
    # construct text splitter to split texts into chunks for processing
    # this takes a while to process, you can increase processing time by using larger chunk_size
    # file size is a factor too of course
    text_splitter = TokenTextSplitter(separator=" ", chunk_size=512, chunk_overlap=128)

    # set the global service context object, avoiding passing service_context when building the index

    metadata_extractor = MetadataExtractor(
        extractors=[
            TitleExtractor(nodes=5, llm=llm),
            QuestionsAnsweredExtractor(questions=7, llm=llm),
            # EntityExtractor(prediction_threshold=0.5),
            SummaryExtractor(summaries=["prev", "self", "next"], llm=llm),
            # KeywordExtractor(keywords=10, llm=llm),
            # CustomExtractor()
        ],
    )

    # create node parser to parse nodes from document
    node_parser = SimpleNodeParser(
        text_splitter=text_splitter,
        metadata_extractor=metadata_extractor,
    )

    # use node_parser to get nodes from the documents
    nodes = node_parser.get_nodes_from_documents(documents)
    for node in nodes:
        pprint(node.metadata)
    return nodes


# cwd = os.getcwd()

# file_repo_path = os.path.join(cwd, "file_repo2/visa_application_visitors_visa") # individual folders

# extract_my_metadata(file_repo_path)
********************

import os, json

def extract_titles_from_files(root_directory):
    title_list = []
    for root, dirs, files in os.walk(root_directory):
        for file_name in files:
            if file_name.endswith(".json"):  # Filter JSON files
                file_path = os.path.join(root, file_name)
                with open(file_path, "r") as json_file:
                    try:
                        data = json.load(json_file)
                        if "title" in data:
                            title = data["title"]
                            title_list.append(title)
                            # print(f"Title from {file_path}: {title}")
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON in {file_path}")
    return title_list


def access_title_docs(root_directory):
    title_list = extract_titles_from_files(root_directory=root_directory)
    title_no_space_list=[]
    for title in title_list:
        title_no_space = title.replace(" ","_")
        title_no_space_list.append(title_no_space)
    return title_no_space_list


# title_list = []
# extract_titles_from_files("file_repo_dirs")
# print(title_list)
