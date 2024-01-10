from llama_index import SimpleDirectoryReader, VectorStoreIndex, load_index_from_storage
from llama_index.storage.storage_context import StorageContext
from llama_index.indices.service_context import ServiceContext
from llama_index.llms import OpenAI
from llama_index.node_parser import SimpleNodeParser
from llama_index.node_parser.extractors import (
    MetadataExtractor,
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
)
from llama_index.text_splitter import TokenTextSplitter
from dotenv import load_dotenv
import openai
import gradio as gr
import sys, os
import logging
import json

#loads dotenv lib to retrieve API keys from .env file
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# enable INFO level logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

#define LLM service
llm = OpenAI(temperature=0.1, model_name="gpt-3.5-turbo", max_tokens=512)
service_context = ServiceContext.from_defaults(llm=llm)

#construct text splitter to split texts into chunks for processing
text_splitter = TokenTextSplitter(separator=" ", chunk_size=512, chunk_overlap=128)

#set the global service context object, avoiding passing service_context when building the index 
from llama_index import set_global_service_context
set_global_service_context(service_context)

#create metadata extractor
metadata_extractor = MetadataExtractor(
    extractors=[
        TitleExtractor(nodes=1, llm=llm),
        QuestionsAnsweredExtractor(questions=3, llm=llm),
        SummaryExtractor(summaries=["prev", "self"], llm=llm),
        KeywordExtractor(keywords=10, llm=llm)
    ],
)

#create node parser to parse nodes from document
node_parser = SimpleNodeParser(
    text_splitter=text_splitter,
    metadata_extractor=metadata_extractor,
)

#loading documents
documents_2022 = SimpleDirectoryReader(input_files=["data/executive-summary-2022.pdf"], filename_as_id=True).load_data()
print(f"loaded documents_2022 with {len(documents_2022)} pages")
documents_2021 = SimpleDirectoryReader(input_files=["data/executive-summary-2021.pdf"], filename_as_id=True).load_data()
print(f"loaded documents_2021 with {len(documents_2021)} pages")

def load_index():
     
    try:
        #load storage context
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        #try to load the index from storage
        index = load_index_from_storage(storage_context)
        logging.info("Index loaded from storage.")
        
    except FileNotFoundError:
        #if index not found, create a new one
        logging.info("Index not found. Creating a new one...")

        nodes_2022 = node_parser.get_nodes_from_documents(documents_2022)
        nodes_2021 = node_parser.get_nodes_from_documents(documents_2021)
        print(f"loaded nodes_2022 with {len(nodes_2022)} nodes")
        print(f"loaded nodes_2021 with {len(nodes_2021)} nodes")

        #print metadata in json format
        for node in nodes_2022:
            metadata_json = json.dumps(node.metadata, indent=4)  # Convert metadata to formatted JSON
            print(metadata_json)

        for node in nodes_2021:
            metadata_json = json.dumps(node.metadata, indent=4)  # Convert metadata to formatted JSON
            print(metadata_json)

        #based on the nodes and service_context, create index
        index = VectorStoreIndex(nodes=nodes_2022 + nodes_2021, service_context=service_context)
        # Persist index to disk
        index.storage_context.persist()
        logging.info("New index created and persisted to storage.")

    return index

def data_querying(input_text):

    # Load index
    index = load_index()

    #queries the index with the input text
    response = index.as_query_engine().query(input_text)
    
    return response.response

iface = gr.Interface(fn=data_querying,
                     inputs=gr.components.Textbox(lines=3, label="Enter your question"),
                     outputs="text",
                     title="Analyzing the U.S. Government's Financial Reports for 2022")

iface.launch(share=False)