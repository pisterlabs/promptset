from llama_index import SimpleDirectoryReader, LLMPredictor, ServiceContext, GPTVectorStoreIndex
from llama_index.response.pprint_utils import pprint_response
from langchain.chat_models import ChatOpenAI
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine
from dotenv import load_dotenv
import gradio as gr
import os, sys
import logging

#loads dotenv lib to retrieve API keys from .env file
load_dotenv()

# enable INFO level logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

#define LLM service
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

#set the global service context object, avoiding passing service_context when building the index or when loading index from vector store
from llama_index import set_global_service_context
set_global_service_context(service_context)

def data_ingestion_indexing():
    #load data
    report_2021_docs = SimpleDirectoryReader(input_files=["reports/executive-summary-2021.pdf"]).load_data()
    print(f"loaded executive summary 2021 with {len(report_2021_docs)} pages")

    report_2022_docs = SimpleDirectoryReader(input_files=["reports/executive-summary-2022.pdf"]).load_data()
    print(f"loaded executive summary 2022 with {len(report_2022_docs)} pages")

    #build indices
    report_2021_index = GPTVectorStoreIndex.from_documents(report_2021_docs)
    print(f"built index for executive summary 2021 with {len(report_2021_index.docstore.docs)} nodes")

    report_2022_index = GPTVectorStoreIndex.from_documents(report_2022_docs)
    print(f"built index for executive summary 2022 with {len(report_2022_index.docstore.docs)} nodes")

    #build query engines
    report_2021_engine = report_2021_index.as_query_engine(similarity_top_k=3)
    report_2022_engine = report_2022_index.as_query_engine(similarity_top_k=3)

    #build query engine tools
    query_engine_tools = [
        QueryEngineTool(
            query_engine = report_2021_engine,
            metadata = ToolMetadata(name='executive_summary_2021', description='Provides information on US government financial report executive summary 2021')
        ),
        QueryEngineTool(
            query_engine = report_2022_engine,
            metadata = ToolMetadata(name='executive_summary_2022', description='Provides information on US government financial report executive summary 2022')
        )
    ]

    #define SubQuestionQueryEngine
    sub_question_engine = SubQuestionQueryEngine.from_defaults(query_engine_tools=query_engine_tools)
    
    return sub_question_engine


def data_querying(input_text):

    #queries the engine with the input text    
    response = sub_question_engine.query(input_text)
    return response.response
    
iface = gr.Interface(fn=data_querying,
                     inputs=gr.components.Textbox(lines=3, label="Enter your question"),
                     outputs="text",
                     title="Analyzing the U.S. Government's Financial Reports for 2021 and 2022")

#data ingestion and indexing
sub_question_engine = data_ingestion_indexing()
iface.launch(share=False)
    
#run queries
#response = sub_question_engine.query('Compare and contrast the DoD costs between 2021 and 2022')
#print(response)

#response = sub_question_engine.query('Compare revenue growth from 2021 to 2022')
#print(response)

