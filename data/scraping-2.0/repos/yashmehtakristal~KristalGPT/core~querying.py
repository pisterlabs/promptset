#!/usr/bin/env python
# coding: utf-8

# Chosen imports

# from llama_index.retrievers import RecursiveRetriever
# from llama_index.response_synthesizers import get_response_synthesizer
# from llama_index.query_engine import RetrieverQueryEngine
# import pandas as pd
# import os
# import time
# import warnings
# warnings.filterwarnings("ignore")

# All imports
# pdf imports
import fitz
from pprint import pprint
import camelot
import PyPDF2
from PyPDF2 import PdfReader
import streamlit as st
# import pdfplumber

# Langchain imports
from langchain.chains import RetrievalQA
from langchain.chains import create_extraction_chain
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import CSVLoader
from langchain.llms import OpenAI
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.openai_functions.utils import (
    _convert_schema,
    _resolve_schema_references,
    get_llm_kwargs,
)
from langchain.output_parsers.openai_functions import (
    JsonKeyOutputFunctionsParser,
    PydanticAttrOutputFunctionsParser,
)
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema import BasePromptTemplate
from langchain.schema.language_model import BaseLanguageModel


# LlamaIndex imports
from llama_hub.file.pymu_pdf.base import PyMuPDFReader
from llama_index import Document, SummaryIndex
from llama_index import VectorStoreIndex, ServiceContext, LLMPredictor
from llama_index.query_engine import PandasQueryEngine, RetrieverQueryEngine
from llama_index.retrievers import RecursiveRetriever
from llama_index.schema import IndexNode
from llama_index.llms import OpenAI
from llama_hub.file.pymu_pdf.base import PyMuPDFReader
from llama_index.retrievers import RecursiveRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response_synthesizers import get_response_synthesizer


# Other library imports
import pandas as pd
import os
import time
from typing import Any, List, Optional
from pathlib import Path
import pickle


# @st.cache_data(show_spinner = False)
@st.cache_resource(show_spinner = False)
def recursive_retriever(orignal_excel_file, vector_retriever, df_id_query_engine_mapping, service_context, llm_prompts_to_use):
    '''
    recursive_retriever: This function uses recursive retriever in our RetrieverQueryEngine

    Input - 
    orignal_excel_file: Dataframe of the results excel file
    vector_retriever: Top 3 nodes of vector index
    df_id_query_engine_mapping: Mapping of the query engine with each dataframe
    service_context: service_context object defined above

    Output -
    recursive_retriever: Instance of RecursiveRetriever class
    response_synthesizer: Output of get_response_synthesizer
    query_engine: Instance of Retriever Query Engine class
    '''

    recursive_retriever = RecursiveRetriever(
    "vector", 
    retriever_dict={"vector": vector_retriever}, 
    query_engine_dict = df_id_query_engine_mapping, 
    verbose = False,
    )

    response_synthesizer = get_response_synthesizer(
        service_context=service_context, 
        response_mode="no_text"
    )    

    query_engine = RetrieverQueryEngine.from_args(
        recursive_retriever, response_synthesizer = response_synthesizer
    )

    output_response = []
    output_context = []
    count = 1

    for prompt in llm_prompts_to_use:

        # Diagnostic purposes
        st.write(f"{count} time entering loop")

        # Diagnostic purposes - Checking prompt
        st.write(f"Prompt used for this iteration is {prompt}")

        # Diagnostic purposes - Query Engine
        # st.write(type(query_engine))
        # st.write(query_engine)  
        
        # Calling query engine 
        response = query_engine.query(f"{prompt}")

        # Appending to list
        output_context.append(response)
        output_response.append(response.response)
        #output_response.append(str(response))
        
        count += 1

        # Diagnostic purposes - response from LLM
        st.write(f"Response from llm is {response.response}")

        # Diagnostic purposes - context from LLM
        st.write(f"Context from LLM is {response}")

        
        # Wait 8 seconds before executing next prompt
        time.sleep(10)

    return output_response, output_context



# @st.cache_data(show_spinner = False)
# @st.cache_resource(show_spinner = False)
def recursive_retriever_old(vector_retriever, df_id_query_engine_mapping, service_context):
    '''
    recursive_retriever: This function uses recursive retriever in our RetrieverQueryEngine

    Input - 
    orignal_excel_file: Dataframe of the results excel file
    vector_retriever: Top 3 nodes of vector index
    df_id_query_engine_mapping: Mapping of the query engine with each dataframe
    service_context: service_context object defined above

    Output -
    recursive_retriever: Instance of RecursiveRetriever class
    response_synthesizer: Output of get_response_synthesizer
    query_engine: Instance of Retriever Query Engine class
    '''

    recursive_retriever = RecursiveRetriever(
    "vector", 
    retriever_dict={"vector": vector_retriever}, 
    query_engine_dict = df_id_query_engine_mapping, 
    verbose = True,
    )

    response_synthesizer = get_response_synthesizer(
        service_context=service_context, 
        response_mode="compact"
    )    

    query_engine = RetrieverQueryEngine.from_args(
        recursive_retriever, response_synthesizer = response_synthesizer, verbose = True
    )


    return recursive_retriever, response_synthesizer, query_engine


# @st.cache_data(show_spinner = False)
@st.cache_resource(show_spinner = False)
def recursive_retriever_orignal(orignal_excel_file, vector_retriever, df_id_query_engine_mapping, service_context):
    '''
    recursive_retriever: This function uses recursive retriever in our RetrieverQueryEngine

    Input - 
    orignal_excel_file: Dataframe of the results excel file
    vector_retriever: Top 3 nodes of vector index
    df_id_query_engine_mapping: Mapping of the query engine with each dataframe
    service_context: service_context object defined above

    Output -
    recursive_retriever: Instance of RecursiveRetriever class
    response_synthesizer: Output of get_response_synthesizer
    query_engine: Instance of Retriever Query Engine class
    '''

    recursive_retriever = RecursiveRetriever(
    "vector", 
    retriever_dict={"vector": vector_retriever}, 
    query_engine_dict = df_id_query_engine_mapping, 
    verbose = False,
    )

    response_synthesizer = get_response_synthesizer(
        service_context=service_context, 
        response_mode="no_text"
    )    

    query_engine = RetrieverQueryEngine.from_args(
        recursive_retriever, response_synthesizer = response_synthesizer
    )


    return recursive_retriever, response_synthesizer, query_engine

