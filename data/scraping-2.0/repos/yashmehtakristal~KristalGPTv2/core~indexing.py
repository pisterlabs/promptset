#!/usr/bin/env python
# coding: utf-8

# All imports
from langchain.chat_models import ChatOpenAI
from llama_index.query_engine import PandasQueryEngine, RetrieverQueryEngine
from llama_index import VectorStoreIndex, ServiceContext, LLMPredictor
from llama_index.schema import IndexNode

import pickle
import pandas as pd
import os
import time
import warnings
import streamlit as st
warnings.filterwarnings("ignore")


# Defining query engine over tables
@st.cache_resource(show_spinner = False)
def query_engine_function(table_dfs):
    '''
    query_engine_function: This function defines the llm, service context object and df_query_engines
    Basic query engine function where we set an integer value for request_timeout, max_tries (parameters fo ChatOpenAI)

    Input - 
    table_dfs: list containing dataframe of various tables

    Output - 
    llm, service context, df_query_engines: The respective defined objects
    '''
    # GPT 4 Model used: "gpt-4-0613"
    # GPT 3.5 Model used: 
    llm = ChatOpenAI(model="gpt-3.5-turbo", request_timeout = 120, max_retries = 6)

    # Create a service context object
    service_context = ServiceContext.from_defaults(llm=llm)

    # Create a query engine for each table in the list of table dataframes
    df_query_engines = [
        PandasQueryEngine(table_df, service_context = service_context)
        for table_df in table_dfs
    ]

    # Returns the llm, service context, and query engine
    return llm, service_context, df_query_engines

@st.cache_resource(show_spinner = False)
def query_engine_function_advanced(table_dfs, model, temperature, request_timeout, max_retries):
    '''
    query_engine: This function defines the llm, service context object and df_query_engines
    Advanced query engine function where we allow for change in parameters like request_timeout, max_tries, temperature (parameters fo ChatOpenAI)

    Input - 
    table_dfs: list containing dataframe of various tables

    Output - 
    llm, service context, df_query_engines: The respective defined objects
    '''
    # GPT 4 Model used: "gpt-4-0613"
    # GPT 3.5 Model used: 
    llm = ChatOpenAI(model = model, request_timeout = request_timeout, max_retries = max_retries, temperature = temperature)

    # Create a service context object
    service_context = ServiceContext.from_defaults(llm=llm)

    # Create a query engine for each table in the list of table dataframes
    df_query_engines = [
        PandasQueryEngine(table_df, service_context = service_context)
        for table_df in table_dfs
    ]

    # Returns the llm, service context, and query engine
    return llm, service_context, df_query_engines


### Build Vector Index
# Cannot cache because query engine cannot be pickled
# @st.cache_resource(show_spinner = False)
def build_vector_index(service_context, df_query_engines, docs, nodes_to_retrieve, storage_context, vector_store, is_chroma_loading):
    '''
    build_vector_index: This function ultimately builds the vector index for each of the documents

    Input - 
    service_context: service_context object defined above
    df_query_engines: Query engine for each table in list of tables dataframe
    docs: A list of documents
    nodes_to_retrieve: Number of nodes to retrieve from vector_retriever

    Output - 
    vector_index: vector_index object created 
    vector_retriever: Top 3 nodes of vector index
    df_id_query_engine_mapping: Mapping of the query engine with each dataframe
    nodes_to_retrieve: Number of nodes we decided to retrieve
    '''

    doc_nodes = []

    for doc in docs:        
        doc_nodes.extend(service_context.node_parser.get_nodes_from_documents(doc))

    summaries = [
        "This node provides information stored in the tables in the PDFs. Information could be anything about the financial product.",
    ]

    df_nodes = [
        IndexNode(text=summary, index_id=f"pandas{idx}")
        for idx, summary in enumerate(summaries)
    ]

    df_id_query_engine_mapping = {
        f"pandas{idx}": df_query_engine
        for idx, df_query_engine in enumerate(df_query_engines)
    }

    # If we are creating a new chroma, use this method of vector_index
    if is_chroma_loading is False:
        vector_index = VectorStoreIndex(
            doc_nodes + df_nodes,
            storage_context = storage_context,
            service_context = service_context
            )
        
    # If we are simply loading chroma, use this method of vector_index
    if is_chroma_loading is True:
        vector_index = VectorStoreIndex.from_vector_store(vector_store)

    
    vector_retriever = vector_index.as_retriever(similarity_top_k = nodes_to_retrieve)


    return vector_index, vector_retriever, df_id_query_engine_mapping, nodes_to_retrieve


