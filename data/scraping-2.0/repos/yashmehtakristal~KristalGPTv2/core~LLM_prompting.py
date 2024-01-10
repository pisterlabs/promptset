#!/usr/bin/env python
# coding: utf-8

# Chosen imports
import streamlit as st
# from core.persist import persist, load_widget_state

# from pages.bulk_upload_advanced import max_retries - This is giving some error

import pickle
import pandas as pd
import os
import time
import warnings
warnings.filterwarnings("ignore")
from tenacity import retry, stop_after_attempt, wait_random_exponential
import openai

# All imports
# pdf imports
import fitz
from pprint import pprint
import camelot
import PyPDF2
from PyPDF2 import PdfReader
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


OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
openai.api_key = OPENAI_API_KEY
openai_api_key = OPENAI_API_KEY

#load_widget_state()


# @st.cache_data(show_spinner = False)
# @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
# @st.cache_resource(show_spinner = False)
def individual_prompt(query_engine, prompt):
    '''
    individual_prompt: This function runs for a single prompt and displays the output result

    Input - 
    query_engine: An instance of the Retriever Query Engine class 
    prompt: The prompt inputted by the user 

    Output -
    final_output: The output of the prompt by the LLM
    '''

    # Query engine prompt
    response = query_engine.query(prompt)

    # The context used for generating output by LLM
    output_context = response

    # The final output from LLM
    # output_response = str(response)
    output_response = response.response

    return output_response, output_context


def individual_prompt_advanced(query_engine, prompt, nodes_to_retrieve, return_all_chunks):
    '''
    individual_prompt_advanced: This function runs for a single prompt
    Additionally, it also factors in the advanced feature of showing all chunks
    # (returning_all_chunks, along with the necessary metadata information like filepath and source)

    Input - 
    query_engine: An instance of the Retriever Query Engine class 
    prompt: The prompt inputted by the user 

    Output -
    final_output: The output of the prompt by the LLM
    '''

    # If user wants to return all chunks
    if return_all_chunks is True:

        individual_context_list = []
        file_path_metadata_list = []
        source_metadata_list = []

        # Query engine prompt
        response = query_engine.query(prompt)

        # The context used for generating output by LLM
        output_context = response

        # The final output from LLM
        # output_response = str(response)
        output_response = response.response

        # Looping through the scores and appending it to a list
        for i in range(nodes_to_retrieve):

            # st.write(response.source_nodes[i].metadata)

            # Appending each individual context in the list
            individual_context = response.source_nodes[i].get_text()
            individual_context_list.append(individual_context)

            # Extracting file_path metadata information & append to list
            if "file_path" in response.source_nodes[i].metadata and response.source_nodes[i].metadata["file_path"] is not None:
                file_path_metadata = response.source_nodes[i].metadata["file_path"]
            else:
                file_path_metadata = ""

            file_path_metadata_list.append(file_path_metadata)

            # Extracting source metadata information & append to list
            if "source" in response.source_nodes[i].metadata and response.source_nodes[i].metadata["source"] is not None:
                source_metadata = response.source_nodes[i].metadata["source"]
            else:
                source_metadata = ""

            source_metadata_list.append(source_metadata)

        return output_response, output_context, individual_context_list, file_path_metadata_list, source_metadata_list
    
    # If user doesn't want to return all chunks
    if return_all_chunks is False:

        context_with_max_score_list = []
        file_path_metadata_list = []
        source_metadata_list = []
        scores = []

        # Query engine prompt
        response = query_engine.query(prompt)

        # The context used for generating output by LLM
        output_context = response

        # The final output from LLM
        # output_response = str(response)
        output_response = response.response

        # Looping through the scores and appending it to a list
        for i in range(nodes_to_retrieve):

            # Append each score to list
            scores.append(response.source_nodes[i].get_score())

        # Finding the maximum score and index at which it was
        max_score = max(scores)
        max_index = scores.index(max_score)

        # Obtain the context which has the corresponding maximum score
        context_with_max_score = response.source_nodes[max_index].get_text()
        context_with_max_score_list.append(context_with_max_score)

        # Extracting file_path metadata information & append to list
        file_path_metadata = response.source_nodes[max_index].metadata["file_path"]
        file_path_metadata_list.append(file_path_metadata)
        
        # Extracting source metadata information
        source_metadata = response.source_nodes[max_index].metadata["source"]
        source_metadata_list.append(source_metadata)

        return output_response, output_context, context_with_max_score_list, file_path_metadata_list, source_metadata_list












        






# @st.cache_data(show_spinner = False)
# @retry(wait = wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
# @st.cache_resource(show_spinner = False)
def prompt_loop(query_engine, llm_prompts_to_use):
    '''
    prompt_loop: This function runs a loop by inputting multiple prompts (from list llm_prompts_to_use) and stores the output 

    Input - 
    query_engine: An instance of the Retriever Query Engine class 
    llm_prompts_to_use: List of input prompts to LLM 

    Output -
    output_response: List containing response of prompts passed to LLM
    output_context: List containing context of the response of prompts passed to LLM
    '''

    output_response = []
    output_context = []
    count = 1

    for prompt in llm_prompts_to_use:

        # Diagnostic purposes
        # st.write(f"{count} time entering loop")

        # Diagnostic purposes - Checking prompt
        # st.write(f"Prompt used for this iteration is {prompt}")

        # Diagnostic purposes - Query Engine
        # st.write(type(query_engine))
        # st.write(query_engine)  
        
        # Calling query engine 
        response = query_engine.query(f"{prompt}")

        # Debugging - Checking if problem is with metadata

        # metadata = response.metadata
        # error_message = metadata.get("error_message")
        # if error_message:
        #     st.write(f"Error message: {error_message}")
        # else:
        #     st.write(f"Response text: {response.response}")

        # Appending to list
        output_context.append(response)
        output_response.append(response.response)
        #output_response.append(str(response))
        
        count += 1

        # Diagnostic purposes - response from LLM
        # st.write(f"Response from llm is {response.response}")

        # Diagnostic purposes - context from LLM
        # st.write(f"Context from LLM is {response}")
        
        # Wait 8 seconds before executing next prompt
        time.sleep(3)

    return output_response, output_context



# @retry(wait = wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
# @st.cache_resource(show_spinner = False)
def prompt_loop_advanced(query_engine, llm_prompts_to_use, nodes_to_retrieve, sleep, return_all_chunks):


    # If we want to return all chunks
    if return_all_chunks is True:

        # variable for keeping track of count
        count = 1

        # These two lists returned will be used in filling up our excel file
        output_response = []
        output_context = []

        # These 3 lists returned will be used in displaying in our UI (will be a list of lists)
        context_with_max_score_list = []
        file_path_metadata_list = []
        source_metadata_list = []

        for prompt in llm_prompts_to_use:

            # st.write(f"Loop #{count}")

            individual_context_list = []
            individual_file_path_metadata_list = []
            individual_source_metadata_list = []
            
            # Calling query engine 
            response = query_engine.query(f"{prompt}")

            # Appending to list
            output_context.append(response)
            output_response.append(response.response)
            #output_response.append(str(response))
            
            count += 1

            # Wait 8 seconds before executing next prompt
            # time.sleep(sleep)

            # Looping through the scores and appending it to a list
            for i in range(nodes_to_retrieve):

                # Appending each individual context in the list
                individual_context = response.source_nodes[i].get_text()
                individual_context_list.append(individual_context)

                # st.write(individual_context)
                # st.write("--")
                # st.write(response.source_nodes[i].metadata)
                # st.write("--")
                # st.write(type(response.source_nodes[i].metadata))
                # st.write("--")
                # st.write(response.source_nodes[i].metadata["file_path"])

                # # Extracting file_path metadata information
                file_path_metadata = response.source_nodes[i].metadata["file_path"]

                # # Split by "\\"
                # split_string = original_string.split("\\")  

                # if len(split_string) == 1:
                    
                #     # Split by "/"
                #     split_string = original_string.split("/")  
                    
                # # Take the last element from list & print it
                # file_path_metadata = split_string[-1]

                # Append file_path_metadata to the list
                individual_file_path_metadata_list.append(file_path_metadata)

                # Extracting source metadata information
                source_metadata = response.source_nodes[i].metadata["source"]
                individual_source_metadata_list.append(source_metadata)

            
            # Now that we have finished iteration over all nodes for a prompt, we update master list.
            # Each variable here will be a list of list, for each prompt.
            context_with_max_score_list.append(individual_context_list)
            file_path_metadata_list.append(individual_file_path_metadata_list)
            source_metadata_list.append(individual_source_metadata_list)

            # sleep for a while before executing next prompt
            time.sleep(sleep)


        return output_response, output_context, context_with_max_score_list, file_path_metadata_list, source_metadata_list


    # If we don't want to return all chunks
    if return_all_chunks is False:

        # variable for keeping track of count
        count = 1

        # These two lists returned will be used in filling up our excel file
        output_response = []
        output_context = []

        # These 3 lists returned will be used in displaying in our UI (for each prompt, will be one value in the list)
        context_with_max_score_list = []
        file_path_metadata_list = []
        source_metadata_list = []


        for prompt in llm_prompts_to_use:

            scores = []
            
            # Calling query engine 
            response = query_engine.query(f"{prompt}")

            # Appending to list
            output_context.append(response)
            output_response.append(response.response)
            #output_response.append(str(response))
            
            count += 1

            # Wait 8 seconds before executing next prompt
            # time.sleep(sleep)


            # Looping through the scores and appending it to a list
            for i in range(nodes_to_retrieve):
                scores.append(response.source_nodes[i].get_score())


            # Finding the maximum score and index at which it was
            max_score = max(scores)
            max_index = scores.index(max_score)


            # Obtain the context which has the corresponding maximum score
            context_with_max_score = response.source_nodes[max_index].get_text()
            context_with_max_score_list.append(context_with_max_score)


            # Extracting file_path metadata information

            original_string = response.source_nodes[max_index].metadata["file_path"]

            # Split by "\\"
            split_string = original_string.split("\\")  

            if len(split_string) == 1:
                
                # Split by "/"
                split_string = original_string.split("/")  
                
            # Take the last element from list & print it
            file_path_metadata = split_string[-1]

            # Append file_path_metadata to the list
            file_path_metadata_list.append(file_path_metadata)

            # Extracting source metadata information
            source_metadata = response.source_nodes[max_index].metadata["source"]
            source_metadata_list.append(source_metadata)
            # print("Page source:", source)

            # sleep for a while before executing next prompt
            time.sleep(sleep)
        

        return output_response, output_context, context_with_max_score_list, file_path_metadata_list, source_metadata_list