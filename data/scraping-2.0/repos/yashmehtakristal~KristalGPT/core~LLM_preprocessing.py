#!/usr/bin/env python
# coding: utf-8

# All imports

import fitz
from pprint import pprint
import camelot
import PyPDF2
from PyPDF2 import PdfReader
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
import pandas as pd
import os
import time
import streamlit as st
from typing import Any, List, Optional
from pathlib import Path
import pickle
import openai
from contextlib import redirect_stdout
import io
import warnings
warnings.filterwarnings("ignore")

from tenacity import retry, stop_after_attempt, wait_random_exponential

@st.cache_data(show_spinner = False)
def conditions_excel(orignal_excel_file):
    '''
    conditions_excel: Checking for certain conditions and creating a filtered dataframe

    Input - 
    orignal_excel_file: Dataframe of results excel file

    Output - 
    LLM_inputs: Displays rows of orignal_excel_file, where source type column is equal to LLM
    info_excel_file: Displays rows of orignal_excel_file, where source type column is equal to Discretionary
    '''

    LLM_inputs = orignal_excel_file[orignal_excel_file["Source Type"] == "LLM"]

    Discretionary_inputs = orignal_excel_file[orignal_excel_file["Source Type"] == "Discretionary"]

    return LLM_inputs, Discretionary_inputs


# Function to extract fund variable
@st.cache_data(show_spinner = False)
def extract_fund_variable(info_excel_file):
    '''
    extract_fund_variable: This function extracts the fund variable

    Input - 
    info_excel_file: Dataframe of the info sheet of results excel file

    Output - 
    fund_variable: Fund variable that was extracted from info sheet of results excel file
    '''
    
    for index, row in info_excel_file.iterrows():

        if "Fund variable" in row.values:

            date_index = list(row).index("Fund variable")

            fund_variable = row[date_index + 1]

    # Return fund_variable
    return fund_variable



# Define function to obtain the prompts where we substitute variable name
# This code should ultimately create a new column, "Automatic Processed Input Prompt"
@st.cache_data(show_spinner = False)
def prompts_to_substitute_variable(orignal_excel_file, fund_variable, LLM_inputs):
    '''
    prompts_to_substitute_variable: This function creates a new column, "Automatic Processed Input Prompt" and writes the prompt result there.

    Input - 
    orignal_excel_file: Dataframe of the results excel file
    fund_variable: Fund variable that was extracted from info sheet of results excel file
    LLM_inputs: Displays rows of orignal_excel_file, where source type column is equal to LLM

    Output -
    orignal_excel_file: Dataframe of the results excel file
    llm_full_index: List of index of rows where "Source Type" column is equal to LLM
    '''
    
    variable_replace = orignal_excel_file['Variable replace'] == 'Yes'
    prompt_values = orignal_excel_file.loc[variable_replace, 'Input prompt'].tolist()
    prompt_indices = orignal_excel_file.loc[variable_replace].index.tolist()

    new_prompt_values = []

    for prompt in prompt_values:
        
        modified_prompt = prompt.replace("fund", fund_variable + " fund")
        new_prompt_values.append(modified_prompt)

    orignal_excel_file.loc[prompt_indices, 'Automatic Processed Input prompt'] = new_prompt_values

    llm_full_index = LLM_inputs.index.tolist()

    rest_of_index = [x for x in llm_full_index if x not in prompt_indices]

    orignal_excel_file.loc[rest_of_index, 'Automatic Processed Input prompt'] = orignal_excel_file.loc[rest_of_index, 'Input prompt']

    excel_columns = orignal_excel_file.columns.tolist() 
    excel_columns.remove('Automatic Processed Input prompt') 
    excel_columns.insert(excel_columns.index('Variable replace'), 'Automatic Processed Input prompt') 
    orignal_excel_file = orignal_excel_file[excel_columns] 


    return orignal_excel_file, llm_full_index



@st.cache_data(show_spinner = False)
def storing_input_prompt_in_list(orignal_excel_file, llm_full_index):
    '''
    storing_input_prompt_in_list: This function creates a list of prompts that we pass into our LLM

    Input - 
    orignal_excel_file: Dataframe of the results excel file
    llm_full_index: List of index of rows where "Source Type" column is equal to LLM

    Output -
    orignal_excel_file: Dataframe of the results excel file
    llm_prompts_to_use: The list prompts that we pass into our LLM (filtered for NA values in rows where Source Type = LLM even)
    llm_prompts_index: Index of the prompts that we have passed to our LLM
    '''

    llm_index_len = len(llm_full_index)

    processed_input_prompts = orignal_excel_file["Automatic Processed Input prompt"]

    non_nan_indices = processed_input_prompts.notna()
    non_nan_values = processed_input_prompts[non_nan_indices]
    llm_prompts_index = non_nan_indices[non_nan_indices].index.tolist()

    # These are the processed input prompts in a list format to use as input to our query engine
    llm_prompts_to_use = non_nan_values.tolist()

    # Return the llm_prompts_to_use
    return orignal_excel_file, llm_prompts_to_use, llm_prompts_index


