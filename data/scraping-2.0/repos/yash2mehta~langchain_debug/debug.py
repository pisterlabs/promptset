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


import openai
## TODO - When trying to replicate this, please use your OpenAI key
openai.api_key = os.environ["OPENAI_API_KEY"]

# initialize PDF reader
reader = PyMuPDFReader()

# specify the directory you want to use
directory = 'Documents'

# Dataframe of the tables from the document we will extract
table_dfs = []

# Initialize docs as an empty list
docs = []

# Iterate over files in that directory
for filename in os.listdir(directory):
    
    # if file has a .pdf extension
    if filename.endswith(".pdf"):
        
        # Construct full file path
        file_path = os.path.join(directory, filename)
        print(f"Processing file: {file_path}")
        
        # Load the file and append the data to docs
        docs.append(reader.load(file_path))

print("Number of documents read:", len(docs))

def get_tables(path: str):

    # Open the file in read binary mode
    with open(path, 'rb') as file:
        
        # Open pdf 
        doc = fitz.open(path)
        
        # Length of pdf
        num_pages = doc.page_count

    # Loop through all pages
    for page in range(0, num_pages):
        
        # Record the start time for this iteration
        start_time = time.time()
                
        # print page number to keep track of which page it is parsing
        print("\n")
        print(f"Current page number evaluation: {page}")
                
        current_page = doc[page]
        
        # Read pdf to extract tables from that specific page
        table_object = current_page.find_tables()
        
        # This uses the tables attribute of object named TableFinder - Gives list of tables
        table_list = table_object.tables
                
        # display number of tables found on certain page 
        print(f"{len(table_list)} total tables (either empty or non-empty) found on page {page}") 
        
        print('***********')
        
        #print(type(table_list))
        #display(table_list)
        #print(table_list)
        
        # non-empty table counter variable
        non_empty_table_counter = 0
        
        # empty table counter variable
        empty_table_counter = 0
        
        # If table_list is empty
        if len(table_list) == 0:
            
            # No tables found on this page
            print(f"No tables found on page {page}")

        else:            
            
            # Iterate through each table in table_list
            for table in table_list:

                # If dataframe (table) is empty
                if table.to_pandas().empty:

                    # Incrementing empty table counter
                    empty_table_counter += 1

                    # Calculate and print elapsed time for this iteration with an empty table
                    elapsed_time_for_empty_table = time.time() - start_time
                    print(f"Time taken for page {page} with empty table #{non_empty_table_counter}: {elapsed_time_for_empty_table} seconds")

                    print('***********')


                # If dataframe (table) is not empty
                else:

                    # Incrementing non-empty table counter
                    non_empty_table_counter += 1

                    # Convert the table to pandas
                    table_df = table.to_pandas()
                    
                    table_df = (
                        # renames the columns of the dataframe based on the first row of the table 
                        # drop the first row & then finally reset index
                        table_df.rename(columns=table_df.iloc[0])
                        .drop(table_df.index[0])
                        .reset_index(drop=True)
                    )

                    # Append to list
                    table_dfs.append(table_df)

                    # Calculate and print elapsed time for this iteration with an empty table
                    elapsed_time_for_table = time.time() - start_time

                    print(f"Time taken for page {page}, table #{non_empty_table_counter}: {elapsed_time_for_table} seconds")
                    print('========')
    
    # return table_dfs dataframe
    return table_dfs


# iterate over files in that directory

# Check filenames in that particular directory
for filename in os.listdir(directory):
    
    # If it is a pdf file
    if filename.endswith(".pdf"):
        
        # Construct full file path
        file_path = os.path.join(directory, filename)
        
        print(f"Processing file: {file_path}")
        
        print('------------')
        print("\n")
        
        # Call get_tables function
        table_dfs = get_tables(file_path)
        
        print('------------')
        print("\n")



for i in range(len(table_dfs)):
    table_dfs[i] = table_dfs[i].replace('\n','', regex=True)

print(len(table_dfs))

for i in range(len(table_dfs)):
    print(table_dfs[i])


# Create a service context object
service_context = ServiceContext.from_defaults(llm=llm)

# Create a query engine for each table in the list of table dataframes
df_query_engines = [
    PandasQueryEngine(table_df, service_context=service_context)
    for table_df in table_dfs
]

# Initialize doc_nodes as an empty list
doc_nodes = []

# Process each document in docs
for doc in docs:
    
    # Call node parser to extract list of nodes from the given document and then add nodes to doc_nodes list
    doc_nodes.extend(service_context.node_parser.get_nodes_from_documents(doc))

print(doc_nodes)


# define index nodes
summaries = [
    "This node provides information stored in the tables in the PDFs. Information could be anything about the financial product.",
]

# For each summary in the summaries list, it creates an IndexNode object with the text of the
# summary and assigns it an index_id that includes "pandas" followed by the index (position) of
# the summary in the summaries list, represented by idx.

df_nodes = [
    IndexNode(text=summary, index_id=f"pandas{idx}")
    for idx, summary in enumerate(summaries)
]

# Below code creates a dictionary called df_id_query_engine_mapping using a dictionary comprehension.
# It is used for mapping index IDs to query engines. For each index ID (which follows the format "pandas0", "pandas1", etc.)
# and its corresponding query engine in the df_query_engines list, it creates a key-value pair in the dictionary.

df_id_query_engine_mapping = {
    f"pandas{idx}": df_query_engine
    for idx, df_query_engine in enumerate(df_query_engines)
}

print(df_id_query_engine_mapping)


# If this dictionary is empty
if not df_id_query_engine_mapping:
    
    empty_table_df = pd.DataFrame()
    df_query_engine = PandasQueryEngine(empty_table_df, service_context=service_context)
    
    # Insert the key-value pair into the dictionary
    df_id_query_engine_mapping["pandas0"] = df_query_engine

# construct top-level vector index + query engine

# Creating a VectorStoreIndex object by concatenating doc_nodes (list of nodes) and df_nodes (IndexNode object)
# vector_index will later be used to perform vector-based similarity searches
vector_index = VectorStoreIndex(doc_nodes + df_nodes)

# Creating a vector_retriever object
# Retriever should return the top 3 most similar results for a given query.
vector_retriever = vector_index.as_retriever(similarity_top_k = 3)

print(vector_retriever)

# Initialize an instance of the RecursiveRetriever class
recursive_retriever = RecursiveRetriever(
    "vector", # Specify retrieval method/strategy to retrieve data 
    retriever_dict={"vector": vector_retriever}, # Defines a mapping where the key "vector" is associated with a retriever object called vector_retriever. 
    query_engine_dict=df_id_query_engine_mapping, # query engine configuration
    verbose = True, # Provide additional output
)

# Create the response synthesizer instance
response_synthesizer = get_response_synthesizer(
    service_context=service_context, # Contains information or context related to a service or application
    response_mode="compact" # Give a compact response
)

# create an instance of the Retriever Query Engine class with specific arguments
query_engine = RetrieverQueryEngine.from_args(
    recursive_retriever, response_synthesizer = response_synthesizer, verbose = True # Associate the above defined response_synthesizer with recursive_retriever
)

# Individual prompting - in case, you don't want to use loop and check for specific value
## TODO - This is where I'm facing an error
fund_name_response = query_engine.query(
    '''
    What is the name of the fund?
    '''
)
