"""
This script creates a persistable vector database using Chroma and langchain
"""

import re
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader
import sys
import csv
csv.field_size_limit(sys.maxsize) # Error in CSVLoader otherwise.

data_path = "/home/msaad/workspace/honors-thesis/data_collection/data/full_cleaned_data.csv"
vectordb_persist_dir = "/home/msaad/workspace/honors-thesis/data_collection/data/noncategorized_chroma"


# Load in data
data = CSVLoader(
    file_path=data_path, 
    source_column='url'
).load()

# Chunk the data
texts = RecursiveCharacterTextSplitter(
    chunk_size=300, 
    chunk_overlap=50
).split_documents(data)

# Standardizing the content for each chunk
def standardize_string(input_string):
    standardized_string = input_string.replace('\n', ' ')
    standardized_string = re.sub(r'\s+', ' ', standardized_string)
    standardized_string = re.sub(r'^[^a-zA-Z0-9]+', '', standardized_string)

    return standardized_string.strip()

# Create new 'texts' with some additional filters
texts_cleaned = []

# Iterate over texts page_content category with this cleaning method.
for id in range(len(texts)):
    texts[id].page_content = standardize_string(texts[id].page_content)

    if len(texts[id].page_content) > 100:
        texts_cleaned.append(texts[id])

texts_cleaned

embedding_function = HuggingFaceBgeEmbeddings(
    model_name = "BAAI/bge-small-en",
    model_kwargs = {'device': 'cuda'},
    encode_kwargs = {'normalize_embeddings': True}
)

vectordb = Chroma.from_documents(
    documents = texts_cleaned,
    embedding = embedding_function,
    persist_directory = vectordb_persist_dir
)

vectordb.persist()