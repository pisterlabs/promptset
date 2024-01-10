
from os import path
import re
import time
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import sys
from tqdm.auto import tqdm
import csv
"""
fills vector database with provided file, filtering by the same criteria as the
test set ids. In this case, extract all news articles which are from the year 2019
"""

DATA_PATH = "data/news-articles-test-set.tsv"

assert path.exists(DATA_PATH), "Missing data file"

def chunk_splitter(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

# load file
# use commas delimiter, double quotes and replace None with empty string
# to not cause parsing problems with the CSVLoader from langchain
CSV_OPTIONS = { "delimiter": "\t", "quotechar": '"'}

csv.field_size_limit(sys.maxsize)
loader = CSVLoader(file_path=DATA_PATH, csv_args=CSV_OPTIONS)
print("Loading documents")
documents = loader.load()

# convert to text snippets
print("Splitting documents")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Embed and store the texts. passing persist_directory will persist the database
print("Embedding and storing documents")
persist_directory = 'db'
embedding = OpenAIEmbeddings(request_timeout=1000)

# cannot add all documents at once, so add in chunks
# initialize db with 100 text snippets
vectordb = Chroma.from_documents(documents=texts[:100], embedding=embedding, persist_directory=persist_directory)
# iterate over all texts in chunks of 100
chunks = [chunk for chunk in chunk_splitter(texts[100:], 100)]
for chunk in tqdm(chunks):
    vectordb = Chroma.from_documents(documents=chunk, embedding=embedding, persist_directory=persist_directory)
    # short timeout so we don't run into timeout errors
    time.sleep(1)