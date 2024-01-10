from langchain.embeddings import CohereEmbeddings
from faiss import write_index, read_index
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
import faiss
from langchain.document_loaders import ApifyDatasetLoader
import os
from langchain.document_loaders.base import Document
import numpy
from langchain.vectorstores import FAISS
from decouple import config

os.environ["COHERE_API_KEY"] = config("COHERE_API_KEY") 
database_id = config("DATABASE_ID")

embeddings = CohereEmbeddings()

batch_size = 400
embedding_model = "cl100k_base"
tokenizer = tiktoken.get_encoding('cl100k_base')

loader = ApifyDatasetLoader(
    dataset_id=database_id,
    dataset_mapping_function=lambda dataset_item: Document(
        page_content=dataset_item["text"], metadata={"source": dataset_item["url"]}
    ),
    )

def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)


data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""])

chunks = text_splitter.split_documents(data) 

str_list = []

for i in range(len(chunks)):
    str_list.append(chunks[i].page_content) 

db = FAISS.from_texts(str_list, embeddings)
db.save_local("cohere_index")