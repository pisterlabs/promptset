import os

import openai
import tiktoken
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY"),


def load_document(file_path: str):
    loader = DirectoryLoader(file_path, glob="*.txt", loader_cls=TextLoader)
    # loader = TextLoader(file_path)
    documents = loader.load()
    return documents


def split_documents(document, chunk_size=10000, chunk_overlap=25):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(document)


def calculate_embedding_token(chunks):
    encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
    model_cost = 0.0001 / 1000
    total_token = 0
    for chunk in chunks:
        total_token = total_token + len(encoding.encode(chunk.page_content))
    print(f"{total_token * model_cost:.7f}")


def new_embedded_document(embedding_model):
    documents = load_document("specialty/")
    chunks = split_documents(documents)
    calculate_embedding_token(chunks)
    return Chroma.from_documents(chunks, embedding_model,
                                 collection_name="poc_specialty_collection",
                                 persist_directory="store/")


openai_lc_client = new_embedded_document(OpenAIEmbeddings(model="text-embedding-ada-002"))
