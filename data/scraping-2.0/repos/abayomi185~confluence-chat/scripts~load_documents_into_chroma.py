from dotenv import load_dotenv

load_dotenv()

import argparse
import os

import chromadb
from chromadb.config import Settings
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

COLLECTION = "confluence_collection"
DATA_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../data"

parser = argparse.ArgumentParser()
parser.add_argument("--new", action="store_true", help="Set up ChromaDB")
args = parser.parse_args()

# ChromaDB setup
chroma_client = chromadb.HttpClient(settings=Settings(allow_reset=True))

huggingface_embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = huggingface_embeddings


def load_txt_documents_and_split_into_chunks(data_dir: str):
    loader = DirectoryLoader(data_dir, glob="./*.txt", loader_cls=TextLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=0)
    chunks = text_splitter.split_documents(documents)
    return chunks


def load_chunks_into_vectorstore(chunks):
    # Reset vector store first
    chroma_client.reset()
    chroma_client.create_collection(COLLECTION)
    Chroma.from_documents(
        chunks, embeddings, client=chroma_client, collection_name=COLLECTION
    )


def get_connection_to_chromadb():
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        client=chroma_client,
    )


if __name__ == "__main__":
    # To set up ChromaDB
    if args.new:
        chunks = load_txt_documents_and_split_into_chunks(DATA_DIR)
        load_chunks_into_vectorstore(chunks)

    vectordb = get_connection_to_chromadb()

    retriever = vectordb.as_retriever()
    query = "How to perform a SQL migration?"
    docs = retriever.get_relevant_documents(query)
    print(docs)
