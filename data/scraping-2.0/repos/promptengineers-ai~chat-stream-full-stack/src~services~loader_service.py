"""Loader service for loading the vectorstore."""
import pickle

from langchain.callbacks import get_openai_callback
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from src.services.logging_service import logger


def load_vectorstore(path: str):
    """Load the vectorstore."""
    with open(path, 'rb') as file:
        vectorstore = pickle.load(file)
        return vectorstore

def split_docs(
    documents,
    chunk_size: int = 500,
    chunk_overlap: int = 0,
):
    """Split the documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
		chunk_size=chunk_size,
		chunk_overlap=chunk_overlap
	)
    chunks = text_splitter.split_documents(documents)
    logger.info("[loader_service.split_docs] Chunks: %s", str(len(chunks)))
    return chunks

def create_vectorstore(docs):
    """Load the vectorstore."""
    with get_openai_callback() as cb:
        embeddings = OpenAIEmbeddings(max_retries=2)
        logger.info(f'[loader_service.create_vectorstore] Tokens: ' + str(cb.total_tokens))
        return FAISS.from_documents(
            split_docs(docs, 1000, 0),
            embeddings
        )

def get_tools(query, retriever, toolkits_dict):
    """Get documents, which contain the Plugins to use"""
    docs = retriever.get_relevant_documents(query)
    # Get the toolkits, one for each plugin
    tool_kits = [toolkits_dict[d.metadata["plugin_name"]] for d in docs]
    # Get the tools: a separate NLAChain for each endpoint
    tools = []
    for tool in tool_kits:
        tools.extend(tool.nla_tools)
    return tools