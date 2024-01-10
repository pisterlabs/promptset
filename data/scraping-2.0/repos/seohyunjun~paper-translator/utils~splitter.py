# langchain module
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

# clean number
from .preprocess import clean_number
from .pinecone import *
# time module
from tqdm import tqdm

import re


from langchain.schema.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

def splitter(config):
    """splitter paper into chunk size

    Args:
        config (_type_): _description_
    """    
    ## splitter
    if config.pdf :
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size, separators=["\n\n"]
        )
    
    if config.html:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size, separators=["\n\n"]
        )
    
    if config.youtube:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size, separators=["\n"]
        )
    ## splitter
    # text_splitter = CharacterTextSplitter(chunk_size=config.chunk_size, separator="\n\n")
    doc = text_splitter.split_documents(config.paper)
    if config.pdf:
        doc = clean_number(doc)
    
    config.document = doc

    if config.pinecone:
        # embedding document and upload to pinecone

        # emebdding document
        embed(index_name=config.index_name, docs=config.document)