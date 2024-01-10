""" This module contains the Memory class, which is used to retrieve the memory of the bot.
It's a kind of hack from Langchain, as we won't store the memory in a Memory class from Langchain but in a file."""

from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

from careai.utils.configuration import Config

def load_long_term_memory(conf: Config):
    loader = TextLoader('conf/bots/memory/anne_frank.txt')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=conf.open_ai_api_key)

    # TODO create a persistant DB: persist_directory = 'db'
    memory_db = Chroma.from_documents(docs, embeddings)
    retriever = memory_db.as_retriever()
    return retriever
