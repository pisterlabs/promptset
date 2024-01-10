import os

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone

from store_docs.constants import index_name


# initialize pinecone
def store_docs(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(
        openai_api_key=os.environ['RTD_OPENAI_API_KEY'],
    )

    return Pinecone.from_documents(docs, embeddings, index_name=index_name)
