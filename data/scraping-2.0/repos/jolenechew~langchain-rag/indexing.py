# Loading documents from a directory with LangChain

from dotenv import load_dotenv
import os
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader

directory = 'data'
load_dotenv()

# get API keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')


def load_docs(directory):
    loader = PyPDFDirectoryLoader(directory)
    pages = loader.load()
    return pages


documents = load_docs(directory)

# Splitting documents


def split_docs(documents, chunk_size=1000, chunk_overlap=150):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
    docs = text_splitter.split_documents(documents)
    return docs


splits = split_docs(documents)

# Creating embeddings
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


# Storing embeddings in Chroma
persist_directory = './vectordb'

# embed and store vectors in vectordb
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)

# Print the number of vectors stored
print(vectordb._collection.count())


# pinecone.init(
#     api_key="",  # find at app.pinecone.io
#     environment=""  # next to api key in console
# )
# index_name = "langchain-chatbot"
# index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
