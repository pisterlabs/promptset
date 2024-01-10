from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

# Loader
m1 = PyPDFLoader("./doc/marketing-1.pdf")
m2 = PyPDFLoader("./doc/marketing-2.pdf")

coke1 = PyPDFLoader("./doc/500067-PDF-ENG.pdf")

# Split and load documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
splits = text_splitter.split_documents(m1.load())
splits2 = text_splitter.split_documents(m2.load())

coke = text_splitter.split_documents(coke1.load())

for i in splits2:
    splits.append(i)

# Retrievers
marketing_retriever = Chroma.from_documents(documents=splits,embedding=OpenAIEmbeddings()).as_retriever()
coke_retriever = Chroma.from_documents(documents=coke,embedding=OpenAIEmbeddings()).as_retriever()

retriever_infos = { "marketing": marketing_retriever
                   , "coke": coke_retriever}

# Chain
from langchain.chains.router import MultiRetrievalQAChain
from langchain.llms import OpenAI
from langchain.schema import (
    BaseOutputParser,
    SystemMessage,
    HumanMessage
)

# # Test
# print(chain.run("What is marketing?"))
# print(chain.run("What is marketing strategy?"))
# print(chain.run("What is coca cola marketing strategy?"))
# print(chain.run("What is coca cola marketing strategy?"))
# print(chain.run("What is coca cola marketing strategy?"))
# print(chain.run("What is coca cola marketing strategy?"))
