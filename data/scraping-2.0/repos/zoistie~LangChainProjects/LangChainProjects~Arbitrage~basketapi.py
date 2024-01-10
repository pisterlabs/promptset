import os
from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)
from langchain.chains import RetrievalQA
from apikey import apikey

#API Stuff
os.environ['OPENAI_API_KEY'] = apikey

#Document Analysis
loader = PyPDFLoader('apibasketball.pdf')
pages = loader.load_and_split()

#LLM
llm = OpenAI(temperature = 0.9)
embeddings = OpenAIEmbeddings()
#Vector Store Stuff
store = Chroma.from_documents(pages, embeddings ,collection_name = "data")

basketapi = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=store.as_retriever()
)