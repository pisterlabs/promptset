import os
from PyPDF2 import PdfReader
from langchain.text_splitter import TextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain import OpenAI, VectorDBQA

from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreRouterToolkit,
    VectorStoreInfo,
)

# load the OpenAI API key from the environment
# if OPENAI_API_KEY is not found, throw error
if 'OPENAI_API_KEY' not in os.environ:
    raise Exception("OPENAI_API_KEY not found in environment variables")

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# gets the Large Language Model from the environment
def get_llm():
    return OpenAI(temperature=0.7)

# Converts PDF to langchain.Document class objects
# Splits the pdf into multiple documents
def convert_pdf_to_documents(pdf_file):
    # read the pdf file
    pdf_reader = PdfReader(pdf_file)
    # get all the text of the pdf from every page
    pdf_text = ""
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()
    
    # split the text into multiple documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 850,
        chunk_overlap  = 150
    )
    text_chunks = text_splitter.split_text(pdf_text)
    docs = text_splitter.create_documents(text_chunks)

    # print ("DEBUG: Total number of documents: ", len(docs))
    # print("DEBUG: Average document length: ", sum([len(doc.page_content) for doc in docs]) / len(docs))

    return docs

# Stores documents in a VectorStore
# Uses FAISS to store the documents
def init_vector_store(docs):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# Performs a similarity search on a vector store
def similarity_search(vectorstore, query):
    return vectorstore.similarity_search(query)

# retrieve information from one or more vectorstores
def retrieve_info(query, vectorstore, llm=None):
    llm = get_llm()
    vectorstore_info = VectorStoreInfo(
        name="PDF Store",
        description="A vector store for PDF documents",
        vectorstore=vectorstore,
    )

    toolkit = VectorStoreRouterToolkit(
        vectorstores=[vectorstore_info], 
        llm=llm
    )
    agent_executor = create_vectorstore_agent(llm=llm, toolkit=toolkit, verbose=True)
    return agent_executor.run(query)