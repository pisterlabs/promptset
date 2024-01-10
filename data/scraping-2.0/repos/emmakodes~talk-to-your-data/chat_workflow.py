import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import os


# @st.cache_resource
def chain_workflow(openai_api_key):
    
    #llm 
    llm_name = "gpt-3.5-turbo"

    # persist_directory
    persist_directory = 'vector_index/'	


    # Load OpenAI embedding model
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    

    # Check if the file exists
    if not os.path.exists("vector_index/chroma.sqlite3"):
        # If it doesn't exist, create it

        # load document
        file = "mydocument/animalsinresearch.pdf"
        loader = PyPDFLoader(file)
        documents = loader.load()

        # split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = text_splitter.split_documents(documents)

        # persist_directory
        persist_directory = 'vector_index/'

        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=persist_directory
        )


        vectordb.persist()
        print("Vectorstore created and saved successfully, The 'chroma.sqlite3' file has been created.")
    else:
        # if vectorstore already exist, just call it
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    
    # Load OpenAI chat model
    llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
    
    # specify a retrieval to retrieve relevant splits or documents
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor,base_retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 3}))

    
    # Create memory 'chat_history' 
    memory = ConversationBufferWindowMemory(k=3,memory_key="chat_history")
    
    # create a chatbot chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=0, openai_api_key=openai_api_key), 
        chain_type="map_reduce", 
        retriever=compression_retriever, 
        memory=memory,
        get_chat_history=lambda h : h,
        verbose=True
    )
    
    
    return qa