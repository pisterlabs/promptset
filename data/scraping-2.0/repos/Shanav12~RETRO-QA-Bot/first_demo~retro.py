import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
import sys
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template
from lp import (
    get_documents, 
    create_index,
    handle_userinput2
)
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
import shutil
import os
import tempfile
import pypdf
import openai
from llama_index import (
    VectorStoreIndex,
    SimpleKeywordTableIndex,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
)

from llama_index import get_response_synthesizer
from llama_index.query_engine import RetrieverQueryEngine

# import QueryBundle
from llama_index import QueryBundle

# import NodeWithScore
from llama_index.schema import NodeWithScore

# Retrievers
from llama_index.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)

from typing import List
from llama_index.node_parser import SimpleNodeParser

from llama_index import GPTVectorStoreIndex, StorageContext, ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores import PineconeVectorStore
from llama_index import Document
OPEN_AI_KEY =  # insert 
openai.api_key = # insert 
PINECONE_API_KEY =  # insert 
PINECONE_API_ENV =  # insert 

# gets the pdf text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# converts the texts into chunks so it is easier to process via langchain
def get_text_chunks(some_text):
    text_splitter = CharacterTextSplitter(separator = "\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(some_text)
    return chunks

# create the OpenAI embeddings that are needed
def create_embeddings(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key = OPEN_AI_KEY)
    text_chunk_embeddings = embeddings.embed_documents([text for text in text_chunks])
    return text_chunk_embeddings

# instantiate the GPT 3.5 turbo model

# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  
    environment=PINECONE_API_ENV, 
    index_name = 'retro-test'
)

# pinecone store embeddings
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=OPEN_AI_KEY)

    @st.cache_resource
    def load_pinecone_existing_index():
        pass
        docsearch = Pinecone.from_existing_index(index_name='retro-test', embedding=embeddings)
        return docsearch
    docsearch=load_pinecone_existing_index()

    vectorstore = docsearch 
    return vectorstore

# create chain that keeps track of questions & memory
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(openai_api_key=OPEN_AI_KEY)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# have a conversation and handle the user imput
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title = "RETRO")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Add buttons for "langchain & pinecone" and "llama-index & pinecone"
    selected_option = st.selectbox("Select backend:", ["Langchain & Pinecone", "Llama-index & Pinecone"])
    engine = None
    st.header("RETRO")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question and selected_option == "Langchain & Pinecone":
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process", 
            accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                if selected_option == "Langchain & Pinecone":
                    # get pdf text
                    raw_text = get_pdf_text(pdf_docs)

                    # get the text chunks
                    text_chunks = get_text_chunks(raw_text)

                    # create vector store
                    vectorstore = get_vectorstore(text_chunks)

                    # create conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                elif selected_option == "Llama-index & Pinecone":
                    file_names = []
                    if pdf_docs:
                        file_names = [file.name for file in pdf_docs]

                    # Get pdf text
                    #raw_text = get_pdf_text(file_names)

                    documents = get_documents(file_names, pdf_docs)

                    #nodes = node(documents)

                    index, vectorstore = create_index(documents)

                    engine = index.as_query_engine()

    if engine and selected_option == "Llama-index & Pinecone":
        print('hello')
        handle_userinput2(user_question, engine)



if __name__ == '__main__':
    main()
