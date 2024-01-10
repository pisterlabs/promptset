import os
import sys

import pinecone
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
import streamlit as st
load_dotenv()

def get_qa_chain():
    pinecone.init(api_key=os.environ['PINECONE_API'], environment='gcp-starter')

    # Load and preprocess the PDF document
    loader = PyPDFLoader('./CV_BOUKLOUHA.pdf')
    documents = loader.load()

    # Split the documents into smaller chunks for processing
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)


    # Use HuggingFace embeddings for transforming text into numerical vectors
    embeddings = HuggingFaceEmbeddings()


    # Set up the Pinecone vector database
    index_name = "index"
    index = pinecone.Index(index_name)
    vectordb = Pinecone.from_documents(texts, embeddings, index_name=index_name)

    llm = OpenAI()
    # Set up the Conversational Retrieval Chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        vectordb.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True
    )
    return qa_chain

qa_chain = get_qa_chain()

chat_history=[]

def old():
    query = input('Prompt: ')
    if query.lower() in ["exit", "quit", "q"]:
        print('Exiting')
        sys.exit()
    result = qa_chain({'question': query, 'chat_history': chat_history})
    print('Answer: ' + result['answer'] + '\n')
    chat_history.append((query, result['answer']))




def send_message_to_llm_api(messages):
    query = messages[-1]['content']

    result = qa_chain({'question': query, 'chat_history': chat_history})
    chat_history.append((query, result['answer']))

    return result['answer']


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


if prompt := st.chat_input():

    message = [{"role": "user", "content":str(prompt)}]
    st.session_state.messages.append(message[0])
    st.chat_message("user").write(prompt)

    response = send_message_to_llm_api(st.session_state.messages)
    response_ = [{"role": "assistant", "content":str(response)}]

    st.session_state.messages.append(response_[0])
    st.chat_message("assistant").write(response)
