'''
model = SentenceTransformer('thenlper/gte-large')

Limitation
This model exclusively caters to English texts, and any lengthy texts will be truncated to a maximum of 512 tokens.

Model Name	Model Size (GB)	Dimension	Sequence Length	Average (56)	Clustering (11)	Pair Classification (3)	Reranking (4)	Retrieval (15)	STS (10)	Summarization (1)	Classification (12)
gte-large	0.67	1024	512	63.13	46.84	85.00	59.13	52.22	83.35	31.66	73.33
'''
import warnings
warnings.filterwarnings('ignore')

import os
from dotenv import load_dotenv
load_dotenv()

from langchain.llms import OpenAI
from langchain.llms import HuggingFaceHub

# from langchain.prompts import (PromptTemplate, FewShotPromptTemplate)
# from langchain.memory import ConversationBufferMemory 
# from langchain.chains import (LLMChain, SimpleSequentialChain, SequentialChain)

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
# from typing_extensions import Concatenate
from PyPDF2 import PdfReader
# from langchain.document_loaders import OnlinePDFLoader

from langchain.schema import HumanMessage, SystemMessage, AIMessage

import streamlit as st
import pandas as pd
import time
import json

import requests

# ''' 
# data = {'query': 'What are the symptoms of alzheimer?'}
# response = requests.post("http://localhost:8004/search", params=data)
# data_list = json.loads(response.content.decode())
# print(data_list)

# '''


llm_map = {
    'flan-t5-xxl' : "google/flan-t5-xxl",
    'Mistral-7B-Instruct' : "mistralai/Mistral-7B-Instruct-v0.1",
    "OpenAI" : "openai-gpt",
    "long-t5-tglobal-xl" : "google/long-t5-tglobal-xl",
    "long-t5-tglobal-base" : "google/long-t5-tglobal-base",
    "long-t5-tglobal-large" : "google/long-t5-tglobal-large",
    'bloomz-560m' : "bigscience/bloomz-560m",
    'umt5-small' : "google/umt5-small",
    'mt5-base' : "google/mt5-base",
}
 
embedding_map = {
    'GTE' : "thenlper/gte-large",
    'bge_large': "BAAI/bge-large-en",
    # 'bge_small' : "BAAI/bge-small-en",
    'bge_small' : "BAAI/bge-small-en-v1.5",
    'MiniLM-L6-v2' : "sentence-transformers/all-MiniLM-L6-v2",
    'OpenAI' : 'openai-gpt',
    'jina-embeddings' : "jinaai/jina-embeddings-v2-small-en",
    'mpnet' : "sentence-transformers/all-mpnet-base-v2",
}

document_search_space = None
DOC_SPACE_DIR = '../../dummy_faiss_doc_space'
# DOC_SPACE_DIR = None
CHAT_COUNT = 1

#______________________________________________________________________________________________________________________
#----Conversation memory---------------------------------------------------------------

#_______________________________________________________________________________________
#----set the page config---------------------------------------------------------------
# st.title('MedBuddy')
st.set_page_config(
    page_title= "MedBuddy", 
    layout="centered", 
    initial_sidebar_state="expanded",
    menu_items={
        'Get help': None,
        'About': None,
    })
st.header("Hey, I'm MedBuddy!, your personal medical assistant.")

#_______________________________________________________________________________________
#---Initializatio: set the initial system message -> flowMessage------------------------
if 'flowMessage' not in st.session_state:
    st.session_state['flowMessage'] = [
        SystemMessage(content="Hi, I'm MedBuddy, your personal medical assistant. How can I help you?")
    ]

#_______________________________________________________________________________________
#---- print the conversation history---------------------------------------------------
def print_flow_message():
    for message in st.session_state['flowMessage'][:-1]:
        with st.chat_message(name=message.type):
            st.markdown(message.content)
    last_response = st.session_state['flowMessage'][-1]
    with st.chat_message(name=last_response.type):
        with st.empty():
            full_message = ""
            for chunk in last_response.content.split():
                full_message = full_message + " " +chunk 
                time.sleep(0.2)
                st.markdown(full_message + "|▌")


#_______________________________________________________________________________________
# flowMessage = [ HumanMessage(question) + AlMessage(answer) ]
def get_chat_model_response(chain, question, docs):
    st.session_state['flowMessage'].append(HumanMessage(content=question))
    answer = chain.run(input_documents = docs,question=question)
    # chain
    # answer = llm(st.session_state['flowMessage'])
    st.session_state['flowMessage'].append(AIMessage(content=answer))
    print_flow_message()


# def get_text(doc_path = '/home/dosisiddhesh/LANGCHAIN_EXP/pdfs', uploaded_file = False, chunk_size = 500, chunk_overlap = 100):
#     myPdfReader = None
#     raw_text = ''
    
#     if uploaded_file == False:
#         pdf_files = [os.path.join(doc_path, f) for f in os.listdir(doc_path) if f.endswith('.pdf')]
#         with st.empty():
#             st.info(f"Number of pdf files found : {len(pdf_files)}")
#         for pdf_file in pdf_files:
#             myPdfReader = PdfReader(pdf_file)
#             for page in myPdfReader.pages:
#                 raw_text += page.extract_text()
#     else:
#         myPdfReader = PdfReader(doc_path)
#         for page in myPdfReader.pages:
#             raw_text += page.extract_text()

#     text_splitter = CharacterTextSplitter(
#         separator='\n',
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         length_function = len,
#     )
#     texts = text_splitter.split_text(raw_text)
#     print(texts)
#     print(type(texts))
#     # show the length to the streamlit as log
#     with st.empty():
#         st.info(f"Length of the chunks: {len(texts)}")
#         # time.sleep(5)
#         st.info("Creating the vector store")
#     return texts    
    
def get_text(doc_path = '/home/dosisiddhesh/LANGCHAIN_EXP/pdfs', uploaded_file = False, chunk_size = 500, chunk_overlap = 100):
    myPdfReader = None
    raw_text = ''
    
    if uploaded_file == False:
        pdf_files = [os.path.join(doc_path, f) for f in os.listdir(doc_path) if f.endswith('.pdf')]
        with st.empty():
            st.info(f"Number of pdf files found : {len(pdf_files)}")
    return "."
        # for pdf_file in pdf_files:
        #     myPdfReader = PdfReader(pdf_file)
        #     for page in myPdfReader.pages:
        #         raw_text += page.extract_text()
    # else:
    #     myPdfReader = PdfReader(doc_path)
    #     for page in myPdfReader.pages:
    #         raw_text += page.extract_text()

    # text_splitter = CharacterTextSplitter(
    #     separator='\n',
    #     chunk_size=chunk_size,
    #     chunk_overlap=chunk_overlap,
    #     length_function = len,
    # )
    # texts = text_splitter.split_text(raw_text)
    # print(texts)
    # print(type(texts))
    # show the length to the streamlit as log
    # with st.empty():
    #     st.info(f"Length of the chunks: {len(texts)}")
    #     # time.sleep(5)
    #     st.info("Creating the vector store")


 
def start_engine(button):
    texts = None
    if button == 'Local':
        response = requests.post("http://localhost:8004/isdbexists").content.decode()
        print("isdbexist",response)
        if response == 'false':
            texts = get_text(doc_path="/home/dosisiddhesh/LANGCHAIN_EXP/pdfs", chunk_size=500, chunk_overlap=100)
            # response = requests.post("http://localhost:8004/start", params={'texts': texts})
            response = requests.post("http://localhost:8004/start", params={'texts': texts, 'chunk_size':500, 'chunk_overlap':100})
            print(response.content.decode())
            if response.content.decode() == '"Success"':
                st.info("Retriever created/loaded")
                with st.empty():
                    st.info("Vector store created")
            else:
                st.error("Failed to create the retriever")
                st.info(f"Error: {response.content.decode()}")

    elif button == 'Upload':
        uploaded_file = st.file_uploader("Upload a PDF file")
        if uploaded_file:
            texts = get_text(doc_path = uploaded_file, uploaded_file = True)
            st.info("Implementation issue")
            time.sleep(10)
            restart()
        # response = requests.post("http://localhost:8004/start", params={'texts': texts})
        # print(response.content.decode())
        # if response.content.decode() == '"Success"':
        #     st.info("Retriever created/loaded")
        #     with st.empty():
        #         st.info("Vector store created")
        # else:
        #     st.error("Failed to create the retriever")
        #     st.info(f"Error: {response.content.decode()}")

    if query := st.chat_input("Enter the question"):
        history_qna = []
        for message in st.session_state['flowMessage']:
            chat_message = f"{message.type}: {message.content}"
            history_qna.append(chat_message)
        print(history_qna)

        response = requests.post("http://localhost:8004/search", params={'query': query, 'history_qna': history_qna})
        response_dict = json.loads(response.content.decode())
        docs = response_dict['documents']
        st.session_state['flowMessage'].append(HumanMessage(content=query))
        answer = response_dict['answer']
        # answer = chain.run(input_documents = docs,question=question)
        st.session_state['flowMessage'].append(AIMessage(content=answer))
        print_flow_message() 
        top_results = [ {
                'text' : doc['page_content'],
                'source' : doc['metadata'].get('source', ""),
                'page' : doc['metadata'].get('page', ""),
            } for doc in docs]

        with st.expander(f"Top 5 context for the question: {query}"):
            st.table(pd.DataFrame(top_results))

def start_chat_bot(button, emb_model_button, llm_button):
    global DOC_SPACE_DIR
    if button and emb_model_button and llm_button :

        data = {'llm_name':llm_map[llm_button],
                'embedding_name':embedding_map[emb_model_button],
                'temperature':float(temp),
                'DOC_SPACE_DIR': f'{DOC_SPACE_DIR}_{llm_button}_{emb_model_button}',
                  }
        response = requests.post("http://localhost:8004/setvalue", params=data)
        print(response.content.decode())
        if response.content.decode() == '"Success"' or response.content.decode() == '"Already set"':
            st.info("Connected to the server and models loaded")
        else:
            st.error("Failed to connect to the server")
            st.info(f"Error: {response.content.decode()}")
        
        DOC_SPACE_DIR = f'{DOC_SPACE_DIR}_{llm_button}_{emb_model_button}'
        
        st.write(f'You selected: :green[{button}] :sunglasses:')
        st.write(f'You selected: :green[{emb_model_button}] :thumbsup:')
        st.write(f'You selected: :green[{llm_button}] 	:wink:')
        
        start_engine(button)

def start_chat_bot_server(button, emb_model_button, llm_button):
    if button and emb_model_button and llm_button :
        data = {'llm_name':llm_map[llm_button],
                'embedding_name':embedding_map[emb_model_button],
                'temperature':float(temp),
                'query_temperature':float(temp2),
                  }
        response = requests.post("http://localhost:8004/setvaluehyde", params=data)
        print(response.content.decode())
        if response.content.decode() == '"Success"' or response.content.decode() == '"Already set"':
            st.info("Connected to the server and models loaded")
            # data_list = json.loads(response.content.decode())
            # print(data_list)
            
            global DOC_SPACE_DIR
            DOC_SPACE_DIR = f'{DOC_SPACE_DIR}_{llm_button}_{emb_model_button}'
            
            st.write(f'You selected: :green[{button}] :sunglasses:')
            st.write(f'You selected: :green[{emb_model_button}] :thumbsup:')
            st.write(f'You selected: :green[{llm_button}] 	:wink:')
            
            if query := st.chat_input("Enter the question"):
                history_qna = []
                for message in st.session_state['flowMessage']:
                    chat_message = f"{message.type}: {message.content}"
                    history_qna.append(chat_message)
                print(history_qna)
                response = requests.post("http://localhost:8004/search", params={'query': query, 'history_qna': history_qna})
                response_dict = json.loads(response.content.decode())
                docs = response_dict['documents']
                st.session_state['flowMessage'].append(HumanMessage(content=query))
                answer = response_dict['answer']
                # answer = chain.run(input_documents = docs,question=question)
                st.session_state['flowMessage'].append(AIMessage(content=answer))
                print_flow_message()
                top_results = [ {
                        'text' : doc['page_content'],
                        'source' : doc['metadata'].get('source', ""),
                        'page' : doc['metadata'].get('page', ""),
                    } for doc in docs]

                with st.expander(f"Top 5 context for the question: {query}"):
                    st.table(pd.DataFrame(top_results))
        else:
            st.error("Failed to connect to the server")
            st.info(f"Error: {response.content.decode()}")
    

# restart 

def restart():
    print(st.session_state)
    global CHAT_COUNT
    CHAT_COUNT += 1
    if 'button' in st.session_state:
        st.session_state['button'] = None 
    if 'emb_model_button' in st.session_state:
        st.session_state['emb_model_button'] = None
    if 'llm_button' in st.session_state:
        st.session_state['llm_button'] = None

    if 'flowMessage' in st.session_state:
        st.session_state['flowMessage'] = [
            SystemMessage(content="Hi, I'm MedBuddy, your personal medical assistant. How can I help you?")
        ]
    response = requests.post("http://localhost:8004/restart")
    print(response.content.decode())
    if response.content.decode() == 'Success':
        print("Restarted the server")
    else:
        print("Failed to restart the server")
    print(st.session_state)
# ____________________________________________________________________________________________________________________________
# ############################################################################################################################


# col1, col2, col3 = st.columns(3)

# with col1:
with st.sidebar:
    button = st.selectbox(
        'Select the source of the document space ',
        ('pubmed-pgvector', 'Local', 'Upload', 'DropBox'),
        help = "Local source for doc. space",
        placeholder = "Select data source",
        index = None,
        key = "button",
    )  


    # select the embeddings model
    # with col2:
    emb_model_button = st.selectbox(
        'Select the embeddings model',
        ('GTE', 'bge_large', 'bge_small', 'OpenAI', 'MiniLM-L6-v2', 'jina-embeddings', 'mpnet'),
        placeholder="Select embeddings model",
        index=None,
        key="emb_model_button",
    )


    # select the generation model
    # with col3:
    llm_button = st.selectbox(
        'Select the generation model',
        ('OpenAI', 'flan-t5-xxl', 'Mistral-7B-Instruct', "long-t5-tglobal-xl", "long-t5-tglobal-base", "long-t5-tglobal-large", 'bloomz-560m', 'umt5-small', 'mt5-base' ),
        placeholder="Select generation model", 
        index=None,   
        key="llm_button",
    )

# st slider
temp = st.slider(
    'Select the temperature for hyde', 0.0, 1.0, 0.1, key="temp",
    # on_change=start_chat_bot, args=(button, emb_model_button, llm_button)            
    )

temp2 = st.slider(
    'Select the temperature for query', 0.0, 1.0, 0.3, key="temp2",
    )

st.button("Restart", on_click=restart)
if button and emb_model_button and llm_button :
    if button == 'pubmed-pgvector':
        start_chat_bot_server(button, emb_model_button, llm_button)
    else:
        start_chat_bot(button, emb_model_button, llm_button)




