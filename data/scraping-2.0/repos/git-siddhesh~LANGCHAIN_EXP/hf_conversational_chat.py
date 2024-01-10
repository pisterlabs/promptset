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

llm_map = {
    'flan-t5-xxl' : "google/flan-t5-xxl",
    'Mistral-7B-Instruct' : "mistralai/Mistral-7B-Instruct-v0.1",
    "openAI" : "openai-gpt",
    "long-t5-tglobal-xl" : "google/long-t5-tglobal-xl",
    "long-t5-tglobal-base" : "google/long-t5-tglobal-base",
    "long-t5-tglobal-large" : "google/long-t5-tglobal-large",
    'bloomz-560m' : "bigscience/bloomz-560m",
    'umt5-small' : "google/umt5-small",
    'mt5-base' : "google/mt5-base",
}

embedding_map = {
    'GTE' : "thenlper/gte-large",
    'MiniLM-L6-v2' : "sentence-transformers/all-MiniLM-L6-v2",
    'openAI' : 'openai-gpt',
    'jina-embeddings' : "jinaai/jina-embeddings-v2-small-en",
    'mpnet' : "sentence-transformers/all-mpnet-base-v2",
}

document_search_space = None
DOC_SPACE_DIR = './faiss_doc_space'
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
                time.sleep(0.5)
                st.markdown(full_message + "▌")


#_______________________________________________________________________________________
# flowMessage = [ HumanMessage(question) + AlMessage(answer) ]
def get_chat_model_response(chain, question, docs):
    st.session_state['flowMessage'].append(HumanMessage(content=question))
    answer = chain.run(input_documents = docs,question=question)
    # chain
    # answer = llm(st.session_state['flowMessage'])
    st.session_state['flowMessage'].append(AIMessage(content=answer))
    print_flow_message()


def train_and_save_doc_space(doc_path = './pdfs', embeddings = None, uploaded_file = False, chunk_size = 500, chunk_overlap = 100):
    myPdfReader = None
    raw_text = ''
    
    if uploaded_file == False:
        pdf_files = [os.path.join(doc_path, f) for f in os.listdir(doc_path) if f.endswith('.pdf')]
        with st.empty():
            st.info(f"Number of pdf files found : {len(pdf_files)}")
            # time.sleep(5)
        for pdf_file in pdf_files:
            myPdfReader = PdfReader(pdf_file)
            for page in myPdfReader.pages:
                raw_text += page.extract_text()
    else:
        myPdfReader = PdfReader(doc_path)
        for page in myPdfReader.pages:
            raw_text += page.extract_text()

    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function = len,
    )
    texts = text_splitter.split_text(raw_text)
    # show the length to the streamlit as log
    with st.empty():
        st.info(f"Length of the chunks: {len(texts)}")
        # time.sleep(5)
        st.info("Creating the vector store")
    document_search_space = FAISS.from_texts(texts, embeddings)
    # print("Vector store created")
    with st.empty():
        st.info("Vector store created")
        # time.sleep(5)
    document_search_space.save_local(DOC_SPACE_DIR)


def load_doc_space_and_chat(chain, embeddings, name = DOC_SPACE_DIR):
    try:
        global document_search_space
        if document_search_space is None:
            document_search_space = FAISS.load_local(name, embeddings)
    except:
        st.info("Document search space not found")
        st.error("Document search space not found")   
    
    if query := st.chat_input("Enter the question"):
        docs = document_search_space.similarity_search(query,k = 5)
        get_chat_model_response(chain, query, docs)
        top_results = [ {
                'text' : doc.page_content,
                'source' : doc.metadata.get('source', ""),
                'page' : doc.metadata.get('page', ""),
            } for doc in docs]

        with st.expander(f"Top 5 context for the question: {query}"):
            st.table(pd.DataFrame(top_results))


def start_engine(button, embedding, chain):
    if button == 'Local':
        if not os.path.exists(DOC_SPACE_DIR):
            train_and_save_doc_space(doc_path="./pdfs/", embeddings=embedding, chunk_size=500, chunk_overlap=100)
        load_doc_space_and_chat( embeddings=embedding, chain=chain)
    elif button == 'Dropbox':
        pass
    elif button == 'Upload':
        uploaded_file = st.file_uploader("Upload a PDF file")
        if uploaded_file:
            train_and_save_doc_space(doc_path = uploaded_file, embeddings = embedding, uploaded_file = True)
            load_doc_space_and_chat(embeddings=embedding, chain=chain)


def start_chat_bot(button, emb_model_button, llm_button):
    if button and emb_model_button and llm_button :
        # temp = st.session_state.temp
        if llm_button == 'OpenAI':
            llm = OpenAI(temperature=float(temp))
        else:
            llm = HuggingFaceHub(
                repo_id=llm_map[llm_button], 
                model_kwargs={"temperature":float(temp)}
            )
        chain = load_qa_chain( llm=llm, chain_type="stuff")

        if emb_model_button == 'OpenAI':
            embedding = OpenAIEmbeddings()   
        else:
            embedding = HuggingFaceEmbeddings(
                 model_name=embedding_map[emb_model_button],
            )
        global DOC_SPACE_DIR
        DOC_SPACE_DIR = f'{DOC_SPACE_DIR}_{llm_button}_{emb_model_button}'
        
        st.write(f'You selected: :green[{button}] :sunglasses:')
        st.write(f'You selected: :green[{emb_model_button}] :thumbsup:')
        st.write(f'You selected: :green[{llm_button}] 	:wink:')
        start_engine(button, embedding, chain)


# restart 
def restart():
    print(st.session_state)
    CHAT_COUNT += 1
    if 'button' in st.session_state:
        st.session_state['button'] = None 
    if 'emb_model_button' in st.session_state:
        st.session_state['emb_model_button'] = None
    if 'llm_button' in st.session_state:
        st.session_state['llm_button'] = None
    
    print(st.session_state)
# ____________________________________________________________________________________________________________________________
# ############################################################################################################################


# col1, col2, col3 = st.columns(3)

# with col1:
with st.sidebar:
    button = st.selectbox(
        'Select the source of the pdfs',
        ('Local', 'Upload', 'DropBox'),
        help = "Local source will used the pretrained doc. space",
        placeholder = "Select data source",
        index = None,
        key = "button",
    )  


    # select the embeddings model
    # with col2:
    emb_model_button = st.selectbox(
        'Select the embeddings model',
        ('GTE','OpenAI', 'MiniLM-L6-v2', 'jina-embeddings', 'mpnet'),
        placeholder="Select embeddings model",
        index=None,
        key="emb_model_button",
    )


    # select the generation model
    # with col3:
    llm_button = st.selectbox(
        'Select the generation model',
        ('flan-t5-xxl', 'OpenAI', 'Mistral-7B-Instruct', "long-t5-tglobal-xl", "long-t5-tglobal-base", "long-t5-tglobal-large", 'bloomz-560m', 'umt5-small', 'mt5-base' ),
        placeholder="Select generation model", 
        index=None,   
        key="llm_button",
    )


# st slider
temp = st.slider(
    'Select the temperature', 0.0, 1.0, 0.5, key="temp",
    # on_change=start_chat_bot, args=(button, emb_model_button, llm_button)
                         )

st.button("Restart", on_click=restart)
if button and emb_model_button and llm_button :
    start_chat_bot(button, emb_model_button, llm_button)




