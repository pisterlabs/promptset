import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import logging
import os
from pymongo import MongoClient
import certifi
from htmlTemplates import css, bot_template, user_template

from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub

# C:\marcuslion\pe_pdf> .\venv311\Scripts\streamlit run app_ab.py

logger = logging.getLogger('streamlit_app')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
#queryRun = False
max_doc_text = 7000
max_total_text = 100000
use_open_ai = True

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_doc_text():
    allText = ""
    # load user + pswd from .env
    load_dotenv()
    user = os.environ.get('MONGODB_USER')
    pswd = os.environ.get('MONGODB_PASSWORD')
    # mongosh -u nick -p <pswd> mongodb+srv://nick:<pswd>@cluster0.5346qcs.mongodb.net/SEC
    port = 27017 #default
    CONNECTION_STRING = 'mongodb+srv://' + user + ':' + pswd + '@cluster0.5346qcs.mongodb.net/SEC'
    searchDoc = '10-K'
    searchDate = '2023-10-12'
    client = MongoClient(CONNECTION_STRING, tlsCAFile=certifi.where())

    # Getting the database instance
    db = client['SEC']

    # Creating a collection
    colDocs = db['company-doc']
    logger.debug('db: ' + str(db))
    logger.debug('coll: ' + str(colDocs))

    #load title to ticker
    allDocs = colDocs.find()
    #docCount = allDocs.count()
    #logger.info(str(docCount) + " documents found")

    stOutput = ""
    for doc in allDocs:
        id = doc['_id']
        try:
            company = doc['company_name']
            if len(company) > 0:
                text = (doc['text'])[0:max_doc_text]
                allText += text
                logger.info("ID " + str(id) + " Company " + str(company) + "processed")
                if len(stOutput) < 2:
                    stOutput = "Loaded 10-K documents from " + str(company)
                else:
                    stOutput += (" - " + str(company))
        except KeyError:
            logger.info("ID " + str(id) + " Company not found")
            #st.write(company)
        if len(allText) > max_total_text:
            break
    st.markdown(stOutput + ", " + str(len(allText)) + " total lines of text.")
    return allText


def get_text_chunks(text):
    #text_splitter = CharacterTextSplitter(
    #    separator= "\n",
    #    chunk_size=900,
    #    chunk_overlap=100,
    #    length_function=len
    #)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    if use_open_ai:
        embeddings = OpenAIEmbeddings()
    else:
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    if use_open_ai:
        llm = ChatOpenAI()
    else:
        llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    if not user_question is None:
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
    global queryRun

    load_dotenv()
    st.set_page_config(page_title="MarcusLion Financial Analyst",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with MarcusLion :books:")
    user_question = st.text_input("Ask a question about your 10-K documents:")
    queryRun = False
    if user_question:
        handle_userinput(user_question)
    with st.sidebar:
        st.subheader("Your companies")
        if st.button("Process"):
            with st.spinner("Processing"):
                # prepare all documents

                # get text from MongoDB
                if not queryRun:
                    queryRun = True
                    raw_text = get_doc_text()

                # get the text chunks
                st.write("Now Chunking Text")
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                st.write("Set Embeddings and Vector Store")
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.write("Set Conversation Chain")
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()