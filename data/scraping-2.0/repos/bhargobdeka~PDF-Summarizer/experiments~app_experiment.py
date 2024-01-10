import streamlit as st
import os
import sys

from dotenv import load_dotenv # for loading environment variables

from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage,SystemMessage,AIMessage
from langchain.llms import OpenAI

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from htmlTemplates import css, bot_template, user_template

directory_path = os.getcwd()
if directory_path not in sys.path:
    sys.path.append(directory_path)
    
# function for reading pdf    
def read_pdf(pdf_file):
    pdfReader = PdfReader(pdf_file)
    count = len(pdfReader.pages)
    all_page_text = ""
    for i in range(count):
        page = pdfReader.pages[i]
        all_page_text += page.extract_text()
    return all_page_text

# function for splitting pdf into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# function for creating vectorstore
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# function for creating llm object
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    
    response = st.session_state.conversation({'question': user_question})
    
    st.session_state.chat_history = response['chat_history']
    # st.write(response["answer"])

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)



def main():
    load_dotenv() # load environment variables
    st.set_page_config(page_title='chat with PDF', page_icon='random', layout='wide', initial_sidebar_state='auto')
    st.write(css, unsafe_allow_html=True)
    
    # if key not in session state, set to None
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header('Chat with PDF')
    user_question = st.text_input('Ask your question here')
    # if user hits enter
    if user_question:
        handle_userinput(user_question)
    
    with st.sidebar:
        st.subheader('Add your PDF')
        pdf_doc = st.file_uploader('Upload your PDF here', type=['pdf'])
        if st.button('Submit'):
            st.write('Submitted')
            with st.spinner('Processing your PDF'):
                # load pdf
                raw_text = read_pdf(pdf_doc)
                # st.write(raw_text)
                
                # split pdf into chunks
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)
                
                # get vectorstore
                vectorstore = get_vectorstore(text_chunks)
                
                
                
                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)
             
    
    
if __name__ == '__main__':
    main()