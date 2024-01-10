from langchain.memory import ConversationBufferMemory
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings,HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from laws import constitution
# import numpy as np

from HtmlTemplates import css,bot_template,user_template

def get_pdf_text(pdf_docs):
    
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks
    
def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    # test = np.load('savefile.npy',allow_pickle=True)
    # if (test):
    #     return test
    embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding = embeddings)
    # vs = np.array(vectorstore)
    # np.save('savefile.npy',vs)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key = 'chat_history',return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever = vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain
    
def handle_userinput(user_question):
    response = st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']
    
    for i,message in enumerate(st.session_state.chat_history):
        if i%2==0:
            st.write(user_template.replace('{{MSG}}',message.content),unsafe_allow_html=True)
        else:
            st.write(bot_template.replace('{{MSG}}',message.content),unsafe_allow_html=True)
            
def main():
    
    load_dotenv()
    st.set_page_config(page_title="Prototype",page_icon=":books:")
    st.write(css,unsafe_allow_html=True)
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None
    with st.spinner("Processing"):
        
        #raw_text = get_pdf_text(['2023050195.pdf'])
        
        text_chunks = get_chunks(constitution)
        
        vectorstore = get_vectorstore(text_chunks)
        
        st.session_state.conversation = get_conversation_chain(vectorstore)
    
    st.header("Legal Assistant :)")
    user_question = st.text_input("Ask a question :")
    if user_question:
        handle_userinput(user_question)
    st.write(user_template.replace("{{MSG}}","Hey Assistant"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}","Hey! How do you do.."), unsafe_allow_html=True)
    
    
    
    
    
if __name__=='__main__':
    main()