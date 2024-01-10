import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import (OpenAIEmbeddings)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import (RecursiveCharacterTextSplitter)
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space

from template import bot_template, css, user_template

#load api key
load_dotenv()
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 128,
        chunk_overlap = 28,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    # vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings, dimension)
    vectorstore = FAISS.from_documents(documents=text_chunks, embedding=embeddings)

    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


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
    st.set_page_config(page_title="🤖RUSSELL-BOT 🤖",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("💬Chat with your PDFs📄📄")
    user_question = st.text_input("Feel free to ask any questions you have about your PDFs.")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your pdf documents")
        pdf_docs = st.file_uploader(
            "Upload PDF file(s) and click on the Process", accept_multiple_files=True, type='pdf')

        # if pdf_docs is not None:
        #     st.write(pdf_docs.)

        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)
                
        add_vertical_space(3)
        st.markdown('''
            ## About APP:
            - This app is a demo for pdf based chatbot.
            - You can upload your pdf file and ask questions about it.


            ## About me:
            - [Linkedin](https://www.linkedin.com/in/rcaliskan/)
            - [Github](https://github.com/russell-ai)
            
            ''')
        add_vertical_space(2)
        st.write('Made with ❤️ by Russell AI')
        st.write('© 2023 Russell AI. All rights reserved.')

if __name__ == '__main__':
    main()


# streamlit run app-v2.py