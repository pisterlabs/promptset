import streamlit as st
import os
import openai
from dotenv import find_dotenv, load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

load_dotenv(find_dotenv())
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

#gets the text from the pdfs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks # returns list of chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings) # creating database
    return vectorstore


#creates the converstaion chain with langchain
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# what to do when the user inputs something
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question}) # note conversation chain already has all the configuration with the vector store and memory
    # st.write(response)
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            # this is for the picture and text boxes
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv(find_dotenv())
    

    st.set_page_config(page_title="Multiple PDFs ChatBot", page_icon=":scroll:")
    st.write(css, unsafe_allow_html=True)
    # openai_api_key = st.session_state.get("OPENAI_API_KEY")
    

    if "conversation" not in st.session_state: # initialize it here when using session state, can use this variable globally now and does not ever reset
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header(":sun_with_face: Chat With Multiple PDFs :scroll:")
    user_question = st.text_input("Ask a question about your documents:")

    if (user_question): # if the user inputs a question
        handle_userinput(user_question)

    
    
    # this creates the sidebar to upload the pdf docs to
    with st.sidebar:
        st.markdown(
            "## How to use\n"
            "1. Upload as many pdf's as you like\n"  # noqa: E501
            "2. Process the documents\n"
            "3. Start talking to your pdf's!"
        )

        st.markdown("---")
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)  # add session state so that this variable is never reinitialized when user pushes button or something like that, streamlit does that sometimes
                


if __name__=='__main__':
    main()