import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from InstructorEmbedding import INSTRUCTOR
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import (HuggingFaceInstructEmbeddings, OpenAIEmbeddings)
from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from pypdf import PdfReader

from htmlTemplates import bot_template, css, user_template


# Function to extract the text of pdf files
def get_text_pdf(uploaded_files):
    """
    Generate a text representation of a PDF document.

    Args:
        uploaded_files (list): A list of paths to the uploaded PDF files.

    Returns:
        str: The combined text from all the pages of the PDF documents.

    """
    doc_text = ''
    for pdf in uploaded_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            doc_text += page.extract_text()
    return doc_text


def get_chunks_text(doc_text):
    """
    Generate the chunks of text from the given document text.

    Args:
        doc_text (str): The document text from which to generate the chunks.

    Returns:
        List[str]: A list of chunks of text generated from the document text.
    """
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(doc_text)
    return chunks


def get_vectorstore(text_chunks):
    """
    Generate a vector store from a list of text chunks.
    
    Args:
        text_chunks (List[str]): A list of text chunks to generate the vector store from.
        
    Returns:
        vectorstore (FAISS): The generated vector store.
    """
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(
    #     model_name='hkunlp/instructor-xl'
    # )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    """
    Generate a conversation chain using a vector store.

    Args:
        vectorstore (VectorStore): The vector store used for retrieval.

    Returns:
        ConversationalRetrievalChain: The conversation chain object.
    """
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain



def handle_userinput(user_question):
    """
    Handles user input by sending it to a conversation API and displaying the response in a chat format.

    Parameters:
    - user_question (str): The user's input question.

    Returns:
    - None
    """
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace('{{MSG}}', message.content),
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace('{{MSG}}', message.content),
                unsafe_allow_html=True,
            )


def main():
    """
    The main function that initializes the OpenAI Chatbot of CogniCore.

    This function sets up the page configuration, writes the CSS, and handles user input.

    Parameters:
        None

    Returns:
        None
    """

    load_dotenv()
    st.set_page_config(
        page_title='OpenAI Chatbot',
        page_icon='🤖',
    )

    st.write(css, unsafe_allow_html=True)

    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None

    st.header('OpenAI Chatbot:🤖')

    col1 = st.columns(spec=1, gap='small')
    st.header('ChatBot:')
    st.text('Resume about what the people going on about.')

    user_question = st.text_input('Ask a question: ')
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader('Ask a question: ')

        uploaded_files = st.file_uploader(
            'Choose a PDF file', accept_multiple_files=True
        )
        if st.button('Submit'):
            with st.spinner('Please wait...'):
                # Get pdf text
                raw_text = get_text_pdf(uploaded_files)
                st.write(raw_text)

                # Get the text chunks
                text_chunks = get_chunks_text(raw_text)
                st.write(text_chunks)

                # Get the vector store
                vectorstore = get_vectorstore(text_chunks)

                # Create a conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore
                )


if __name__ == '__main__':
    main()