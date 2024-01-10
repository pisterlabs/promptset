import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import Together
from transformers import pipeline

from htmlTemplates import bot_template, user_template, css

os.environ["HUGGINGFACEHUB_API_TOKEN"]="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"

def get_pdf_text(pdf_files):
    text = ""
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def get_chunk_text(text):

    text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap = 200,
    length_function = len
    )

    chunks = text_splitter.split_text(text)

    return chunks


def get_vector_store(text_chunks):
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_texts(text_chunks, embeddings)
    
    return vectorstore

def get_conversation_chain(vector_store):

    llm = Together(
    model="togethercomputer/llama-2-70b-chat",
    temperature=0.5,
    max_tokens=512,
    top_k=50,
    together_api_key="c569ea6d1993191e2d591c7d2103ca37f5669807a1a2534981be856c7fd605b7"
        )

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vector_store.as_retriever(),
        memory = memory
    )

    return conversation_chain

def handle_user_input(question):

    response = st.session_state.conversation({'question':question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title='Chat with Your own PDFs', page_icon=':books:')

    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header('Chat with Your own Documents :books:')
    question = st.text_input("Ask anything to your PDF: ")

    if question:
        handle_user_input(question)
    
    with st.sidebar:
        st.subheader("Upload your Documents Here (as many you want!!): ")
        pdf_files = st.file_uploader("Choose your PDF Files and Press OK", type=['pdf'], accept_multiple_files=True)

        if st.button("OK"):
            with st.spinner("Processing your PDFs..."):

                # Get PDF Text
                raw_text = get_pdf_text(pdf_files)
                print("Text Extraction Completed")
                
                # Get Text Chunks
                text_chunks = get_chunk_text(raw_text)
                print("Chunking Completed")
                
                # Create Vector Store
                vector_store = get_vector_store(text_chunks)
                print("Embeddings Completed")
                st.write("DONE")

                # Create conversation chain
                st.session_state.conversation =  get_conversation_chain(vector_store)
                print("conversation Chain Started!")

if __name__ == '__main__':
    main()
