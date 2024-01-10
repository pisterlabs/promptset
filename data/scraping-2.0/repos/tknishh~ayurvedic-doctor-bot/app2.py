import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Fixed directory containing PDF documents
pdf_directory = "data/"

# Function to list PDF files in a directory
def list_pdf_files(directory):
    pdf_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            pdf_files.append(os.path.join(directory, filename))
    return pdf_files

# Function to extract data from all the PDF documents
def get_data(docs):
    data = ""
    for doc in docs:
        pdf_reader = PdfReader(doc)
        for page in pdf_reader.pages:
            data += page.extract_text()
    return data

# Initialize the vector store and conversation chain only once per runtime
vectorstore = None
conversation_chain = None

def initialize_vectorstore():
    global vectorstore
    global conversation_chain
    if vectorstore is None:
        # Get data from the documents in the directory
        pdf_files = list_pdf_files(pdf_directory)
        if not pdf_files:
            st.warning("No PDF files found in the specified directory.")
        else:
            raw_data = get_data(pdf_files)

            # Divide data into chunks
            chunks = get_chunks(raw_data)

            # Convert to embeddings and create a vector store
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)

            if vectorstore:
                st.caption("Vector Store Initialized!")

            # Create a conversation chain (user input)
            conversation_chain = get_conversation_chain(vectorstore)

# ... (other functions remain the same)

def main():
    load_dotenv()

    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        st.write("OPENAI_API_KEY is not set. Please add your key in .env file.")
        exit(1)

    st.set_page_config(page_title="Ayurvedic Doctor Chatbot", page_icon="💬")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Ayurvedic Doctor Chatbot 💬")

    with st.chat_message("assistant"):
        st.write("Hello👋, How can I help you today?")

    user_input = st.chat_input("Ask your query")

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        handle_user_input(user_input)

    initialize_vectorstore()  # Initialize vector store if not done already

    # Process user queries using the existing vector store and conversation chain
    if conversation_chain:
        with st.spinner("Processing..."):
            st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()
