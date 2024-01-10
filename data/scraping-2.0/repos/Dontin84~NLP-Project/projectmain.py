import os
import pickle
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
import langchain
import pandas as pd
import streamlit as st

langchain.verbose = False

# Load environment variables from .env
os.environ['OPENAI_API_KEY'] = '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'



llm = OpenAI(model="text-davinci-003", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

file_path = "faiss_store_openai.pkl"

# Function to process URLs and create embeddings
def process_urls(urls):
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    docs = text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore_openai = FAISS.from_documents(docs, embeddings)

    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

# Function to handle queries
def handle_query(query):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            return result

# Define the Streamlit app in a function
def streamlit_app():
    st.title("Streamlit App for Langchain")

    # Sidebar with user input
    num_urls = st.number_input("Enter the number of URLs (max 3)", min_value=1, max_value=3, step=1, value=2)

    urls = [st.text_input(f"Enter URL {i + 1}") for i in range(num_urls)]
    query = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        if not any(urls):
            st.warning("Please enter at least one URL.")
        elif query:
            process_urls(urls)
            result = handle_query(query)
            st.write("Answer:", result["answer"])

            # Display sources if available
            if result.get("sources"):
                st.write("Sources:", result["sources"])
        else:
            st.warning("Please enter a question.")

# Run the Streamlit app
streamlit_app()
