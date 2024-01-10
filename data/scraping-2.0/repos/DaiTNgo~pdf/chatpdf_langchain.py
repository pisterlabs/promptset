from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import openai
import os
import streamlit as st
load_dotenv()

openai.api_key = os.environ.get('OPENAI_API_KEY=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')
chat = ChatOpenAI(model_name="gpt-3.5-turbo-0613", temperature=0)
st.title("Chat with PDF")
pdf = st.file_uploader('Upload your PDF Document', type='pdf')
# Load the PDF using PyPDFLoader
if pdf is not None:
    loader = PyPDFLoader("Chí Phèo.pdf")
    pages = loader.load_and_split()

    # Define a text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    # Split text into documents
    docs = text_splitter.split_documents(pages)

    # Create sentence embeddings using SentenceTransformer
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create a Chroma vector store from the documents
    docsearch = Chroma.from_documents(docs, embedding_function,persist_directory=r"\chromadb")

    # Persist the vector store
    docsearch.persist()

    # Create a RetrievalQA instance
    qa = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={"k": 2}))

    # Define and run a query
    query = st.text_input("Enter your query")
    if st.button("Search"):
        if query:
            # Create a RetrievalQA instance
            qa = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={"k": 2}))

            # Run the query
            results = qa.run(query)

            # Display the results
            st.write("Results:")
            st.write(results)
        else:
            st.warning("Please enter a query")
st.text("© 2023 Your App Name")