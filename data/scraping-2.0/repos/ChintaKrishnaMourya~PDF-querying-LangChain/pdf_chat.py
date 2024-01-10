import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os

####set openai key

# Streamlit app title and description
st.title("Talk to PDF")
st.subheader("Ask what's in pdf")

# PDF file upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Check if a file is uploaded
if uploaded_file is not None:
    # Read text from PDF
    pdfreader = PdfReader(uploaded_file)
    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content

    # Split the text using Character Text Split
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    # Download embeddings from OpenAI
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    document_search = FAISS.from_texts(texts, embeddings)

    # Load the question answering chain
    chain = load_qa_chain(OpenAI(), chain_type="stuff")

    # User input for search query
    user_input = st.text_input("Enter your search query")

    # Search and display results
    if st.button("Search"):
        if user_input:
            # Perform the search
            docs = document_search.similarity_search(user_input)
            result = chain.run(input_documents=docs, question=user_input)

            # Display the result
            st.write("Search Result:")
            st.write(result)

        else:
            st.warning("Please enter a search query.")
