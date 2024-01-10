import tempfile
import camelot
import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=".env")

def extract_tables_from_pdf(uploaded_pdf):
    # Save the PDF from the upload to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
        temp_pdf.write(uploaded_pdf.getbuffer())
        temp_pdf_path = temp_pdf.name

    # Now that we have a file path, we can use Camelot to read the PDF
    tables = camelot.read_pdf(temp_pdf_path, pages='all', flavor='lattice')

    return [table.df for table in tables]

def answer_question(question, data):
    # Split data into chunks
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(data)

    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Create a knowledge base
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    # Search the knowledge base for documents related to the user's question
    docs = knowledge_base.similarity_search(question)

    # Initialize an OpenAI model
    llm = OpenAI()

    # Load a question-answering chain using the OpenAI model
    chain = load_qa_chain(llm, chain_type="stuff")

    # Generate a response
    response = chain.run(input_documents=docs, question=question)
    return response

def main():
    st.title('PDF Table Extractor and Question Answering System')

    # Step 1: User uploads a PDF
    uploaded_pdf = st.file_uploader("Upload a PDF containing tables", type="pdf")
    
    if uploaded_pdf:
        # Step 2: Extract tables from the PDF
        try:
            extracted_tables = extract_tables_from_pdf(uploaded_pdf)
            table_names = [f"Table {i+1}" for i in range(len(extracted_tables))]
            st.success('Tables extracted successfully!')
        except Exception as e:
            st.error(f'An error occurred when extracting tables: {e}')
            return

        # Sidebar to select a table to view
        table_to_view = st.sidebar.selectbox("Select a table to view", table_names)
        view_index = table_names.index(table_to_view)
        st.subheader(f'Viewing {table_to_view}')
        st.dataframe(extracted_tables[view_index])

        # Sidebar to select a table to ask a question on
        table_to_question = st.sidebar.selectbox("Select a table to ask a question on", table_names)
        question_index = table_names.index(table_to_question)

        # User inputs a question
        user_question = st.sidebar.text_input(f"Enter your question related to {table_to_question}")

        if user_question:
            # Utilize Language Model
            data_as_string = extracted_tables[question_index].to_string(index=False)
            answer = answer_question(user_question, data_as_string)

            # Present the answer
            st.subheader('Answer')
            st.write(answer)

if __name__ == "__main__":
    main()
