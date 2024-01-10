import streamlit as st
from PyPDF2 import PdfReader
from io import BytesIO
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

# Set OpenAI API key as an environment variable
os.environ['OPENAI_API_KEY'] = 'your_api_key_here'

# Function to extract text from PDF


def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() if page.extract_text() else ""
    return text

# Function to extract and process information


def extract_and_process_information(text, categories):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=text)

    embeddings = OpenAIEmbeddings()
    VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

    extracted_info = {}
    for category in categories:
        query = f"Extract the {category} of the document."
        docs = VectorStore.similarity_search(query=query, k=3)

        llm = OpenAI()
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=query)
        extracted_info[category] = response

    return extracted_info

# Main function of the Streamlit app


def main():
    st.title('Research Paper AI📚')

    # Upload existing Excel file
    existing_excel = st.file_uploader(
        "Upload Existing Excel File (optional)", type=['xlsx'])

    # Initialize DataFrame from existing Excel or create a new one
    if existing_excel is not None:
        df_existing = pd.read_excel(existing_excel)
    else:
        df_existing = pd.DataFrame(
            columns=["Title", "Author Name", "Methodology", "Conclusions", "Future Works"])

    # Upload PDF
    pdf_file = st.file_uploader("Upload your PDF", type='pdf')

    if pdf_file is not None:
        text = extract_text_from_pdf(pdf_file)
        categories = ["Title", "Author Name",
                      "Methodology", "Conclusions", "Future Works"]
        extracted_info = extract_and_process_information(text, categories)

        # Append new data to the DataFrame
        df_existing = df_existing.append(extracted_info, ignore_index=True)

        # Display updated DataFrame (optional)
        st.dataframe(df_existing)

    # Export to Excel button
    if st.button("Export to Excel"):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_existing.to_excel(writer, index=False)
            writer.save()
        st.download_button(label="Download Updated Excel file",
                           data=output.getvalue(),
                           file_name="updated_information.xlsx",
                           mime="application/vnd.ms-excel")


if __name__ == '__main__':
    main()
