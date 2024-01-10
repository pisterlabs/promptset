import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.neo4j_vector import Neo4jVector
from streamlit.logger import get_logger
from chains import load_embedding_model
from dotenv import load_dotenv

load_dotenv(".env")

# Neo4j and embedding configurations
url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
os.environ["NEO4J_URL"] = url

logger = get_logger(__name__)

embeddings, _ = load_embedding_model(
    embedding_model_name, config={"ollama_base_url": ollama_base_url}, logger=logger
)


def process_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:  # Add text only if it exists
            text += page_text

    # Splitting text for better handling
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text=text)

    # Store in Neo4j
    Neo4jVector.from_texts(
        chunks,
        url=url,
        username=username,
        password=password,
        embedding=embeddings,
        index_name="pdf_storage",
        node_label="PdfDocument",
        pre_delete_collection=False,  # Set to True if you want to clear previous data
    )


def main():
    st.title("PDF Data Uploader")
    st.subheader("Upload PDF files to store their data in Neo4j")

    # Upload PDF
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if pdf is not None:
        with st.spinner("Processing PDF..."):
            process_pdf(pdf)
            st.success("PDF processed and data stored successfully!")


if __name__ == "__main__":
    main()
