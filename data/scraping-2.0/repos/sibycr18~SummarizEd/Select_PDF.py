import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import re
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb

## Intitialization
# Intialize ChromaDB
@st.cache_resource
def init_db():
    db_client = chromadb.PersistentClient(path="./db")
    return db_client

# Initialize Embeddings
@st.cache_resource
def init_embedding():
    embeddings = HuggingFaceEmbeddings(model_name="infgrad/stella-base-en-v2")
    return embeddings

def sanitize_string(input_str):
    # Remove non-alphanumeric, underscores, hyphens, and periods
    sanitized = re.sub(r"[^A-Za-z0-9_.-]", "", input_str)

    # Replace consecutive periods with a single period
    sanitized = re.sub(r"\.{2,}", ".", sanitized)

    # Ensure the string starts and ends with an alphanumeric character
    sanitized = re.sub(r"^[^A-Za-z0-9]+", "", sanitized)
    sanitized = re.sub(r"[^A-Za-z0-9]+$", "", sanitized)

    # Truncate or pad string to meet the 3-63 character length requirement
    sanitized = sanitized[:63] if len(
        sanitized) > 63 else sanitized.ljust(3, "_")

    return sanitized

st.set_page_config(
    page_title="SummarizEd.ai",
    page_icon="📚",
    layout="centered",
)

# Session states
db_client = st.session_state.db_client = init_db()
embeddings = st.session_state.embeddings = init_embedding()

# Already uploaded files
collections = st.session_state.db_client.list_collections()

## App Title
# st.title("Summariz:orange[Ed] :gray[- PDF Summarizer]")
st.title("Summariz:orange[Ed]:grey[.ai]")
st.subheader("", divider="gray")    # maybe not be a proper way but i like this


pdf_list = tuple(collection.name for collection in collections)
placeholder = "Select the PDF file to search..." if len(pdf_list) > 0 else "No PDFs uploaded"
file_name = st.selectbox(
   "Select PDF file:",
   pdf_list,
   index=None,
   placeholder = placeholder,
   label_visibility="visible"
)
# st.session_state.selected_file = selected_file

st.subheader("OR")


# Display file uploader
uploaded_file = st.file_uploader("Upload a new PDF file", type=["pdf"])

if uploaded_file is not None:
    file_name = sanitize_string(uploaded_file.name)
    # Read and display the content of the PDF file
    pdf_reader = PdfReader(uploaded_file)
    pdf_text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        pdf_text += page.extract_text()

    # print(pdf_text)

    if st.button("Process PDF", type="primary"):
        if file_name in {collection.name for collection in collections}:
                st.warning(
                    f"PDF '{file_name}' has already been processed. Select it from the above list.")
        else:
            with st.spinner("Processing PDF..."):
                ## Db insertion
                collection = db_client.create_collection(name=file_name)

                # Split text into chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = text_splitter.split_text(pdf_text)

                # Convert chunks to vector representations and store in ChromaDB
                documents_list = []
                embeddings_list = []
                ids_list = []

                for idx, chunk in enumerate(chunks):
                    vector = embeddings.embed_query(chunk)
                    documents_list.append(chunk)
                    embeddings_list.append(vector)
                    ids_list.append(f"{file_name}_{idx}")

                collection.add(
                    embeddings=embeddings_list,
                    documents=documents_list,
                    ids=ids_list
                )
            st.success("PDF has been processed successfully")


st.session_state.file_name = file_name
