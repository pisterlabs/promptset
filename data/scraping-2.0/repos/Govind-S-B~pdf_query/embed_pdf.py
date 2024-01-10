import streamlit as st
import re
import chromadb
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader


@st.cache_resource
def init_db():
    db_client = chromadb.PersistentClient(path="./db")
    return db_client


@st.cache_resource
def init_embedding():
    # embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
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


st.session_state.db_client = init_db()
st.session_state.embeddings = init_embedding()

client = st.session_state.db_client
embeddings = st.session_state.embeddings

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100)

st.title('Embed PDF')
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    if uploaded_file.type == 'application/pdf':
        if st.button("Embed"):
            file_name = sanitize_string(uploaded_file.name)

            collections = client.list_collections()
            if file_name in {collection.name for collection in collections}:
                st.warning(
                    f"PDF '{file_name}' has already been processed. Please test with a different PDF.")
            else:
                collection = client.create_collection(name=file_name)

                # Convert PDF to text
                pdf_reader = PdfReader(uploaded_file)
                text = ""
                for page_num in range(len(pdf_reader.pages)):
                    text += pdf_reader.pages[page_num].extract_text()

                 # Split text into chunks
                chunks = text_splitter.split_text(text)

                # Convert chunks to vector representations and store in Chroma DB
                documents_list = []
                embeddings_list = []
                ids_list = []

                for i, chunk in enumerate(chunks):
                    vector = embeddings.embed_query(chunk)

                    documents_list.append(chunk)
                    embeddings_list.append(vector)
                    ids_list.append(f"{file_name}_{i}")

                collection.add(
                    embeddings=embeddings_list,
                    documents=documents_list,
                    ids=ids_list
                )

                st.success("PDF has been vectorized")

                # check if collecton under filename exist or not
                # if exist send warnin
                # else create a new collection in chroma db and store it in .db
                # vectorize and store the pdf
                # send message indicating success that the pdf has been vectorized
