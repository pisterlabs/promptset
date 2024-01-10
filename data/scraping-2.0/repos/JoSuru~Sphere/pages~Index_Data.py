import streamlit as st
from langchain.document_loaders import (
    PyPDFLoader,
)  # to read pdfs, urls

from sphere.index_data import main as index_data


def get_documents(url):
    loader = PyPDFLoader(url, extract_images=True)
    return loader.load()


st.set_page_config(page_title=" Sphere Assistant", layout="wide")


uploaded_file = st.file_uploader("Choose your .pdf file", type="pdf")
if uploaded_file is not None:
    with st.spinner("Indexation en cours..."):
        # save uploaded file on file "data"
        with open(f"data/{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getbuffer())

        pdf_reader = get_documents(f"data/{uploaded_file.name}")
        df = index_data(pdf_reader)
    st.success("Indexation terminée !")
