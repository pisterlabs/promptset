import modules.settings as settings

import streamlit as st
import pypdf
import validators

from langchain.embeddings import OllamaEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import SeleniumURLLoader
from langchain.vectorstores import Chroma
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.document_loaders.parsers.pdf import (
    PyPDFParser,
)

def splitAndImport(link, data):
    with st.spinner("Splitting into documents…"):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
        all_splits = text_splitter.split_documents(data)

    numDocuments = len(all_splits) + 1

    progressText = "Importing " + str(numDocuments) + " documents"
    progressBar = st.progress(0, text = progressText)

    for i in range (len(all_splits)):
        settings.chroma.add_texts(ids=[link + "." + str(i)], metadatas=[all_splits[i].metadata], texts=[all_splits[i].page_content])
        progressText = "Importing " + str(numDocuments - i) + " documents"
        progressBar.progress((i + 1) / numDocuments, progressText) 
#                progressBar.progress(i + 1, all_splits[i].page_content[:100].replace("\n", " ") + "…") 
    progressBar.empty()
    st.write("Finished import.")


settings.init()

st.set_page_config(page_title="Import documents", page_icon="📄")

st.markdown("# Import documents")

tabUploadPDF, tabPDF, tabHTML = st.tabs(["Upload PDF", "Import PDF from URL", "Import Website"])

with tabUploadPDF:
    uploadedFiles = st.file_uploader('Upload your PDF Document', type='pdf', accept_multiple_files=True)

    for uploadedFile in uploadedFiles:
        st.spinner("Importing " + uploadedFile.name)
        parser = PyPDFParser(extract_images=False)
        blob = Blob.from_data(data = uploadedFile.read(), path = uploadedFile.name)
        data = parser.parse(blob)
        reader = pypdf.PdfReader(uploadedFile)
        splitAndImport(uploadedFile.name, data)

with tabPDF:
    with st.form("tabPDF_Form"):
       pdfLink = st.text_input('Link to PDF document', placeholder="https://example.com/example.pdf")
       pdfLinkSubmitted = st.form_submit_button("Import")

    if pdfLinkSubmitted:
        pdfLink = pdfLink.strip()
        if validators.url(pdfLink):
            with st.spinner("Loading " + pdfLink + "…"):
                loader = PyPDFLoader(pdfLink, extract_images=False)
                data = loader.load()
            splitAndImport(pdfLink, data)
        else:
           st.write("**Please input a valid URL**")

with tabHTML:
    with st.form("tabHTML_Form"):
       htmlLink = st.text_input('Link to Web page', placeholder="https://example.com/")
       htmlLinkSubmitted = st.form_submit_button("Import")
       st.write("*Note: This will only import the page at the URL given. Subpages will not be crawled.*")

    if htmlLinkSubmitted:
        htmlLink = htmlLink.strip()
        if validators.url(htmlLink):
            with st.spinner("Loading " + htmlLink + "…"):
                loader = SeleniumURLLoader(urls=[htmlLink])
                data = loader.load()
            splitAndImport(htmlLink, data)
        else:
           st.write("**Please input a valid URL**")