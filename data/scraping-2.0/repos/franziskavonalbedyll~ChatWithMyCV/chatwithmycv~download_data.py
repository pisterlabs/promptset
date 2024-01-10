"""Download CV explanations from Azure Blob Storage."""
import streamlit as st
from azure.storage.blob import BlobServiceClient
from langchain.schema.document import Document


def download() -> Document:
    """Download CV explanations from Azure Blob Storage."""
    blob_service_client = BlobServiceClient(
        account_url=st.secrets["ACCOUNT_SAS_URL"]
    )

    blob_client = blob_service_client.get_blob_client(
        container="cvdata", blob="cv_explanations.txt"
    )

    blob_data = blob_client.download_blob().readall()
    content = blob_data.decode("utf-8")
    doc = Document(page_content=content)

    return doc
