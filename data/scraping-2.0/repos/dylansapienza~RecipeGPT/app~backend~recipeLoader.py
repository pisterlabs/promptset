import langchain
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import os
import getpass

os.environ['OPENAI_API_KEY'] = getpass.getpass('OpenAI API Key:')


def combine_documents(documents):
    combined_documents = []

    building_document_content = ""
    current_source = documents[0].metadata.get(
        'source')  # Initialize from the first document
    for document in documents:
        if document.metadata.get('source') == current_source:
            building_document_content += document.page_content
        else:
            combined_documents.append(langchain.schema.document.Document(
                metadata={"source": current_source}, page_content=building_document_content))
            building_document_content = document.page_content
            current_source = document.metadata.get('source')

    # Append the last document
    combined_documents.append(langchain.schema.document.Document(
        metadata={"source": current_source}, page_content=building_document_content))

    return combined_documents


embeddings = OpenAIEmbeddings()


loader = PyPDFDirectoryLoader("./data/")
documents = loader.load()

combined_documents = combine_documents(documents)

db = FAISS.from_documents(combined_documents, embeddings)
db.save_local("./faiss_index_combined")
