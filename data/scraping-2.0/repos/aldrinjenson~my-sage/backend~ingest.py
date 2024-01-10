from constants import CHROMA_SETTINGS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import UnstructuredURLLoader, PyPDFLoader
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
import os
from dotenv import load_dotenv
load_dotenv()


#persist_directory = os.environ.get('PERSIST_DIRECTORY')
source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')


def load_data(urls, filepath):
    #    urls = ["https://en.wikipedia.org/wiki/Frozen_(2013_film)"]
    url_loader = UnstructuredURLLoader(urls=urls)
    url_docs = url_loader.load()
    print('loaded urls')

    pdf_loader = PyPDFLoader(filepath)
    print("going to load pdfs")
    pdf_docs = pdf_loader.load()
    print('loaded pdfs')

    # Combine documents
    docs = url_docs + pdf_docs
    return docs


def ingest_docs(urls, filepath, userId):
    if not urls or not filepath or not userId:
        print("parameters whole not passed")
        return
    print('gonna start loading')
    raw_documents = load_data(urls, filepath)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(raw_documents)

    print("creating embeddings")
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    persist_directory = "models/"+userId
    print("Creating vector store")
    db = Chroma.from_documents(
        documents, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None
    print("Done saving")
    return "Done"


if __name__ == "__main__":
    ingest_docs()
