"""
This script creates a database of information gathered from local text files.
"""

from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def load():
    # define what documents to load
    # loader = DirectoryLoader("./", glob="*.txt", loader_cls=TextLoader)
    loader = PyPDFLoader("data/Business Conduct.pdf")
    # interpret information in the documents
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                              chunk_overlap=50)
    texts = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'})
    # create and save the local database
    db = FAISS.from_documents(texts, embeddings)
    db.save_local("faiss")


if __name__ == '__main__':
    load()