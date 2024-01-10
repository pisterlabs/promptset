from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.document_loaders import UnstructuredHTMLLoader, BSHTMLLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

DB_PATH = "vectorstores/db/"

def create_vector_db():
    loader = PyPDFLoader('./doc.pdf')
    documents = loader.load()
    print(f"Processed {len(documents)} pdf files")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts=text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=texts, embedding=HuggingFaceEmbeddings(),persist_directory=DB_PATH)      
    vectorstore.persist()

if __name__=="__main__":
    create_vector_db()