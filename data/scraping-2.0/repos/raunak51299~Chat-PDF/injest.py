from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Loading document
DATA_DIR = "data/"
DB_FAISS_PATH = "vectorstores/db_faiss"

#create vector database
def create_vector_database():
    loader = DirectoryLoader(DATA_DIR, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2",
    model_kwargs = {'device': 'cpu'})

    db = FAISS.from_documents(text ,embeddings)
    db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    create_vector_database()
   