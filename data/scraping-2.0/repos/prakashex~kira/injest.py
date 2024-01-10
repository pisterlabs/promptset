from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader , TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


DATA_PATH = "data/"
DATA_FAISS = "vector_store/db_faiss"

# create a vector store

def create_vector_store():
    loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    text_spiltter = RecursiveCharacterTextSplitter(chunk_size=500 , chunk_overlap=50)
    texts = text_spiltter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name= "sentence-transformers/all-MiniLM-L6-v2", model_kwargs= {"device": "cpu"})

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DATA_FAISS)

if __name__ == "__main__":
    create_vector_store()



