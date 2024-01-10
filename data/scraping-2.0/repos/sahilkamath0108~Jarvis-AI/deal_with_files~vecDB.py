from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
   
def vectorize(path):
    # loader = DirectoryLoader(path, glob="./*.pdf", loader_cls=PyPDFLoader)
    # documents = loader.load()
    loader_cls = PyPDFLoader
    loader = loader_cls(path)
    documents = loader.load()
   
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    embeddings = SentenceTransformerEmbeddings(model_name="multi-qa-mpnet-base-dot-v1")
    persist_directory = "data"
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
    
    return True

if __name__ == '__main__':
    path = "C:\\Users\\Hp\\Desktop\\realmadrid.pdf"
    vectorize(path)
