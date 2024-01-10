

# imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# constants
DIR_DATA = "/home/javaprog/Data/"
DIR_DOCS = DIR_DATA + "ML/Llama2Test/PPARG/Docs"
DIR_DB = DIR_DATA + "ML/Llama2Test/PPARG/VectorStore"

def create_vector_store(dir_db, dir_docs, log=False):
    '''
    create the vector store
    '''
    # load the documents
    loader = DirectoryLoader(DIR_DOCS, glob='*.docx', loader_cls=Docx2txtLoader, recursive=True)
    documents = loader.load()
    if log:
        print("loading the documents in: {}".format(dir_docs))

    # split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    if log:
        print("splitting the documents in: {}".format(dir_docs))

    # embedd the docs and save the file
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(dir_db)
    if log:
        print("saved embeddings to: {}".format(dir_db))

if __name__ == "__main__":
    create_vector_store(dir_db=DIR_DB, dir_docs=DIR_DOCS, log=True)

