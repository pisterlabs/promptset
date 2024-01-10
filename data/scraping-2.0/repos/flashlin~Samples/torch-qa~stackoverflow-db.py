from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

DATA_PATH = "data-StackOverflow/"
DB_FAISS_PATH = "models/db_faiss"


def create_vector_db():
    loader = DirectoryLoader(
        DATA_PATH,
        loader_cls=TextLoader,
        recursive=True,
        show_progress=True,
        use_multithreading=True,
        max_concurrency=8
    )
    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    texts = text_splitter.split_documents(raw_documents)
    embeddings = HuggingFaceEmbeddings(
        model_name="microsoft/unixcoder-base",
        model_kwargs={'device': 'cpu'}
    )
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)


if __name__ == '__main__':
    create_vector_db()
