import os
import sys
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(MAIN_DIR)

DB_PATH = os.environ.get('DB_PATH')
DOCS_PATH = os.environ.get('DOCS_PATH')
EMBEDDINGS_PATH = os.environ.get('EMBEDDINGS_PATH')
EMBEDDINGS_MODEL = os.environ.get('EMBEDDINGS_MODEL')
CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE'))
CHUNK_OVERLAP = int(os.environ.get('CHUNK_OVERLAP'))


def setup_knowledge_base(docs_dir, db_path, embeddings_model):
    loader = DirectoryLoader(docs_dir, glob="**/*.txt")
    docs = loader.load()
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=embeddings_model,
        cache_folder=EMBEDDINGS_PATH
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = text_splitter.split_documents(docs)
    _ = Chroma.from_documents(texts, embeddings, persist_directory=db_path)


if __name__ == "__main__":
    setup_knowledge_base(
        os.path.join(MAIN_DIR, DOCS_PATH),
        os.path.join(MAIN_DIR, DB_PATH),
        EMBEDDINGS_MODEL
    )