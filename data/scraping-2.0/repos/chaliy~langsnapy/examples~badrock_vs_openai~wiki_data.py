import logging
import sys
import pickle
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

LOCAL_DATA_PATH = Path(__file__).parent / "local-data"
LOCAL_DATA_PATH.mkdir(parents=True, exist_ok=True)

def load_wiki_docs() -> list[Document]:
    wiki_docs_path = LOCAL_DATA_PATH / "wiki-docs.pkl"

    if wiki_docs_path.exists():
        with open(wiki_docs_path, "rb") as f:
            wiki_docs = pickle.load(f)
    else:
        from langchain.document_loaders.wikipedia import WikipediaLoader

        logging.info("⏩ Loading information about Ukraine from Wikipedia")

        wiki_docs = [
            doc 
            for lang in ["uk", "en"]
            for doc in WikipediaLoader(query="Ukraine", lang=lang, load_max_docs=50).load()
        ]

        with open(wiki_docs_path, "wb") as f:
            pickle.dump(wiki_docs, f)

        logging.info("✅ Information about Ukraine from Wikipedia - Loaded")
        
    return wiki_docs

def get_openai_embeddings() -> Embeddings:
    from langchain.embeddings.openai import OpenAIEmbeddings

    return OpenAIEmbeddings()


def load_wiki_docs_faiss() -> VectorStore:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.vectorstores.faiss import FAISS

    embeddings = get_openai_embeddings()

    wiki_docs_faiss_path = LOCAL_DATA_PATH / "wiki-docs-faiss"

    if wiki_docs_faiss_path.exists():
        wiki_docs_faiss = FAISS.load_local(
            wiki_docs_faiss_path, 
            embeddings=embeddings
        )
    else:
        wiki_docs = load_wiki_docs()
        logging.info("⏩ Creating FAISS from Wikipedia documents")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=0
        )
        wiki_docs_chunked = text_splitter.split_documents(wiki_docs)

        wiki_docs_faiss = FAISS.from_documents(
            embedding=embeddings, documents=wiki_docs_chunked
        )

        wiki_docs_faiss.save_local(wiki_docs_faiss_path)
        logging.info("✅ FAISS from Wikipedia documents - Created")

    return wiki_docs_faiss
