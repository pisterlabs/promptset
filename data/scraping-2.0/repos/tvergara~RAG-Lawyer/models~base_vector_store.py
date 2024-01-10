from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import os
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

PERSIST_DIRECTORY = "./models/chroma_db"


def get_vector_store(model_name: str, name: str, pdfs: list[str]):
    embeddings = SentenceTransformerEmbeddings(model_name=model_name)

    if os.path.exists(PERSIST_DIRECTORY):
        store = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
            collection_name=name
        )
        return store

    pages = []
    for pdf in pdfs:
        loader = PyPDFLoader(pdf)
        pages += loader.load_and_split()

    store = Chroma.from_documents(
        pages,
        embeddings,
        collection_name=name,
        persist_directory=PERSIST_DIRECTORY
    )
    return store

if __name__ == "__main__":
    model_name = "thenlper/gte-base"
    store = get_vector_store(
        model_name=model_name,
        name="basic-laws",
        pdfs=['./data/basic-laws-book-2016.pdf', './data/european-human-rights.pdf'])
    print(store.similarity_search("europe human rights", k=5))
