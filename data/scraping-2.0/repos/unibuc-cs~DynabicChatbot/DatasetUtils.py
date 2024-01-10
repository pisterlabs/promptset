"""Drops a collection from the document storage."""
import os

import etl.pdfs
import projsecrets

import json
from pathlib import Path
import pprint
import pdb

from etl import markdown, pdfs, shared, videos

import docstore
import vecstore

from utils import pretty_log

pp = pprint.PrettyPrinter(indent=2)


def prep_documents_for_vector_storage(documents):
    """Prepare documents from document store for embedding and vector storage.

    Documents are split into chunks so that they can be used with sourced Q&A.

    Arguments:
        documents: A list of LangChain.Documents with text, metadata, and a hash ID.
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=100, allowed_special="all"
    )
    ids, texts, metadatas = [], [], []
    for document in documents:
        text, metadata = document["text"], document["metadata"]
        doc_texts = text_splitter.split_text(text)
        doc_metadatas = [metadata] * len(doc_texts)
        ids += [metadata.get("sha256")] * len(doc_texts)
        texts += doc_texts
        metadatas += doc_metadatas

    return ids, texts, metadatas



# TODO: use 8 CPUs here use parallelism !!!
def create_vector_index(collection: str = None, db: str = None):
    """Creates a vector index for a collection in the document database."""
    import docstore

    pretty_log("connecting to document store")
    db = docstore.get_database(db)
    pretty_log(f"connected to database {db.name}")

    collection = docstore.get_collection(collection, db)
    pretty_log(f"collecting documents from {collection.name}")
    docs = docstore.get_documents(collection, db)

    pretty_log("splitting into bite-size chunks")
    ids, texts, metadatas = prep_documents_for_vector_storage(docs)

    pretty_log(f"sending to vector index {vecstore.INDEX_NAME}")
    embedding_engine = vecstore.get_embedding_engine(disallowed_special=())
    vector_index = vecstore.create_vector_index(
        vecstore.INDEX_NAME, embedding_engine, texts, metadatas
    )
    vector_index.save_local(folder_path=vecstore.VECTOR_DIR, index_name=vecstore.INDEX_NAME)
    pretty_log(f"vector index {vecstore.INDEX_NAME} created")

def solve_vector_storage():
    VECTOR_DIR = vecstore.VECTOR_DIR
    vector_storage = "vector-vol"

    create_vector_index(os.environ["MONGODB_COLLECTION"], os.environ["MONGODB_DATABASE"])


def drop_collection():
    docstore.drop(os.environ["MONGODB_COLLECTION"], os.environ["MONGODB_DATABASE"], os.environ["MONGODB_CLIENT"])


def create_knowledge_database():
    drop_collection()

    etl.pdfs.transform_papers_to_json(Path("data") / "pdfpapers.json")
    etl.markdown.main(Path("data") / "webcontent.json")
    etl.videos.main(Path("data") / "videos.json")

    solve_vector_storage()

def __main__():
    create_knowledge_database()

if __name__ == "__main__":
    __main__()
