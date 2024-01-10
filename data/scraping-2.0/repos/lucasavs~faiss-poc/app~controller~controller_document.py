from ..databases_holder import database_container, embeddings
from langchain.vectorstores.faiss import FAISS
from langchain.schema import Document
from fastapi import HTTPException


def _check_database_status(database_name: str):  # TODO turn this into a decorator
    if database_name not in database_container:
        raise HTTPException(status_code=404, detail="knowledge base not found")


def add_document(database_name: str, payload: str):
    _check_database_status(database_name)
    doc = Document(page_content=payload, metadata={"source": "local"})
    if database_container[database_name] is None:
        db = FAISS.from_documents(
            [doc], embeddings
        )  # TODO find a way to create an empty db
        database_container[database_name] = db
    else:
        database_container[database_name].add_documents([doc])


def delete_document(database_name: str, document_index: str):
    _check_database_status(database_name)
    db = database_container[database_name]
    try:
        db.delete([document_index])
    except (ValueError):
        raise HTTPException(status_code=404, detail="Value does not exist on database")
