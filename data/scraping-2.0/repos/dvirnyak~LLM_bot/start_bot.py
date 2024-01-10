import json

from bot import bot
from config import db_engine, db_meta, Session
import sqlalchemy as db
from models import *
import os
from docarray import BaseDoc
from docarray.typing import NdArray
from langchain.embeddings import GPT4AllEmbeddings
from langchain.retrievers import DocArrayRetriever
embeddings = GPT4AllEmbeddings()

def prepare_document_db():
    session = Session()
    documents_dir = 'documents/'
    for filename in os.listdir(documents_dir):
        if filename.endswith('.txt'):
            with open(documents_dir + filename, 'r') as file:
                text = file.read()
                if not(text is None or len(text) == 0):
                    embedding = embeddings.embed_query(text=text)
                    doc = Document(text=text, text_embedding=json.dumps(embedding))
                    doc.save(session)


if __name__ == "__main__":
    db.MetaData.reflect(db_meta, bind=db_engine)
    Base.metadata.create_all(bind=db_engine)

    prepare_document_db()

    bot()