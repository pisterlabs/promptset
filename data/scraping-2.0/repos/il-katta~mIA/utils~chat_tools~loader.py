import json
from typing import Optional, Dict, List, Union

import faiss
from langchain.cache import SQLiteCache
from langchain.docstore.base import Docstore, AddableMixin
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.schema.embeddings import Embeddings
from langchain.vectorstores import Chroma, FAISS
from pydantic.v1 import PrivateAttr
from sqlalchemy import Column, String, Engine, create_engine, select
from sqlalchemy.orm import declarative_base, Session

import config


def load_chroma_vectorstore(embedding: Optional[Embeddings] = None) -> Chroma:
    if embedding is None:
        embedding = OpenAIEmbeddings()
    persist_directory = config.DATA_DIR / "vectorstore" / "uploaded_files"
    if not persist_directory.exists():
        persist_directory.mkdir(parents=True)
    return Chroma(
        embedding_function=embedding,
        collection_name="uploaded_files",
        persist_directory=str(persist_directory)
    )


Base = declarative_base()


class DocStoreModel(Base):  # type: ignore
    __tablename__ = "my_docstore"
    key = Column(String, primary_key=True)
    document = Column(String)


class MyDockstore(Docstore, AddableMixin):
    _cache: SQLiteCache = PrivateAttr()

    def __init__(self, engine: Optional[Engine] = None):
        if engine is None:
            dbpath = config.DATA_DIR / "docstore" / "mydocstore.db"
            if not dbpath.parent.exists():
                dbpath.parent.mkdir(parents=True)
            engine = create_engine(f"sqlite:///{dbpath}")
        self.engine = engine
        DocStoreModel.metadata.create_all(self.engine)

    def add(self, texts: Dict[str, Document]) -> None:
        items = [
            DocStoreModel(key=key, document=json.dumps(doc.dict()))
            for key, doc in texts.items()
        ]
        with Session(self.engine) as session, session.begin():
            for item in items:
                session.merge(item)
            session.commit()

    def delete(self, ids: List) -> None:
        with Session(self.engine) as session:
            for _id in ids:
                session.query(DocStoreModel).filter_by(key=_id).delete()
            session.commit()

    def search(self, search: str) -> Union[str, Document]:
        stmt = (
            select(DocStoreModel.key).select(DocStoreModel.document)
            .where(DocStoreModel.key == search)  # type: ignore
            .limit(1)
        )
        with Session(self.engine) as session:
            result = session.execute(stmt).first()
        if result is None:
            return f"ID {search} not found."
        else:
            return Document(**json.loads(result[1]))

    def get_keys(self) -> List[str]:
        stmt = select(DocStoreModel.key)
        with Session(self.engine) as session:
            result = session.execute(stmt).fetchall()
        return [r[0] for r in result]

    def __len__(self):
        with Session(self.engine) as session:
            return session.query(DocStoreModel).count()


EMBEDDING_SIZE = 1536


def load_faiss_vectorstore(embedding: Optional[Embeddings] = None) -> FAISS:
    if embedding is None:
        embedding = OpenAIEmbeddings()
    docstore = MyDockstore()
    return FAISS(
        embedding_function=embedding.embed_query,
        index=faiss.IndexFlatL2(EMBEDDING_SIZE),
        docstore=docstore,
        index_to_docstore_id={
            i: key
            for i, key in enumerate(docstore.get_keys())
        },
    )
