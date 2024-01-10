import hashlib
from typing import List
from cat.log import log
from cat.mad_hatter.decorators import hook, plugin
from pydantic import BaseModel, Field
from langchain.docstore.document import Document
from sqlalchemy import ForeignKey, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, Session


DEFAULT_SQLITE_FILEPATH = 'sqlite:///cat/data/dietician.db'


class Base(DeclarativeBase):
    pass


class DietDocument(Base):
    __tablename__= 'document'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(256), unique=True)
    hash: Mapped[str] = mapped_column(String(64), unique=True)
    
    chunks: Mapped[List["Chunk"]] = relationship(back_populates="document")


    def __repr__(self) -> str:
        return f'DietDocument(name={self.name!r}, hash={self.hash!r})'


class Chunk(Base):
    __tablename__ = 'chunk'
    id: Mapped[int] = mapped_column(primary_key=True)
    chunk_count: Mapped[int]
    document_id: Mapped[int] = mapped_column(ForeignKey("document.id"))

    document: Mapped["DietDocument"] = relationship(back_populates="chunks")

    def __repr__(self) -> str:
        return f'Chunk(chunk_count={self.chunk_count!r})'


# sqlalchemy sqlite engine
engine = None
# Base.metadata.create_all(engine, checkfirst=True)
# log.warning(f"Dietician is using a sqlite file located here: {DEFAULT_SQLITE_FILEPATH}. You can change the path in the plugin settings.")



class PluginSettings(BaseModel):
    sqlite_db_path: str = Field(
        default=DEFAULT_SQLITE_FILEPATH,
        title="Sqlite filepath. Change it only if you know what you are doing!",
    )


@plugin
def settings_model():
    return PluginSettings


@hook(priority=10)
def before_rabbithole_splits_text(doc, cat):
    # doc is a list with only one element, always
    cat.working_memory['ccat-dietician'] = {
        'name': doc[0].metadata['source'],
        'hash': hashlib.sha256(doc[0].page_content.encode()).hexdigest()
    }

    global engine
    db_filepath = cat.mad_hatter.get_plugin().load_settings()["sqlite_db_path"]
    engine = create_engine(db_filepath)
    Base.metadata.create_all(engine, checkfirst=True)
    log.warning(f"Dietician is writing on the sqlite db located here: {db_filepath}. You can change the path in the plugin settings.")


    return doc


@hook(priority=10)
def before_rabbithole_stores_documents(docs: List[Document], cat) -> List[Document]:
    cat.working_memory['ccat-dietician']['chunk_count'] = len(docs)

    with Session(engine) as session:
        try:
            doc_by_name = session.query(DietDocument).filter_by(name=cat.working_memory['ccat-dietician']['name']).first()
            if doc_by_name is None:
                doc_by_hash = session.query(DietDocument).filter_by(hash=cat.working_memory['ccat-dietician']['hash']).first()
                if doc_by_hash is None:
                        db_doc = DietDocument(name=cat.working_memory['ccat-dietician']['name'], hash=cat.working_memory['ccat-dietician']['hash'], chunks=[Chunk(chunk_count=cat.working_memory['ccat-dietician']['chunk_count'])])
                        session.add(db_doc)
                        session.commit()
                        log.info(f"Dietician is allowing the ingestion of a new document: {db_doc}")
                        return docs
                else:
                    if cat.working_memory['ccat-dietician']['chunk_count'] in [c.chunk_count for c in doc_by_hash.chunks]:
                        log.info(f"Dietician detected {cat.working_memory['ccat-dietician']['name']} as a duplicate of {doc_by_hash.name}, since the number of chunks ({cat.working_memory['ccat-dietician']['chunk_count']}) coincides to what is already in declarative memory, this ingestion is going to be avoided.")
                        return []
                    else:
                        doc_by_hash.chunks.append(Chunk(chunk_count=cat.working_memory['ccat-dietician']['chunk_count']))
                        session.add(doc_by_hash)
                        session.commit()
                        log.info(f"Dietician detected {cat.working_memory['ccat-dietician']['name']} as a duplicate of {doc_by_hash.name}, since the number of chunks ({cat.working_memory['ccat-dietician']['chunk_count']}) produced now is different from what is already in declarative memory, this ingestion is going to be allowed.")
                        return docs
            else:
                if cat.working_memory['ccat-dietician']['hash'] == doc_by_name.hash:
                    if cat.working_memory['ccat-dietician']['chunk_count'] in [c.chunk_count for c in doc_by_name.chunks]:
                        log.info(f"Dietician detected that {doc_by_name.name} was already ingested, since the number of chunks ({cat.working_memory['ccat-dietician']['chunk_count']}) coincides to what is already in declarative memory, this ingestion is going to be avoided.")
                        return []
                    else:
                        doc_by_name.chunks.append(Chunk(chunk_count=cat.working_memory['ccat-dietician']['chunk_count']))
                        session.add(doc_by_name)
                        session.commit()
                        log.info(f"Dietician detected that {doc_by_name.name} was already ingested, since the number of chunks ({cat.working_memory['ccat-dietician']['chunk_count']}) produced now is different from what is already in declarative memory, this ingestion is going to be allowed.")
                        return docs
                else:
                    old_chunks, _ = cat.memory.vectors.declarative.client.scroll(
                        collection_name=cat.memory.vectors.declarative.collection_name,
                        scroll_filter=cat.memory.vectors.declarative._qdrant_filter_from_dict({'source': doc_by_name.name}),
                        with_payload=True
                    )
                    old_chunks_text = [c.payload['page_content'] for c in old_chunks]
                    new_chunks_text = [d.page_content for d in docs]

                    # we have to delete all chunks in declarative memory that are not in the new document because those chunks are related an old version of the document
                    old_chunks_to_delete_ids = [c.id for c in old_chunks if c.payload['page_content'] not in new_chunks_text]

                    if len(old_chunks_to_delete_ids) > 0:
                        cat.memory.vectors.declarative.delete_points(old_chunks_to_delete_ids)

                    log.info(f"Dietician detected an hash change for the document {doc_by_name}, this means that the document has beed updated. Allowing the ingestion of new chunks and deleting all the old chunks not any more present in the current document version.")
                    
                    # docs contain only chunks never inserted in declarative memory, we keep into the vectordb any chunk previously inserted (to avoid unnecessary calls to the embedding model)
                    return [d for d in docs if d.page_content not in old_chunks_text]

        except Exception as e:
            session.rollback()
            log.error(f"Something weird happened: {str(e)}. Dietician is preventing the ingestion of {cat.working_memory['ccat-dietician']['name']}")
            return []

