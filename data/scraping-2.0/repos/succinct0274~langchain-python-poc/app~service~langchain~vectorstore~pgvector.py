# from typing import Optional, Tuple

# import sqlalchemy
# from typing import Dict
# from langchain.vectorstores.pgvector import PGVector, DistanceStrategy, BaseModel
# from pgvector.sqlalchemy import Vector
# from sqlalchemy.dialects.postgresql import JSON, UUID
# from sqlalchemy.orm import Session, relationship

# from langchain.vectorstores.pgvector import BaseModel
# from langchain.vectorstores._pgvector_data_models import CollectionStore, EmbeddingStore

# class CollectionStoreQueryByMetadata(CollectionStore):
#     """Collection store."""

#     __tablename__ = "langchain_pg_collection"
#     __table_args__ = {'extend_existing': True} 

#     # name = sqlalchemy.Column(sqlalchemy.String)
#     # cmetadata = sqlalchemy.Column(JSON)

#     embeddings = relationship(
#         "EmbeddingStore",
#         back_populates="collection",
#         passive_deletes=True,
#     )

#     @classmethod
#     def get_by_name(cls, session: Session, name: str) -> Optional["CollectionStoreQueryByMetadata"]:
#         return session.query(cls).filter(cls.name == name).first()  # type: ignore
    
#     @classmethod
#     def get_by_name_and_conversation_id(cls, session: Session, name: str, cmetadata: dict | None=None) -> Optional["CollectionStoreQueryByMetadata"]:
#         if cmetadata is None or 'conversation_id' not in cmetadata:
#             return cls.get_by_name(session, name)
        
#         conversation_id = cmetadata['conversation_id']
#         return session.query(cls).filter(cls.name == name, cls.cmetadata.op('->>')('conversation_id').cast(sqlalchemy.String) == conversation_id).first()

#     @classmethod
#     def get_or_create(
#         cls,
#         session: Session,
#         name: str,
#         cmetadata: Optional[dict] = None,
#     ) -> Tuple["CollectionStoreQueryByMetadata", bool]:
#         """
#         Get or create a collection.
#         Returns [Collection, bool] where the bool is True if the collection was created.
#         """
#         created = False
#         collection = cls.get_by_name_and_conversation_id(session, name, cmetadata)
#         if collection:
#             return collection, created

#         collection = cls(name=name, cmetadata=cmetadata)
#         session.add(collection)
#         session.commit()
#         created = True
#         return collection, created

# class PGVectorWithMetadata(PGVector):
    
#     def __post_init__(self) -> None:
#         self.create_vector_extension()
#         self.CollectionStore = CollectionStoreQueryByMetadata
#         self.EmbeddingStore = EmbeddingStore
#         self.create_tables_if_not_exists()
#         self.create_collection()
