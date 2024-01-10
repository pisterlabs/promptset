import hashlib
from typing import List, Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import numpy as np
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from database import SessionLocal
from modules.conversations.services.llm_connector import get_embeddings
from modules.conversations.services.conversation_service import ConversationService
from modules.conversations.models import ConversationModel
from ..models import *
from ..utils import *


class DocumentManager:
    def __init__(self, db=None, conversation_id=None):
        self.db = db if db else SessionLocal()
        self.conversation_id = conversation_id
        self.faiss_path = None
        self.faiss_index = None
        self.load_faiss_index()

    def load_faiss_index(self):
        current_path = os.path.dirname(os.path.realpath(__file__))
        data_path = os.path.join(current_path, "..", "..", "..", "data")
        self.faiss_path = os.path.join(data_path, "faiss_index.bin")

        if os.path.exists(self.faiss_path):
            self.faiss_index = faiss.read_index(self.faiss_path)
        else:
            self.faiss_index = self.create_faiss_index()
            self.save_index_to_disk()

    def create_faiss_index(self):
        index = faiss.IndexFlatL2(1536)
        return index

    def save_index_to_disk(self):
        faiss.write_index(self.faiss_index, self.faiss_path)

    """
    Document processing methods
    """

    async def save_documents(self, documents, split=True, chunk_size=1500):
        document_service = DocumentService(self.db)
        embeddings = []
        metadatas = []
        print(f"Saving {len(documents)} documents")
        for document in documents:
            if isinstance(document, Document):
                text = document.page_content
                document_metadata = document.metadata
            else:
                text = document["text"]
                document_metadata = document["metadata"]

            if not text:
                continue

            if split:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size, chunk_overlap=0
                )
                text_chunks = text_splitter.split_text(text)
            else:
                text_chunks = [text]

            for text_chunk in text_chunks:
                chunk_hash_id = str(hashlib.sha256(text_chunk.encode()).hexdigest())
                print("Creating embeddings..")
                embedding = get_embeddings(text_chunk)

                # Store the embedding and metadata
                embeddings.append(embedding)
                metadata = {
                    "id": chunk_hash_id,
                    "type": document_metadata.get("document_type"),
                    "conversation_id": self.conversation_id,
                    "collection": document_metadata.get("collection"),
                    "title": document_metadata.get("title"),
                    "content": text_chunk,
                    "source": document_metadata.get("source"),
                    "document_key": document_metadata.get("document_key"),
                }
                metadatas.append(metadata)

        # Add embeddings to the Faiss index
        if embeddings:
            self.faiss_index.add(np.array(embeddings, dtype=np.float32))
        else:
            raise Exception("No usuable text extracted!")

        # Save the Faiss index to disk
        self.save_index_to_disk()

        for i, metadata in enumerate(metadatas):
            doc_chunk = document_service.create_document(
                metadata, self.faiss_index.ntotal - len(metadatas) + i
            )

            # Associate the document with the conversation
            conversation = ConversationService(self.db).get_conversation_by_id(
                self.conversation_id
            )

            if conversation:
                document_service.add_document_to_conversation(conversation, doc_chunk)

    """
    Loader methods
    """

    async def load_url(self, url):
        docs = await url_loader([url])
        await self.save_documents(docs)

    async def load_files(self, files):
        docs = []

        for file in files:
            print(f"{file.filename} => {file.content_type}")
            if file.content_type == "application/pdf":
                docs = await pdf_loader([file])
            elif file.content_type == "text/plain":
                docs = await text_loader([file])
            elif file.content_type == "application/json":
                docs = await text_loader([file])
            elif file.content_type == "text/csv":
                pass
            elif (
                file.content_type
                == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ):
                pass
            elif (
                file.content_type
                == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ):
                pass

        await self.save_documents(docs)

    """
    Search & document discovery
    """

    def similar_search(self, text, results=10, recent=False, conversation_filter=True):
        embeddings = get_embeddings(text)
        search_vector = np.array([embeddings], dtype=np.float32)

        # Query the Faiss index to find the most similar embeddings, and get distances
        distances, index_positions = self.faiss_index.search(search_vector, results)

        # Retrieve the associated metadata for the most similar embeddings from the database
        found_documents = DocumentService(self.db).get_documents_by_faiss_indices(
            index_positions[0],
            self.conversation_id if conversation_filter else None,
        )

        if recent:
            found_documents.sort(key=lambda x: x.timestamp, reverse=True)

        # Serialize the results and add similarity scores
        results = [
            {**doc.serialize(), "similarity_score": 1 / (1 + float(dist))}
            for doc, dist in zip(found_documents, distances[0])
        ]

        return results

    def get_documents(self, conversation_only=False):
        """
        Get a list of available documents for management on frontend
        """
        if conversation_only:
            conversation = ConversationService(self.db).get_conversation_by_id(
                self.conversation_id
            )
            if conversation:
                document_chunks = conversation.documents
            else:
                document_chunks = []
        else:
            document_chunks = DocumentService(self.db).get_all_documents()

        output = []
        seen_document_keys = set()

        for doc in document_chunks:
            # Fetch the associated conversation IDs
            conversation_ids = [c.id for c in doc.conversations]

            document_data = {
                "id": doc.id,
                "conversation_ids": conversation_ids,
                "document_key": doc.document_key,
                "title": doc.title,
                "source": doc.source,
                "type": doc.type,
            }

            if doc.document_key not in seen_document_keys:
                output.append(document_data)
                seen_document_keys.add(doc.document_key)

        return output

    """
    Utility methods
    """

    def add_conversation_id(self, document_key):
        documents = DocumentService(self.db).get_documents_by_key(
            document_key=document_key
        )

        if documents:
            conversation = ConversationService(self.db).get_conversation_by_id(
                self.conversation_id
            )

            if conversation:
                for document in documents:
                    DocumentService(self.db).add_document_to_conversation(
                        conversation, document
                    )

    def remove_conversation_id(self, document_key):
        documents = DocumentService(self.db).get_documents_by_key(
            document_key=document_key
        )

        if documents:
            conversation = ConversationService(self.db).get_conversation_by_id(
                self.conversation_id
            )

            if conversation:
                for document in documents:
                    DocumentService(self.db).remove_document_from_conversation(
                        conversation, document
                    )

    def delete_documents(self, document_key):
        # Get all documents with the given document key
        documents_to_delete = DocumentService(self.db).get_documents_by_key(
            document_key=document_key
        )

        # Remove the corresponding documents from the Faiss index
        faiss_indexes_to_remove = [doc.faiss_index for doc in documents_to_delete]
        self.faiss_index.remove_ids(np.array(faiss_indexes_to_remove, dtype=np.int64))

        # Delete the filtered documents from database
        DocumentService(self.db).delete_documents_by_key(document_key=document_key)


class DocumentService:
    def __init__(self, db: Session):
        self.db = db

    def create_document(self, metadata: dict, faiss_index: int):
        try:
            doc_model = DocumentModel(
                chunk_hash_id=metadata["id"],
                faiss_index=faiss_index,
                type=metadata["type"],
                collection=metadata["collection"],
                title=metadata["title"],
                content=metadata["content"],
                source=metadata["source"],
                document_key=metadata["document_key"],
            )
            self.db.add(doc_model)
            self.db.commit()
            return doc_model
        except IntegrityError:
            print(
                f'Document with chunk_hash_id {metadata["id"]} already exists. Skipping.'
            )
            self.db.rollback()

    def get_all_documents(self):
        return self.db.query(DocumentModel).all()

    def get_documents_by_faiss_indices(
        self, faiss_indices: List[int], conversation_id: Optional[int] = None
    ):
        faiss_indices = [int(index) for index in faiss_indices if index != -1]

        base_query = self.db.query(DocumentModel).filter(
            DocumentModel.faiss_index.in_(faiss_indices)
        )

        if conversation_id is not None:
            base_query = base_query.join(
                ConversationModel, DocumentModel.conversations
            ).filter(ConversationModel.id == conversation_id)

        return base_query.all()

    def get_documents_by_key(self, document_key):
        return (
            self.db.query(DocumentModel)
            .filter(DocumentModel.document_key == document_key)
            .all()
        )

    def delete_documents_by_key(self, document_key: str):
        documents = (
            self.db.query(DocumentModel)
            .filter(DocumentModel.document_key == document_key)
            .all()
        )
        for document in documents:
            self.db.delete(document)
        self.db.commit()

    def add_document_to_conversation(
        self, conversation: ConversationModel, document: DocumentModel
    ):
        conversation.documents.append(document)
        self.db.commit()

    def remove_document_from_conversation(
        self, conversation: ConversationModel, document: DocumentModel
    ):
        conversation.documents.remove(document)
        self.db.commit()
