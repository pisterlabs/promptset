import os
from server.utils.parse import ProcessedDocumentReturnObject
from server.services.loaders import LoaderError
from server.utils.parse import DocumentsObjectInterface
from server.services.vector_store import VectorStore
from langchain.docstore.document import Document
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from server.services.loaders import LoaderResult
from chromadb.api.models.Collection import Collection
from typing import List, Dict


class DocumentReturnObject:
    def __init__(
        self,
        documents: Dict[str, list[ProcessedDocumentReturnObject]],
        errors: List[LoaderError],
    ):
        self.documents = documents
        self.errors = errors


class DocumentService:
    vector_store = VectorStore()

    def get_documents(
        self, collection_name
    ) -> Dict[str, list[ProcessedDocumentReturnObject]]:
        collection: Collection = DocumentService.vector_store.get_collection(
            collection_name
        )

        documents = DocumentService.vector_store.get_documents(collection)

        return documents

    def add_documents(self, collection_name: str, documents: List[FileStorage]):
        results: List[LoaderResult] = []
        errors: List[LoaderError] = []
        save_temp_folder = "server/save_temp_files"

        for file in documents:
            if file.filename:
                filename = secure_filename(file.filename)
                file_path = os.path.join(save_temp_folder, filename)
                file.save(file_path)

                results.append(
                    DocumentService.vector_store.add_documents(
                        collection_name, file_path, filename
                    )
                )
                os.remove(file_path)

        for result in results:
            errors.extend(result.errors)

        collection: Collection = DocumentService.vector_store.get_collection(
            collection_name
        )

        returned_documents: Dict[
            str, list[ProcessedDocumentReturnObject]
        ] = DocumentService.vector_store.get_documents(collection)

        return DocumentReturnObject(documents=returned_documents, errors=errors)

    def delete_documents(self, collection_name: str, filename: str):
        return DocumentService.vector_store.delete_documents(collection_name, filename)
