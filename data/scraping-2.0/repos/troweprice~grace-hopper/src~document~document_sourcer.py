import os
import chromadb

from src.document.chroma_documents import ChromaDocuments
from langchain.text_splitter import RecursiveCharacterTextSplitter

COLLECTION_NAME = "all-documents"
DOCUMENTS_PATH = 'src/document/saved_documents'


class DocumentSourcer:

    def get_preloaded_vector_database(self) -> chromadb.API:
        """
        Create an in-memory vector db and load initial documents.
        """
        # Setup Chroma in-memory
        client = chromadb.Client()

        # Create a single collection for all our documents
        collection = client.create_collection(COLLECTION_NAME)

        # Get chunked document content and metadata from specified location
        chroma_documents = self._get_documents(DOCUMENTS_PATH)

        # Add document chunks to the collection
        collection.add(
            documents=chroma_documents.document_contents,
            metadatas=chroma_documents.metadata,
            ids=chroma_documents.ids,
        )
        return client

    def _get_documents(self, path_to_documents: str) -> ChromaDocuments:
        """
        Loads all documents from given directory, formats metadata for chromadb.
        """
        full_document_contents = []
        metadata = []
        for file_name in os.listdir(path_to_documents):
            with open(os.path.join(path_to_documents, file_name), 'r') as file:
                content = file.read()
                full_document_contents.append(content)
                metadata.append({'source_file': file_name})

        chroma_documents = self._get_documents_as_chunks(full_document_contents, metadata)
        return chroma_documents

    def _get_documents_as_chunks(self, document_contents: [], document_metadata: [], chunk_size=350,
                                 chunk_overlap=20) -> ChromaDocuments:
        """
        Split the documents into chunks.
        """

        # Use the LangChain text splitter to convert documents to chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        documents = text_splitter.create_documents(document_contents, document_metadata)
        chunked_documents = text_splitter.split_documents(documents)

        # Extract the document chunks and metadata from the LangChain document object
        document_contents = []
        metadata = []
        document_ids = []
        for index, chunk in enumerate(chunked_documents):
            document_contents.append(chunk.page_content)
            metadata.append(chunk.metadata)
            document_ids.append(str(index))
        return ChromaDocuments(document_contents, metadata, document_ids)
